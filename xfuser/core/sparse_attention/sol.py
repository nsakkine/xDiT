"""Sol-Attn: block-sparse fp8 attention with a pooled-KV correction for the blocks it skips.

Plain block-sparse attention routes each query tile to a subset of KV blocks and DROPS the rest.
Sol-Attn (arXiv 2607.24027) computes the same selected blocks exactly and additionally recovers the
contribution of the skipped blocks from pooled (mean) K/V, so the mass outside the selection is
approximated instead of discarded.
"""
from types import SimpleNamespace

import torch

_SUPPORTED_ARCH = "gfx950"


class SolAttnUnsupported(RuntimeError):
    """Raised when the inputs or the environment cannot be served correctly by the Sol-Attn kernel."""


def _probe_aiter():
    """aiter's Sol-Attn entry points, or None when this build does not ship them.

    Resolved at import. This module is imported lazily, only once the backend is selected, so builds
    that never touch Sol-Attn pay nothing; binding the entry points once also keeps repeated imports
    out of any traced graph. native_fp8_format is held as a function rather than called here because
    it queries the device, which import time is too early for.
    """
    try:
        from aiter.ops.mha_v4 import (
            AttentionScaleMode,
            mha_v4_sol_attn,
            mha_v4_sol_attn_packed,
            native_fp8_format,
            quantize_fp8,
        )
        from aiter.ops.triton.attention.utils import (
            SOL_ATTN_TS_KV,
            SOL_ATTN_TS_QO,
            sol_attn_prepare,
        )
    except ImportError:
        return None
    return SimpleNamespace(
        sol_attn=mha_v4_sol_attn,
        sol_attn_packed=mha_v4_sol_attn_packed,
        quantize_fp8=quantize_fp8,
        prepare=sol_attn_prepare,
        native_fp8_format=native_fp8_format,
        per_tensor_scale=AttentionScaleMode.F32_PER_TENSOR,
        ts_qo=SOL_ATTN_TS_QO,
        ts_kv=SOL_ATTN_TS_KV,
    )


_AITER = _probe_aiter()
SOL_ATTN_AVAILABLE = _AITER is not None


def _probe_hadamard():
    """(rotate fn, per-device matrix) for the fp8 rotation AITER_FP8 uses, or (None, None).

    attention_backend builds the matrix table at its own import and has no module-level edge back
    here, so resolving this at module scope is safe and keeps the lookup out of any traced graph.
    """
    try:
        from xfuser.core.distributed.attention_backend import (
            FP8_HADAMARD_MATRIX,
            _fp8_hadamard_rotate,
        )
    except ImportError:
        return None, None
    return _fp8_hadamard_rotate, FP8_HADAMARD_MATRIX


_HADAMARD_ROTATE, _HADAMARD_MATRIX = _probe_hadamard()


def check_sol_attn_device(device=None):
    """Raise unless this build and device can run Sol-Attn. Defaults to the current CUDA device.

    Both facts are fixed for the process, so runtime_state calls this once during backend setup and
    the per-call path does not repeat it. Callers reaching sol_attn_bhsd directly, outside xDiT's
    setup, should call it themselves.
    """
    if not SOL_ATTN_AVAILABLE:
        raise SolAttnUnsupported(
            "Sol-Attn requires aiter with mha_v4_sol_attn and sol_attn_prepare; "
            "please update AITER")
    if device is None:
        if not torch.cuda.is_available():
            raise SolAttnUnsupported("Sol-Attn is a GPU kernel and no CUDA device is available")
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type != "cuda":
        raise SolAttnUnsupported(f"Sol-Attn is a GPU kernel, got device {device}")
    arch = torch.cuda.get_device_properties(device).gcnArchName or ""
    if not arch.startswith(_SUPPORTED_ARCH):
        raise SolAttnUnsupported(
            f"Sol-Attn ships only a {_SUPPORTED_ARCH} kernel, this device reports '{arch}'")


def check_sol_attn_supported(query, key, value, is_causal, ring_world_size=1):
    """Validate the per-call constraints aiter's Sol-Attn contract does not already cover.
    """
    if not query.dtype == key.dtype == value.dtype == torch.bfloat16:
        raise SolAttnUnsupported(
            f"Sol-Attn's mha_v4 row takes bf16 Q/K/V and returns bf16, got q={query.dtype} "
            f"k={key.dtype} v={value.dtype}. Select another attention backend for other dtypes.")
    if is_causal:
        raise SolAttnUnsupported(
            "Sol-Attn has no causal variant: its pooled correction assumes every skipped block is "
            "fully attendable, which a causal mask breaks. Use AITER_FP8 for causal attention.")
    if ring_world_size > 1:
        raise SolAttnUnsupported(
            "Sol-Attn does not support ring parallelism: merging partial outputs by LSE is not valid "
            "once each rank has added a pooled correction for the blocks it skipped. Use "
            "ulysses_degree for sequence parallelism instead.")


def _hadamard_rotate_qk(query, key):
    """Rotate Q and K by the shared orthonormal Hadamard matrix AITER_FP8 uses, along head_dim.
    """
    r = _HADAMARD_MATRIX.get(query.device) if _HADAMARD_MATRIX is not None else None
    if r is None:
        raise SolAttnUnsupported(
            "No Hadamard matrix could be built for this device. The rotation is required by the "
            "AITER_SOL_FP8 backend rather than optional, so there is no flag to disable it; please "
            "update AITER or select a different attention backend.")
    return _HADAMARD_ROTATE(query, r).contiguous(), _HADAMARD_ROTATE(key, r).contiguous()


def sol_attn_routing_for(q_fp8, k_fp8, v_fp8, beta):
    """Pooled K/V, ragged LUT and selection bitmap for one call, as the kernel consumes them.
    """
    return _AITER.prepare(q_fp8, k_fp8, v_fp8, beta, _AITER.ts_qo, _AITER.ts_kv)


def _head_cost_from_routing(routing):
    """Per-head count of exactly-computed KV blocks, float32 (nheads_q,).
    """
    mask = routing["block_attn_mask"]  # (batch, nheads_q, num_q_tiles, num_kv_blocks) bool
    return mask.sum(dim=(0, 2, 3), dtype=torch.float32)


def _maybe_dump(path, query, key, value):
    """Save one call's BSHD q/k/v so kernel benchmarks can be replayed on real tensors.
    """
    if not path or getattr(_maybe_dump, "_done", False):
        return
    _maybe_dump._done = True
    torch.save({"q": query.detach().cpu(), "k": key.detach().cpu(), "v": value.detach().cpu()}, path)


def sol_attn_bhsd(query, key, value, is_causal=False, beta=1.0, softmax_scale=None,
                  routing=None, ring_world_size=1, dump_path=None, hadamard=True,
                  return_head_cost=False):
    """Sol-Attn over BHSD tensors, returning (BHSD bf16 output, per-head cost or None).

    query/key/value are (batch, nheads, seqlen, head_dim) bf16 tensors. They are permuted into the
    BSHD layout the kernel takes, optionally rotated, and handed to aiter's mha_v4 Sol-Attn
    entrypoint, which owns quantization, routing, format validation and the ASM launch.

    return_head_cost asks for the per-head count of exactly-computed KV blocks, float32 (nheads_q,),
    which the Ulysses head balancer consumes. It is opt-in rather than free: the routing dict it
    reduces is internal to the raw entrypoint, so asking for it moves this call onto aiter's packed
    API and makes quantization and routing explicit here. A caller-supplied `routing` takes the same
    path. Both are traceable; neither graph-breaks.
    """
    if _AITER is None:
        raise SolAttnUnsupported(
            "Sol-Attn requires aiter with mha_v4_sol_attn and sol_attn_prepare; "
            "please update AITER")

    query = query.permute(0, 2, 1, 3).contiguous()
    key = key.permute(0, 2, 1, 3).contiguous()
    value = value.permute(0, 2, 1, 3).contiguous()

    check_sol_attn_supported(query, key, value, is_causal, ring_world_size=ring_world_size)
    _maybe_dump(dump_path, query, key, value)

    # After the dump, so it captures the model's own operands rather than this backend's rotation of
    # them, and before aiter quantizes, which is the whole point of rotating.
    if hadamard:
        query, key = _hadamard_rotate_qk(query, key)

    if routing is None and not return_head_cost:
        out = _AITER.sol_attn(query, key, value, beta, softmax_scale=softmax_scale)
        return out.permute(0, 2, 1, 3), None

    q_fp8, q_descale = _AITER.quantize_fp8(query)
    k_fp8, k_descale = _AITER.quantize_fp8(key)
    v_fp8, v_descale = _AITER.quantize_fp8(value)
    if routing is None:
        routing = sol_attn_routing_for(q_fp8, k_fp8, v_fp8, beta)
    # One fp8 row, so every operand shares the format and a per-tensor scale, and the pooled K/V
    # inherit K's: pooling can reuse a descale only because mean(x) * descale == mean(x * descale),
    # which holds for a per-tensor scale and not for a per-block one.
    fp8, scale = _AITER.native_fp8_format(), _AITER.per_tensor_scale
    out = _AITER.sol_attn_packed(
        q_fp8, k_fp8, v_fp8,
        q_descale, k_descale, v_descale,
        routing["mean_k"], routing["mean_v"],
        routing["kv_block_indices"], routing["lut_start"], routing["lut_count"],
        routing["block_bitmap"],
        fp8, fp8, fp8,
        scale, scale, scale,
        fp8, scale,
        softmax_scale=softmax_scale,
    )
    return out.permute(0, 2, 1, 3), _head_cost_from_routing(routing)


def sol_attn_dump_path():
    """Operand dump path from the environment, or None. See XFUSER_SOL_ATTN_DUMP in xfuser/envs.py.

    beta and hadamard are not read here: beta arrives per call through attention_kwargs from
    --solattn_beta, and the rotation is a property of the selected backend rather than a user knob.
    """
    from xfuser.envs import environment_variables

    return environment_variables["SOL_ATTN_DUMP"]() or None
