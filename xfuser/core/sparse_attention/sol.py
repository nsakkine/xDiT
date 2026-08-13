"""Sol-Attn: block-sparse fp8 attention with a pooled-KV correction for the blocks it skips.

Plain block-sparse attention routes each query tile to a subset of KV blocks and DROPS the rest.
Sol-Attn (arXiv 2607.24027) computes the same selected blocks exactly and additionally recovers the
contribution of the skipped blocks from pooled (mean) K/V, so the mass outside the selection is
approximated instead of discarded.s
"""
import typing

import torch

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
_SUPPORTED_HEAD_DIM = 128
_SUPPORTED_ARCH = "gfx950"


class SolAttnUnsupported(RuntimeError):
    """Raised when the inputs or the environment cannot be served correctly by the Sol-Attn kernel."""


def _is_pow2(n):
    return n > 0 and (n & (n - 1)) == 0


def _misrouted_heads(nheads_q, nheads_kv):
    """Q heads that the kernel's shift-based GQA would send to the wrong KV head."""
    ratio = max(nheads_q // max(nheads_kv, 1), 1)
    shift = max(s for s in range(6) if (1 << s) <= ratio)
    return [h for h in range(nheads_q) if (h >> shift) != h // ratio]


def sol_attn_block_sizes():
    """(BLOCK_M, BLOCK_N) the kernel's routing must be built with, taken from aiter, not hardcoded."""
    from aiter.ops.triton.attention.utils import (
        FMHA_FWD_V3_SOL_ATTN_TS_KV,
        FMHA_FWD_V3_SOL_ATTN_TS_QO,
    )

    return FMHA_FWD_V3_SOL_ATTN_TS_QO, FMHA_FWD_V3_SOL_ATTN_TS_KV


def sol_attn_available():
    """True when the aiter entry points and the gfx950 kernel are importable on this build."""
    try:
        from aiter.ops.mha import fmha_v3_fwd_sol_attn  # noqa: F401
        from aiter.ops.triton.attention.utils import sol_attn_prepare  # noqa: F401
    except ImportError:
        return False
    return True


def check_sol_attn_supported(query, key, value, is_causal, ring_world_size=1):
    """Validate a call against every constraint of the gfx950 Sol-Attn kernel.

    query/key/value are BSHD here (batch, seqlen, nheads, head_dim), i.e. already permuted out of
    xDiT's BHSD convention. Raises SolAttnUnsupported with an actionable message, never returns False,
    because each of these misconfigurations produces wrong output rather than a clean failure.
    """
    if not sol_attn_available():
        raise SolAttnUnsupported(
            "Sol-Attn requires aiter with fmha_v3_fwd_sol_attn and sol_attn_prepare; please update AITER")

    if query.device.type != "cuda":
        raise SolAttnUnsupported(f"Sol-Attn is a GPU kernel, got device {query.device}")
    arch = torch.cuda.get_device_properties(query.device).gcnArchName or ""
    if not arch.startswith(_SUPPORTED_ARCH):
        raise SolAttnUnsupported(
            f"Sol-Attn ships only a {_SUPPORTED_ARCH} kernel, this device reports '{arch}'")

    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise SolAttnUnsupported("Sol-Attn expects 4D query/key/value")

    b, sq, hq, d = query.shape
    bk, sk, hk, dk = key.shape
    bv, sv, hv, dv = value.shape
    if not (b == bk == bv):
        raise SolAttnUnsupported(f"batch mismatch: q={b} k={bk} v={bv}")
    if sk != sv:
        raise SolAttnUnsupported(f"key and value must share seqlen, got k={sk} v={sv}")
    if hk != hv:
        raise SolAttnUnsupported(f"key and value must share head count, got k={hk} v={hv}")
    if not (d == dk == dv == _SUPPORTED_HEAD_DIM):
        raise SolAttnUnsupported(
            f"Sol-Attn is a head_dim={_SUPPORTED_HEAD_DIM} kernel, got q={d} k={dk} v={dv}")
    if hq % hk:
        raise SolAttnUnsupported(f"nheads_q {hq} must be divisible by nheads_kv {hk}")
    if not _is_pow2(hq // hk):
        raise SolAttnUnsupported(
            f"Sol-Attn needs a power-of-2 GQA ratio, got nheads_q={hq} / nheads_kv={hk} = {hq // hk}. "
            f"The kernel finds the KV head by shifting right by floor(log2(ratio)), so q-heads "
            f"{_misrouted_heads(hq, hk)} would read the wrong (often out-of-bounds) KV head, giving "
            f"silently wrong output, NaNs, or a page fault. Use a head count whose ratio is a power of "
            f"two, or select a different attention backend.")
    if is_causal:
        raise SolAttnUnsupported(
            "Sol-Attn has no causal variant: its pooled correction assumes every skipped block is fully "
            "attendable, which a causal mask breaks. Use AITER_FP8 for causal attention.")
    if ring_world_size > 1:
        raise SolAttnUnsupported(
            "Sol-Attn does not support ring parallelism: merging partial outputs by LSE is not valid "
            "once each rank has added a pooled correction for the blocks it skipped. Use ulysses_degree "
            "for sequence parallelism instead.")
    return b, sq, sk, hq, hk, d


def _sol_attn_output_strides(shape):
    """Strides of the BHSD output, which are fixed by the implementation rather than by the caller.
    """
    _, h, s, d = shape
    return (h * s * d, d, h * d, 1)


def _per_tensor_quant_fp8(x):
    """Per-tensor fp8 e4m3 quantization returning (quantized, descale) with a 1-element descale."""
    import aiter

    quant, descale = aiter.per_tensor_quant(
        x, scale=None, quant_dtype=aiter.dtypes.fp8, dtypeMax=torch.finfo(aiter.dtypes.fp8).max)
    return quant, descale.reshape(1).float()


def _hadamard_rotate_qk(query, key):
    """Rotate Q and K by the shared orthonormal Hadamard matrix AITER_FP8 uses, along head_dim.
    """
    from xfuser.core.distributed.attention_backend import (
        FP8_HADAMARD_MATRIX,
        _fp8_hadamard_rotate,
    )

    r = FP8_HADAMARD_MATRIX[query.device]
    if r is None:
        raise SolAttnUnsupported(
            "No Hadamard matrix could be built for this device. The rotation is required by the "
            "AITER_SOL_FP8 backend rather than optional, so there is no flag to disable it; please "
            "update AITER or select a different attention backend.")
    return (_fp8_hadamard_rotate(query, r).contiguous(),
            _fp8_hadamard_rotate(key, r).contiguous())


def sol_attn_routing_for(q_fp8, k_fp8, v_fp8, beta):
    """Pooled K/V, ragged LUT and selection bitmap for one call, as the kernel consumes them."""
    from aiter.ops.triton.attention.utils import sol_attn_prepare

    block_m, block_n = sol_attn_block_sizes()
    return sol_attn_prepare(q_fp8, k_fp8, v_fp8, beta=beta, BLOCK_M=block_m, BLOCK_N=block_n)


def _head_cost_from_routing(prep):
    """Per-head count of exactly-computed KV blocks, float32 (nheads_q,).
    """
    mask = prep["block_attn_mask"]  # (batch, nheads_q, num_q_tiles, num_kv_blocks) bool
    return mask.to(torch.float32).sum(dim=(0, 2, 3))


def _maybe_dump(path, query, key, value):
    """Save one call's BSHD q/k/v so kernel benchmarks can be replayed on real tensors.

    Sparse attention quality is governed almost entirely by how concentrated the attention is, which
    synthetic tensors do not reproduce, so being able to re-run the microbenchmark on captured operands
    is what makes its accuracy numbers trustworthy. Only the first call is written.
    """
    if not path or getattr(_maybe_dump, "_done", False):
        return
    _maybe_dump._done = True
    torch.save({"q": query.detach().cpu(), "k": key.detach().cpu(), "v": value.detach().cpu()}, path)


def _sol_attn_bhsd_eager(query, key, value, is_causal=False, beta=1.0, softmax_scale=None,
                         routing=None, ring_world_size=1, dump_path=None, hadamard=True):
    """Sol-Attn over BHSD tensors, returning (BHSD output in the input dtype, per-head cost).

    query/key/value are (batch, nheads, seqlen, head_dim) high-precision tensors; they are permuted to
    BSHD, quantized per tensor to fp8 e4m3, routed, and handed to the ASM kernel. `routing` may carry a
    precomputed sol_attn_prepare dict to skip routing, in which case it must have been built from the
    same quantized operands this call produces.

    The second return value is the per-head count of exactly-computed KV blocks, float32 (nheads,); see
    _head_cost_from_routing.
    """
    from aiter.ops.mha import fmha_v3_fwd_sol_attn

    out_dtype = query.dtype
    query = query.permute(0, 2, 1, 3).contiguous()
    key = key.permute(0, 2, 1, 3).contiguous()
    value = value.permute(0, 2, 1, 3).contiguous()

    check_sol_attn_supported(query, key, value, is_causal, ring_world_size=ring_world_size)
    _maybe_dump(dump_path, query, key, value)

    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5

    # After the dump, so a dump captures the model's own operands rather than this backend's rotation of
    # them, and before quantization, which is the whole point: the rotation shrinks the per-tensor amax.
    if hadamard:
        query, key = _hadamard_rotate_qk(query, key)

    q_fp8, q_descale = _per_tensor_quant_fp8(query)
    k_fp8, k_descale = _per_tensor_quant_fp8(key)
    v_fp8, v_descale = _per_tensor_quant_fp8(value)

    prep = routing if routing is not None else sol_attn_routing_for(q_fp8, k_fp8, v_fp8, beta)

    out, _ = fmha_v3_fwd_sol_attn(
        q_fp8,
        k_fp8,
        v_fp8,
        softmax_scale,
        prep["kv_block_indices"],
        prep["lut_start"],
        prep["lut_count"],
        prep["mean_k"],
        prep["mean_v"],
        prep["block_bitmap"],
        q_descale,
        k_descale,
        v_descale,
    )
    out = out.to(out_dtype).permute(0, 2, 1, 3)
    # Honour the layout contract the fake kernel declares (see _sol_attn_output_strides). The kernel
    # writes a contiguous BSHD tensor, so the permute above already yields exactly these strides; this
    # only repairs the layout if a dtype conversion or a future aiter output layout changes that. Getting
    # it wrong surfaces as an opaque Inductor assert_size_stride failure, so it is checked here instead.
    expected = _sol_attn_output_strides(out.shape)
    if tuple(out.stride()) != expected:
        out = out.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)
    return out, _head_cost_from_routing(prep)


# torch.compile support.
#
# Dynamo must NOT trace into this. Two independent reasons, either of which is fatal:
#
#   1. aiter's fmha_v3_fwd_sol_attn has no fake/meta implementation. aiter's compile_ops accepts a
#      gen_fake hook but this op does not pass one, so fake-tensor propagation falls through to the
#      python stub, whose body is `...` and therefore returns None against a schema declaring Tensor[].
#      That surfaces as: TypeError("Object of type 'NoneType' is not an instance of 'sequence'").
#   2. More fundamentally, the routing has DATA-DEPENDENT output shapes. kv_block_indices is sized by
#      how many blocks the threshold selected, which is a property of the operand values, not of their
#      shapes. No fake tensor can express that, so no amount of fixing (1) would make the routing
#      traceable.
#
# The whole quantize -> route -> kernel sequence is therefore wrapped as one opaque custom op. Only its
# OUTPUT needs a fake, and that is static: same shape, dtype and device as the query. This matches how
# xDiT already handles the flydsl attention kernel.
_SOL_ATTN_OP_NAME = "xfuser::sol_attn_fp8"
_HAS_SOL_ATTN_OP = False

try:
    from torch.library import custom_op as _custom_op

    @_custom_op(_SOL_ATTN_OP_NAME, mutates_args=())
    def _sol_attn_fp8_op(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool,
        beta: float,
        softmax_scale: float,
        ring_world_size: int,
        dump_path: str,
        hadamard: bool,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        return _sol_attn_bhsd_eager(
            query,
            key,
            value,
            is_causal=is_causal,
            beta=beta,
            softmax_scale=softmax_scale,
            routing=None,
            ring_world_size=ring_world_size,
            dump_path=dump_path or None,
            hadamard=hadamard,
        )

    @_sol_attn_fp8_op.register_fake
    def _sol_attn_fp8_fake(query, key, value, is_causal, beta, softmax_scale, ring_world_size,
                           dump_path, hadamard):
        return (
            torch.empty_strided(
                tuple(query.shape), _sol_attn_output_strides(query.shape),
                dtype=query.dtype, device=query.device),
            torch.empty(query.shape[1], dtype=torch.float32, device=query.device),
        )

    _HAS_SOL_ATTN_OP = True
except (ImportError, AttributeError):
    # Older torch without torch.library.custom_op: the eager path still works, callers just cannot
    # torch.compile through it.
    pass


def sol_attn_bhsd(query, key, value, is_causal=False, beta=1.0, softmax_scale=None,
                  routing=None, ring_world_size=1, dump_path=None, hadamard=True):
    """Sol-Attn over BHSD tensors, returning (BHSD output in the input dtype, per-head cost).

    The per-head cost is float32 (nheads_q,), the number of exactly-computed KV blocks per head, for the
    Ulysses head balancer; callers that do not balance can ignore it. It is returned unconditionally
    rather than behind a flag because the op's schema is fixed, and it is cheap: a reduction over a mask
    the routing has already materialized.

    Routes through the opaque custom op so that torch.compile can call it without tracing the routing.
    Caller-supplied `routing` cannot cross an op boundary (it is a dict of tensors with data-dependent
    sizes), so that path stays eager and will graph-break under torch.compile.
    """
    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5
    if routing is not None or not _HAS_SOL_ATTN_OP:
        return _sol_attn_bhsd_eager(
            query, key, value, is_causal=is_causal, beta=beta, softmax_scale=softmax_scale,
            routing=routing, ring_world_size=ring_world_size, dump_path=dump_path,
            hadamard=hadamard)
    return torch.ops.xfuser.sol_attn_fp8(
        query, key, value, is_causal, float(beta), float(softmax_scale), int(ring_world_size),
        dump_path or "", bool(hadamard))


def sol_attn_dump_path():
    """Operand dump path from the environment, or None. See XFUSER_SOL_ATTN_DUMP in xfuser/envs.py.

    beta and hadamard are not read here: beta arrives per call through attention_kwargs from
    --solattn_beta, and the rotation is a property of the selected backend rather than a user knob.
    """
    from xfuser.envs import environment_variables

    return environment_variables["SOL_ATTN_DUMP"]() or None
