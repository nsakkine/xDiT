"""Sol-Attn: block-sparse fp8 attention with a pooled-KV correction for the blocks it skips.

Plain block-sparse attention routes each query tile to a subset of KV blocks and DROPS the rest.
Sol-Attn (arXiv 2607.24027) computes the same selected blocks exactly and additionally recovers the
contribution of the skipped blocks from pooled (mean) K/V, so the mass outside the selection is
approximated instead of discarded. It runs on the gfx950 ASM kernel shipped in aiter as
fwd_hd128_fp8_sol_attn.co and driven by aiter.ops.mha.fmha_v3_fwd_sol_attn.

This module holds everything the backend needs that is not xDiT plumbing: the hard compatibility
constraints, the Hadamard rotation, per-tensor fp8 quantization, routing construction, and the kernel
call.

WHY THE CONSTRAINTS ARE HARD ERRORS. The kernel family derives the KV head for a Q head by SHIFTING
the head index right by floor(log2(nheads_q / nheads_kv)) rather than dividing:

    # s[58] = gqa_ratio (power-of-2: 1, 2, 4, 8, 16)
    s_lshr_b32(_s_tmp1, _s_tgid_y, s[58])   # kv_head = tgid_y >> gqa_shift

A non-power-of-2 GQA ratio therefore reads the WRONG KV head, and because it is a shift the wrong head
is frequently out of bounds. Measured at hd128 fp8 on gfx950, for dense as well as the sparse and
Sol-Attn paths, the observable outcome is one of: silently wrong output (cosine 0.72 vs reference at
nheads_q=10 / nheads_kv=2), non-finite output confined to exactly the misrouted heads, or an
unrecoverable GPU page fault -- and which one you get depends on the allocation layout, so it is not
reproducible run to run. There is no accuracy gate that reliably catches the first case, so this module
refuses the configuration up front rather than letting a pipeline emit quietly corrupted frames.

ROUTING COST. sol_attn_prepare (pooling + routing, both host-launched Triton) costs roughly 1-2 ms and
is largely shape-insensitive, against a kernel that runs 1.0-2.6 ms at video shapes. Recomputing it on
every attention call is therefore a real fraction of the win, and at short sequence lengths it exceeds
the entire kernel time. It is still recomputed per call here, because reuse requires knowing WHICH
layer is calling: xDiT hands the registered attention function only (query, key, value, dropout_p,
is_causal, attention_kwargs), and every transformer block in a model like Wan has identical Q/K shapes,
so any cache keyed on shape would hand layer 0's routing to all 40 blocks. A caller that can supply
that identity may pass precomputed routing through attention_kwargs["sol_attn_routing"], which is used
as-is; see sol_attn_routing_for().
"""
import os
import typing

import torch

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})

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

    The ASM kernel writes its own contiguous BSHD (batch, seqlen, nheads, head_dim) tensor and the
    result is returned as a BHSD permute of it, so the strides are always these regardless of how the
    caller laid out its query. The fake kernel must declare the same thing: torch.empty_like(query)
    would instead PRESERVE the query's layout, and in a model like Wan the query is itself a transposed
    view of a BSHD-contiguous tensor, so the fake and the implementation would disagree and Inductor
    would fail an assert_size_stride check on the op's output.
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

    Reuses the dense fp8 backend's matrix and rotation so the two backends cannot drift apart; it is a
    full-head rotation here, since that matrix is built at block_r=128 and Sol-Attn requires
    head_dim==128 exactly, so there is no partial block to reason about.

    Sol-Attn stays correct under this for a reason dense attention does not need: besides leaving Q@K.T
    untouched, the rotation COMMUTES WITH THE POOLING the approximate branch depends on. mean_k averages
    keys along the sequence while the rotation acts on head_dim and is applied identically to every
    token, so mean(K @ R) == mean(K) @ R, and because sol_attn_prepare pools the quantized operands it is
    handed, rotating before quantization puts mean_k in the same rotated space as Q automatically. The
    routing therefore sees the same proxy scores and picks the same blocks, up to the near-tau fp8
    rounding _sol_attn_route documents. V is deliberately NOT rotated: mean_v is accumulated straight
    into the output, so a rotation there would have to be undone afterwards.
    """
    from xfuser.core.distributed.attention_backend import (
        FP8_HADAMARD_MATRIX,
        _fp8_hadamard_rotate,
    )

    r = FP8_HADAMARD_MATRIX[query.device]
    if r is None:
        raise SolAttnUnsupported(
            "XFUSER_SOL_ATTN_HADAMARD is set but no Hadamard matrix could be built. Set "
            "XFUSER_SOL_ATTN_HADAMARD=0 to quantize the raw operands instead.")
    return (_fp8_hadamard_rotate(query, r).contiguous(),
            _fp8_hadamard_rotate(key, r).contiguous())


def sol_attn_routing_for(q_fp8, k_fp8, v_fp8, beta):
    """Pooled K/V, ragged LUT and selection bitmap for one call, as the kernel consumes them."""
    from aiter.ops.triton.attention.utils import sol_attn_prepare

    block_m, block_n = sol_attn_block_sizes()
    return sol_attn_prepare(q_fp8, k_fp8, v_fp8, beta=beta, BLOCK_M=block_m, BLOCK_N=block_n)


def _head_cost_from_routing(prep):
    """Per-head count of exactly-computed KV blocks, float32 (nheads_q,).

    This is what the Ulysses head balancer needs from a block-sparse backend, and it is the same
    quantity the sparge backends publish, computed the same way. Sol-Attn's per-head cost is genuinely
    data dependent: the routing threshold selects however many blocks a head's pooled proxy scores put
    above tau, so heads differ in how much exact work they cost, which is the imbalance the balancer
    corrects. The skipped blocks are ignored here because their pooled correction is a fixed, uniform
    cost per head and so cannot contribute to imbalance.

    The count must be summed here rather than outside the op: the routing lives inside the opaque custom
    op (its LUT sizes are data dependent), so this is the only place the mask exists. Reducing it to a
    (nheads,) tensor is what makes the cost expressible in the op's fake kernel, since that shape
    depends only on the operand shapes.
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


def sol_attn_settings():
    """(beta, dump_path, hadamard) resolved from the environment, validated."""
    from xfuser.envs import environment_variables

    raw_beta = environment_variables["SOL_ATTN_BETA"]()
    try:
        beta = float(raw_beta)
    except (TypeError, ValueError):
        raise SolAttnUnsupported(
            f"XFUSER_SOL_ATTN_BETA must be a float, got {raw_beta!r}") from None
    dump_path = environment_variables["SOL_ATTN_DUMP"]() or None
    raw_hadamard = str(environment_variables["SOL_ATTN_HADAMARD"]()).strip().lower()
    if raw_hadamard not in _TRUTHY | _FALSY:
        raise SolAttnUnsupported(
            f"XFUSER_SOL_ATTN_HADAMARD must be one of {sorted(_TRUTHY | _FALSY)}, got "
            f"{raw_hadamard!r}")
    return beta, dump_path, raw_hadamard in _TRUTHY
