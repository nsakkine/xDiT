"""Sol-Attn: block-sparse fp8 attention with a pooled-KV correction for the blocks it skips.

Plain block-sparse attention routes each query tile to a subset of KV blocks and DROPS the rest.
Sol-Attn (arXiv 2607.24027) computes the same selected blocks exactly and additionally recovers the
contribution of the skipped blocks from pooled (mean) K/V, so the mass outside the selection is
approximated instead of discarded.
"""
from types import SimpleNamespace
from typing import NamedTuple

import torch

# gfx950 carries a mode-2 manifest row for every recipe below; gfx942 carries only the two
# per-tensor ones, so the rest are refused per recipe rather than per device.
_SUPPORTED_ARCHS = ("gfx942", "gfx950")
_ARCH_RECIPES = {"gfx942": ("fp8", "i8fp8")}


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
            AttentionFormat,
            AttentionScaleMode,
            mha_v4_kv_tile,
            mha_v4_packed,
            mha_v4_q_multiplier,
            mha_v4_sol_attn,
            mxfp4_k_view,
            mxfp4_v_view,
            native_fp8_format,
            quantize_fp8,
            quantize_fp8_rotated,
            quantize_int8,
            quantize_mxfp4_k,
            quantize_mxfp4_q,
            quantize_mxfp8_k,
            quantize_mxfp8_q,
            quantize_v_mxfp4,
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
        packed=mha_v4_packed,
        prepare=sol_attn_prepare,
        native_fp8_format=native_fp8_format,
        kv_tile=mha_v4_kv_tile,
        q_multiplier=mha_v4_q_multiplier,
        quantize_fp8=quantize_fp8,
        quantize_fp8_rotated=quantize_fp8_rotated,
        quantize_int8=quantize_int8,
        quantize_mxfp8_q=quantize_mxfp8_q,
        quantize_mxfp8_k=quantize_mxfp8_k,
        quantize_mxfp4_q=quantize_mxfp4_q,
        quantize_mxfp4_k=quantize_mxfp4_k,
        quantize_mxfp4_v=quantize_v_mxfp4,
        mxfp4_k_view=mxfp4_k_view,
        mxfp4_v_view=mxfp4_v_view,
        fmt=AttentionFormat,
        per_tensor_scale=AttentionScaleMode.F32_PER_TENSOR,
        block_scale=AttentionScaleMode.E8M0_PER_1X32,
        ts_qo=SOL_ATTN_TS_QO,
        ts_kv=SOL_ATTN_TS_KV,
    )


_AITER = _probe_aiter()
SOL_ATTN_AVAILABLE = _AITER is not None

# Held as a module global rather than an _AITER field, which is not a style choice. aiter decorates
# mha_v4_kv_tile with functools.cache, so the name is an _lru_cache_wrapper -- a C object that
# implements the descriptor protocol. Reached through an attribute Dynamo binds the owner as self
# and the call dies with "Too many positional arguments: got 1, expected 0", which under
# fullgraph=True is a hard error and otherwise a graph break in the middle of every attention layer.
# Eager does not bind (SimpleNamespace holds it in an instance dict) so this shows up only compiled.
_kv_tile = _AITER.kv_tile if _AITER is not None else None


class _Recipe(NamedTuple):
    """One mode-2 manifest row: its formats, its quantizers, and how its scales survive pooling.

    The recipes divide on one question, which is whether an operand's scale varies along the
    sequence axis pooling reduces. A per-tensor descale does not, so the pooled operand reuses it
    and the kernel's pooled-scale slots stay empty. An E8M0 1x32 scale does, so that operand pools
    in dequantized space and hands the kernel a pooled scale of its own.

    quantize_q takes a multiplier because the MX quantizers fold softmax_scale * log2(e) into Q;
    the per-tensor ones ignore it and the kernel applies the scale itself.
    """

    id: str
    qk_format: str            # attribute name on AttentionFormat, resolved per device
    qk_scale_mode: str        # "per_tensor" or "block"
    quantize_q: str           # attribute name on the _AITER namespace
    quantize_k: str
    quantize_v: str = "quantize_fp8"
    v_format: str = "native"
    v_scale_mode: str = "per_tensor"
    # Set when the stored codes are sub-byte and permuted into the ASM's tile order. Such an
    # operand cannot be pooled from its codes at all, so pooling works from the BF16 source and
    # the operand's own scale must not be handed to sol_attn_prepare.
    packed_format: str | None = None

    @property
    def routes_through_raw(self) -> bool:
        """Whether mha_v4_sol_attn can serve this row, or it has to go packed.

        The raw entry point quantizes for the caller and only knows the per-tensor recipes. The MX
        rows reach the same kernels through mha_v4_packed with operands quantized here.
        """
        return self.qk_scale_mode == "per_tensor" and self.packed_format is None


_RECIPES = {
    r.id: r
    for r in (
        _Recipe("fp8", "native", "per_tensor",
                "quantize_fp8_rotated", "quantize_fp8_rotated"),
        _Recipe("i8fp8", "INT8", "per_tensor",
                "quantize_int8", "quantize_int8"),
        _Recipe("mxfp8", "native", "block",
                "quantize_mxfp8_q", "quantize_mxfp8_k"),
        _Recipe("mxfp4", "MXFP4", "block",
                "quantize_mxfp4_q", "quantize_mxfp4_k",
                quantize_v="quantize_mxfp4_v", v_format="MXFP4",
                v_scale_mode="block", packed_format="mxfp4"),
    )
}

SOL_ATTN_RECIPES = tuple(_RECIPES)


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
    arch = _device_arch(device)
    if arch is None:
        reported = torch.cuda.get_device_properties(device).gcnArchName or ""
        raise SolAttnUnsupported(
            f"Sol-Attn ships kernels for {', '.join(_SUPPORTED_ARCHS)}, "
            f"this device reports '{reported}'")


def _device_arch(device=None):
    """The supported arch this device is, or None. gcnArchName carries a target-feature suffix."""
    if device is None:
        if not torch.cuda.is_available():
            return None
        device = torch.device("cuda", torch.cuda.current_device())
    name = torch.cuda.get_device_properties(device).gcnArchName or ""
    return next((arch for arch in _SUPPORTED_ARCHS if name.startswith(arch)), None)


def check_sol_attn_recipe(recipe_id, device=None):
    """Raise unless this device has a Sol-Attn manifest row for recipe_id.

    Separate from check_sol_attn_device because the answer is per recipe: gfx942 runs the two
    per-tensor rows and has no MX ones, so selecting aiter_mxfp8_sol_attn there has to fail at
    setup with the reason rather than at the first launch with a missing-kernel error.
    """
    check_sol_attn_device(device)
    allowed = _ARCH_RECIPES.get(_device_arch(device))
    if allowed is not None and recipe_id not in allowed:
        raise SolAttnUnsupported(
            f"Sol-Attn '{recipe_id}' has no {_device_arch(device)} manifest row; "
            f"this device serves {', '.join(allowed)}")


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


def _resolve_recipe(recipe):
    """The named recipe, or a clear error listing what this build actually ships."""
    try:
        return _RECIPES[recipe]
    except KeyError:
        raise SolAttnUnsupported(
            f"unknown Sol-Attn recipe {recipe!r}; this build has "
            f"{', '.join(SOL_ATTN_RECIPES)}") from None


def _format(name):
    """An AttentionFormat by name, with "native" resolved against the active GPU."""
    return _AITER.native_fp8_format() if name == "native" else getattr(_AITER.fmt, name)


def _scale_mode(name):
    return _AITER.per_tensor_scale if name == "per_tensor" else _AITER.block_scale


def _quantize(recipe, query, key, value, softmax_scale):
    """Quantize BSHD bf16 Q/K/V for one recipe, returning (tensor, descale, source) per operand.

    The MX quantizers fold softmax_scale * log2(e) into Q, which is why the scale has to be
    resolved before quantizing rather than left for the kernel. The per-tensor quantizers take no
    multiplier and the kernel applies the scale itself, so it is passed on either way.

    source is the pre-quantization tensor, kept only for a packed recipe: its codes are sub-byte
    and permuted into tile order, so neither pooling nor routing can read them back.
    """
    packed = recipe.packed_format is not None
    multiplier = _AITER.q_multiplier(softmax_scale)

    if recipe.qk_scale_mode == "block":
        q, q_descale = getattr(_AITER, recipe.quantize_q)(query, multiplier)
    else:
        q, q_descale = getattr(_AITER, recipe.quantize_q)(query)

    if recipe.packed_format == "mxfp4":
        # These packers emit a flat backing buffer plus its scale; the kernel wants the strided
        # view over that buffer, so build it here exactly as aiter's own recipe does.
        k_raw, k_descale = _AITER.quantize_mxfp4_k(key)
        k = _AITER.mxfp4_k_view(k_raw, k_descale)
        v_raw, v_descale = _AITER.quantize_mxfp4_v(value)
        v = _AITER.mxfp4_v_view(v_raw, v_descale, value.shape[1])
    else:
        k, k_descale = getattr(_AITER, recipe.quantize_k)(key)
        v, v_descale = getattr(_AITER, recipe.quantize_v)(value)

    return ((q, q_descale, query if packed else None),
            (k, k_descale, key if packed else None),
            (v, v_descale, value if packed else None))


def _force_blocks_from_tokens(exact_tokens, key_seqlen, padded_seqlen):
    """Per-KV-block "compute this exactly" flags from a per-token mask.

    A block is forced on when it holds any flagged token, because the block is the finest thing
    the selection can express. Follows K/V through the same trim and tile pad so the two cannot
    fall out of step, and is pure tensor work -- no host-side read of device data -- so it stays
    traceable.
    """
    if key_seqlen is not None and key_seqlen < exact_tokens.shape[0]:
        exact_tokens = exact_tokens[:key_seqlen]
    pad = padded_seqlen - exact_tokens.shape[0]
    if pad > 0:
        # The tile pad is zero keys, which nothing needs computed exactly.
        exact_tokens = torch.nn.functional.pad(exact_tokens, (0, pad))
    return exact_tokens.reshape(-1, _kv_tile()).any(dim=1)


def sol_attn_routing_for(q, k, v, beta, recipe=_RECIPES["fp8"], force_blocks=None):
    """Pooled K/V, ragged LUT and selection bitmap for one call, as the kernel consumes them.

    q/k/v are the (tensor, descale, source) triples _quantize returns. num_heads is passed
    explicitly because the routing is per query head under GQA, while the pooled K/V it returns
    carry the KV head count instead.
    """
    packed = recipe.packed_format
    block_scaled = recipe.qk_scale_mode == "block"
    return _AITER.prepare(
        # Routing scores Q, and a packed Q is no more addressable than a packed K, so a packed
        # recipe routes on its source. Routing is scale invariant, which is what makes the two
        # interchangeable, and the packers' Hadamard rotation is orthogonal so it drops out too.
        q[2] if packed is not None else q[0],
        k[0],
        v[0],
        beta,
        _AITER.ts_qo,
        # From the manifest, not aiter's SOL_ATTN_TS_KV default, which is the gfx950 tile: gfx942
        # pools 64 rows per block against gfx950's 128. Pooling at the wrong one is not a rounding
        # difference, it hands the kernel pooled tensors of the wrong height. The pad above already
        # reads the same source, so taking the two from one place keeps them from drifting apart.
        _kv_tile(),
        num_heads=(q[2] if packed is not None else q[0]).shape[2],
        # A packed operand pools from its source and is quantized again, so it has no stored scale
        # to pool and offering one is an error.
        k_scale=k[1] if block_scaled and packed is None else None,
        v_scale=v[1] if recipe.v_scale_mode == "block" and packed is None else None,
        k_source=k[2],
        v_source=v[2],
        k_packed_format=packed,
        v_packed_format=packed,
        force_block_mask=force_blocks,
    )


def _head_cost_from_routing(routing):
    """Per-head count of exactly-computed KV blocks, float32 (nheads_q,).
    """
    mask = routing["block_attn_mask"]  # (batch, nheads_q, num_q_tiles, num_kv_blocks) bool
    return mask.sum(dim=(0, 2, 3), dtype=torch.float32)


# Below this many KV blocks, routing has too few samples to say anything. tau is a mean plus a
# multiple of a standard deviation taken across the blocks, so with a handful of them the
# threshold is noise, and whatever it does not select is replaced by block means that are then a
# large fraction of the whole sequence. 16 is a judgement call, not a measured cliff.
_MIN_USEFUL_KV_BLOCKS = 16
_short_kv_warned = set()


def _warn_if_kv_too_short(seqlen_k):
    """Say something when a call is too short for Sol-Attn to be worth routing.

    This is the failure mode that does not announce itself: nothing raises, the answer is just
    quietly worse than the dense backend would have given, which is easy to ship by accident when
    a model turns out to have a second attention site on a short sequence.

    Skipped under torch.compile. The branch folds away at trace time, which keeps the warning from
    being a side effect in a traced region -- so a fully compiled run will not see it, and this is
    a backstop for eager runs rather than a guarantee.
    """
    if torch.compiler.is_compiling():
        return
    blocks = seqlen_k // _kv_tile()
    if blocks >= _MIN_USEFUL_KV_BLOCKS or blocks in _short_kv_warned:
        return
    _short_kv_warned.add(blocks)
    import warnings

    warnings.warn(
        f"Sol-Attn was handed a {seqlen_k}-token KV, i.e. {blocks} blocks of {_kv_tile()}. Its "
        f"routing threshold is a mean plus a multiple of a standard deviation over the blocks, "
        f"which says little at that count, and the pooled correction is then standing in for much "
        f"of the sequence. Expect a worse result than a dense backend, not a faster one. If this "
        f"is a short text-only attention inside a model whose main attention is long, that site "
        f"should be on a dense backend.",
        RuntimeWarning,
        stacklevel=3,
    )


def _pad_kv_to_tile(key, value):
    """Right-pad BSHD K/V with zero tokens so seqlen_k is a whole number of KV blocks.

    The LUT-based mha_v4 rows index KV in whole tiles and are handed no real token count, so aiter
    rejects a ragged seqlen_k outright rather than read past the last block. Wan lands on one at
    every standard size -- 720p is 21 latent frames x 80 x 45 = 75600 tokens, which is 590.6 blocks
    -- so this is the common case, not the corner.

    Zero tokens are what xDiT's Sparge path already pads with, and they are not free: a zero key
    scores q . 0 = 0 rather than -inf, so the pad draws softmax weight exp(-max) per token instead
    of none, and its zero value pulls the row toward the origin by that weight. The pad is at most
    one block against Wan's 591 and the mass it takes is exponentially small in the row max, which
    is why this is a pad and not a mask. Q is deliberately left alone: nothing constrains seqlen_q,
    and padding it would only add rows to slice back off the output.
    """
    pad = -key.shape[1] % _kv_tile()
    if pad == 0:
        return key, value
    # BSHD, so the seqlen axis is the second of four and F.pad counts from the last.
    widths = (0, 0, 0, 0, 0, pad)
    return (torch.nn.functional.pad(key, widths),
            torch.nn.functional.pad(value, widths))


def _maybe_dump(path, query, key, value):
    """Save one call's BSHD q/k/v so kernel benchmarks can be replayed on real tensors.
    """
    if not path or getattr(_maybe_dump, "_done", False):
        return
    _maybe_dump._done = True
    torch.save({"q": query.detach().cpu(), "k": key.detach().cpu(), "v": value.detach().cpu()}, path)


def sol_attn_bhsd(query, key, value, is_causal=False, beta=1.0, softmax_scale=None,
                  routing=None, ring_world_size=1, dump_path=None,
                  return_head_cost=False, recipe="fp8", key_seqlen=None,
                  exact_tokens=None):
    """Sol-Attn over BHSD tensors, returning (BHSD bf16 output, per-head cost or None).

    query/key/value are (batch, nheads, seqlen, head_dim) bf16 tensors. They are permuted into the
    BSHD layout the kernel takes and handed to aiter's mha_v4 Sol-Attn row for `recipe`, which owns
    the Q/K Hadamard rotation, format validation and the ASM launch. See SOL_ATTN_RECIPES for the
    rows this build ships.

    return_head_cost asks for the per-head count of exactly-computed KV blocks, float32 (nheads_q,),
    which the Ulysses head balancer consumes. On a per-tensor recipe it is opt-in rather than free:
    the routing dict it reduces is internal to the raw entrypoint, so asking for it moves the call
    onto aiter's packed API and makes quantization and routing explicit here. The MX recipes have no
    raw entrypoint and take that path always. A caller-supplied `routing` also forces it. Every path
    is traceable; none graph-break.

    key_seqlen is the number of real KV tokens when the caller has already padded the sequence for
    its own alignment, as MiniMax-H3 does to pack one request into one audiovisual sequence. Sol-Attn
    takes no attention mask, so the padded tail is dropped here rather than attended: left in, it
    would draw softmax weight and also enter the routing threshold's per-tile mean and standard
    deviation, which is a quieter error than the weight itself. Defaults to the whole tensor.

    exact_tokens is a per-KV-token bool mask, (seqlen_k,), naming tokens that must be computed
    exactly rather than left to the pooled approximation. Routing still runs; these are added to
    what it picked. A caller whose sequence is one modality does not need this. A caller packing
    several into one sequence does: the threshold is built from statistics over every block, so a
    modality holding a small share of the sequence is judged against a distribution the majority
    wrote, and its own queries lose the blocks they needed. Flagging the minority costs about one
    exact block per block it occupies, which is few by the same token.
    """
    if _AITER is None:
        raise SolAttnUnsupported(
            "Sol-Attn requires aiter with mha_v4_sol_attn and sol_attn_prepare; "
            "please update AITER")
    recipe = _resolve_recipe(recipe)

    query = query.permute(0, 2, 1, 3).contiguous()
    key = key.permute(0, 2, 1, 3).contiguous()
    value = value.permute(0, 2, 1, 3).contiguous()

    check_sol_attn_supported(query, key, value, is_causal, ring_world_size=ring_world_size)
    _maybe_dump(dump_path, query, key, value)
    # Before the tile pad, so the two do not stack: the caller's alignment is dropped and only the
    # kernel's own remainder is padded back, which is at most one block of zero tokens.
    if key_seqlen is not None and key_seqlen < key.shape[1]:
        key, value = key[:, :key_seqlen], value[:, :key_seqlen]
    key, value = _pad_kv_to_tile(key, value)
    _warn_if_kv_too_short(key.shape[1])

    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5

    force_blocks = (
        None if exact_tokens is None
        else _force_blocks_from_tokens(exact_tokens, key_seqlen, key.shape[1])
    )

    # The raw entrypoint routes internally from beta alone, so it cannot be told about forced
    # blocks any more than it can be asked for the head cost.
    if (routing is None and not return_head_cost and force_blocks is None
            and recipe.routes_through_raw):
        qk, v_fmt = _format(recipe.qk_format), _format(recipe.v_format)
        out = _AITER.sol_attn(query, key, value, qk, qk, v_fmt, beta=beta,
                              softmax_scale=softmax_scale)
        return out.permute(0, 2, 1, 3), None

    # Quantize exactly as the raw entrypoint would. On the fp8 row that means quantize_fp8_rotated
    # for Q/K: rotating both by the same orthonormal matrix leaves Q @ K.T alone while spreading the
    # outliers that dominate fp8 error, and V is not rotated because nothing cancels a rotation of
    # it. Reusing aiter's fused rotation rather than doing a second one here is what keeps this path
    # numerically identical to the raw one, so asking for the head cost cannot change the output.
    q, k, v = _quantize(recipe, query, key, value, softmax_scale)
    if routing is None:
        routing = sol_attn_routing_for(q, k, v, beta, recipe, force_blocks=force_blocks)
    qk_fmt, v_fmt = _format(recipe.qk_format), _format(recipe.v_format)
    qk_scale, v_scale = _scale_mode(recipe.qk_scale_mode), _scale_mode(recipe.v_scale_mode)
    out = _AITER.packed(
        q[0], k[0], v[0],
        q[1], k[1], v[1],
        qk_fmt, qk_fmt, v_fmt,
        qk_scale, qk_scale, v_scale,
        softmax_scale=softmax_scale,
        kv_block_indices=routing["kv_block_indices"],
        lut_start=routing["lut_start"],
        lut_count=routing["lut_count"],
        mean_k=routing["mean_k"],
        mean_v=routing["mean_v"],
        block_bitmap=routing["block_bitmap"],
        # None on the per-tensor rows, where the pooled operand reuses the source descale. The MX
        # rows pool in dequantized space and requantize, so the pooled tensor has a scale of its own
        # and the kernel has to be told about it or it reads the approximate branch at the wrong
        # magnitude.
        mean_k_scale=routing["mean_k_scale"],
        mean_v_scale=routing["mean_v_scale"],
    )
    return out.permute(0, 2, 1, 3), _head_cost_from_routing(routing)


def sol_attn_dump_path():
    """Operand dump path from the environment, or None. See XFUSER_SOL_ATTN_DUMP in xfuser/envs.py.

    beta and hadamard are not read here: beta arrives per call through attention_kwargs from
    --solattn_beta, and the rotation is a property of the selected backend rather than a user knob.
    """
    from xfuser.envs import environment_variables

    return environment_variables["SOL_ATTN_DUMP"]() or None
