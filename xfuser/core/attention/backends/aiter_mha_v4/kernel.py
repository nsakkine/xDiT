"""MHA v4 launchers. Imported when one of the family is selected, so AITER is
present and its enums can be translated once here rather than per call."""

from dataclasses import dataclass

import torch

from aiter.ops.mha_v4 import (
    AttentionFormat,
    AttentionScaleMode,
    mha_v4,
    native_fp8_format,
)

from xfuser.core.attention.numerics.layout import (
    from_bshd,
    make_contiguous,
    to_bshd,
)
from xfuser.core.attention.requirements import PARAM, device_arch
from xfuser.core.attention.sparsity.sparge import (
    SpargeConfig,
    build_block_mask,
    cost_sink_from,
    restore_sparge_output,
)
from xfuser.core.attention.spec import AttnCall

from .spec import FORMATS, Fmt, MhaV4Format, Scale

FORMAT = {f: getattr(AttentionFormat, f.name) for f in Fmt if f is not Fmt.NATIVE_FP8}
FORMAT[Fmt.NATIVE_FP8] = native_fp8_format()
SCALE = {s: getattr(AttentionScaleMode, s.name) for s in Scale}


@dataclass(frozen=True)
class SparseGeometry:
    """How one format's sorted-sparse row cuts and bounds the keys.

    ``kv_tile`` is the block the Sparge mask must be cut at. The rows no longer
    agree on it: gfx950 routes BF16 and BF16/FP8 on 64 keys and the rest on 128.
    ``ragged_kv`` says the row masks the keys past a short last block itself, so
    the tail needs no zero pad -- which is not just a copy saved, since a zero
    key inside a selected block is attended at score 0 rather than skipped.
    """

    kv_tile: int
    ragged_kv: bool = False


def _sparse_geometry(fmt: MhaV4Format) -> SparseGeometry:
    # Not hoisted: older AITER builds have neither query, and answer the KV tile
    # for every row at once.
    try:
        from aiter.ops.mha_v4 import (
            MHA_V4_SPARSE_MODE,
            AttentionPack,
            mha_v4_kv_tile,
            mha_v4_operands,
            scale_modes_for_formats,
        )
    except ImportError:
        return SparseGeometry(64 if "gfx942" in device_arch() else 128)

    qk, v = FORMAT[fmt.qk], FORMAT[fmt.v]
    if fmt.qk_scale is None:
        scales = scale_modes_for_formats(qk, qk, v)
    else:
        scales = (SCALE[fmt.qk_scale], SCALE[fmt.qk_scale], SCALE[fmt.v_scale])
    # AITER repacks an MX V for the FP6 P operand on every row that has one,
    # dense and sparse alike, so the row is only found under that pack.
    pack = AttentionPack.V_FOR_FP6_P if fmt.v in (Fmt.MXFP6, Fmt.MXFP4) else AttentionPack.DEFAULT
    operands = mha_v4_operands(qk, qk, v, *scales, pack)
    kv_tile = int(mha_v4_kv_tile(operands, MHA_V4_SPARSE_MODE))
    try:
        from aiter.ops.mha_v4 import mha_v4_ragged_kv
    except ImportError:
        return SparseGeometry(kv_tile)
    return SparseGeometry(kv_tile, bool(kv_tile) and mha_v4_ragged_kv(operands, MHA_V4_SPARSE_MODE))


# Read once per format, here, because the kernel module is imported at backend
# selection: a manifest query per call would sit inside the traced region. A
# format this GPU has no sparse row for reads a zero tile; its spec is
# unselectable there, so the entry is never used.
SPARSE_GEOMETRY = {fmt.name: _sparse_geometry(fmt) for fmt in FORMATS if fmt.sparge_on is not None}

# Whether this AITER can be told how many keys each batch really has. Read once,
# here, because the kernel module is imported at backend selection -- evaluating
# it per call would put a signature probe inside the traced region.
HAS_SEQLENS_K = PARAM("aiter.ops.mha_v4:mha_v4", "seqlens_k").satisfied()


def _launch(q, k, v, fmt: MhaV4Format, block_mask=None, seqlens_k=None, return_lse=False):
    """One MHA v4 launch. Tensors are BSHD. Returns (output, lse-or-None)."""
    kwargs = {}
    if fmt.qk_scale is not None:
        kwargs = {
            "q_scale_mode": SCALE[fmt.qk_scale],
            "k_scale_mode": SCALE[fmt.qk_scale],
            "v_scale_mode": SCALE[fmt.v_scale],
        }
    if seqlens_k is not None:
        kwargs["seqlens_k"] = seqlens_k
    if return_lse:
        kwargs["return_lse"] = True
    qk = FORMAT[fmt.qk]
    result = mha_v4(q, k, v, qk, qk, FORMAT[fmt.v], block_mask=block_mask, **kwargs)
    return result if return_lse else (result, None)


def _shorten_keys(key, value, call: AttnCall, fmt: MhaV4Format):
    """Fold a key pack into a dense call. Returns (key, value, seqlens_k), BSHD.

    MHA v4 has no key-padding mask and does not need one. One sequence's valid
    keys are simply a shorter K/V, so attending over them is exact rather than
    approximate. Several are regrouped into a padded batch whose true lengths
    travel in seqlens_k, which the kernels read per batch, so the padding is
    never visited.

    ``valid_kv_len`` is one length for the whole call, so it can only describe a
    single sequence. A batch of several is gathered even when it declares one,
    because slicing would cut every row to the longest row's length and leave
    the shorter ones attending over their own pad. Nothing rejects that: the
    declaration passes both checks, since the longest segment genuinely is the
    valid count for one of the rows.

    Q is left alone either way: it is never packed, and a key-side length would
    be wrong for cross attention, where the two sequences differ.
    """
    valid = call.attention_kwargs.get("valid_kv_len")
    if valid is not None and key.shape[0] == 1:
        # A declared trailing pad on one sequence, which accepts has already
        # checked describes the pack truthfully. One copy, where gathering
        # costs two.
        return key[:, :valid].contiguous(), value[:, :valid].contiguous(), None

    # Gathered over K's own shape rather than through layout.pack_kv, which
    # flattens K against the *query* length. Equal for self attention, but this
    # path is reached by cross attention too, where the two differ.
    batch, seq_len, heads, head_dim = key.shape
    flat = (batch * seq_len, heads, head_dim)
    indices = call.varlen.indices_k
    k_packed = torch.index_select(key.reshape(flat), 0, indices)
    v_packed = torch.index_select(value.reshape(flat), 0, indices)

    if batch == 1:
        return (
            k_packed.reshape(1, -1, heads, head_dim),
            v_packed.reshape(1, -1, heads, head_dim),
            None,
        )

    if not HAS_SEQLENS_K:
        raise NotImplementedError(
            "this AITER build cannot express per-batch key lengths for MHA v4, "
            "so varlen packed keys with batch size > 1 are unsupported"
        )
    if fmt.qk is not Fmt.BF16:
        # AITER rejects the other objects rather than attend over the padding.
        # Name the ones that work instead of relaying a message about formats
        # the caller never chose.
        raise NotImplementedError(
            f"MHA v4 carries per-batch key lengths only on its BF16 Q/K rows, "
            f"so {fmt.name} Q/K cannot serve a batch of several padded "
            "sequences. Use AITER_BF16 or AITER_BF16FP8 for this model, or a "
            "configuration whose per-call batch is one."
        )

    # Scatter the packed rows back into a [batch, max_k, ...] buffer, one
    # sequence per row, and hand the true lengths over. The zeros are never
    # visited, which is what makes this exact rather than approximate.
    device = k_packed.device
    cu_k = call.varlen.cu_seqlens_k.to(device=device, dtype=torch.int32)
    lengths = (cu_k[1:] - cu_k[:-1]).contiguous()
    counts = lengths.to(torch.int64)
    rows = torch.arange(k_packed.shape[0], device=device)
    slot = rows - torch.repeat_interleave(cu_k[:-1].to(torch.int64), counts)
    row = torch.repeat_interleave(torch.arange(batch, device=device), counts)
    shape = (batch, int(call.varlen.max_seqlen_k), heads, head_dim)
    key, value = k_packed.new_zeros(shape), v_packed.new_zeros(shape)
    key[row, slot] = k_packed
    value[row, slot] = v_packed
    return key, value, lengths


def mha_v4_dense(query, key, value, call: AttnCall, *, fmt: MhaV4Format):
    # K/V stay views until they are shortened, so the pad is dropped in the
    # same copy that makes them contiguous rather than in one after it. Both
    # routes out of _shorten_keys are contiguous already: a slice and a gather
    # each materialise, and the scatter writes into a fresh buffer.
    q = to_bshd(query, contiguous=True)
    k, v = to_bshd(key, value)
    seqlens_k = None
    if call.varlen is not None:
        k, v, seqlens_k = _shorten_keys(k, v, call, fmt)
    else:
        k, v = make_contiguous(k, v)

    # Only ring needs the log-sumexp; asking for it otherwise buys a write the
    # caller discards. The degree comes off the call rather than the process
    # group, so this stays traceable and testable without one.
    output, softmax_lse = _launch(
        q,
        k,
        v,
        fmt,
        seqlens_k=seqlens_k,
        return_lse=call.ring_world_size > 1,
    )
    # The kernel writes LSE as [batch, heads, Sq], already the layout the ring
    # merge expects, so only O is permuted back.
    return from_bshd(output), softmax_lse


def mha_v4_sparge(query, key, value, call: AttnCall, *, fmt: MhaV4Format):
    geometry = SPARSE_GEOMETRY[fmt.name]
    q, k, v, state, block_mask = build_block_mask(
        query,
        key,
        value,
        is_causal=call.is_causal,
        config=SpargeConfig.from_kwargs(call.attention_kwargs),
        block_m=256,
        block_n=geometry.kv_tile,
        ulysses_world_size=call.ulysses_world_size,
        cost_sink=cost_sink_from(call.attention_kwargs),
        pad_block_divisible=not geometry.ragged_kv,
    )
    q, k, v = to_bshd(q, k, v, contiguous=True)
    output, _ = _launch(q, k, v, fmt, block_mask=block_mask)
    return restore_sparge_output(from_bshd(output), state), None
