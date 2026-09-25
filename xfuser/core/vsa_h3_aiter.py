# SPDX-License-Identifier: Apache-2.0
"""FastH3 VSA-H3 over AITER's 64x64 block-sparse MHA v4 rows.

Pure sparse (mode 1) rather than Sol-Attn (mode 2): Sol-Attn folds a pooled approximation of the
blocks it skipped back inside the softmax, and FastH3 already carries a pooled branch of its own
that it mixes outside the softmax with a learned gate. Routing here through Sol-Attn would count
that mass twice, once per formulation.

The selection is shared with the other two VSA-H3 kernels, not reimplemented: build_h3_vsa_kv_list
already returns the ascending per-query-tile key-tile list that the sorted-sparse row's ragged LUT
wants, so the conversion below is index arithmetic on shapes and no device work at all.
"""
from __future__ import annotations

import functools
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from xfuser.core.vsa_h3_attention import (
    FASTH3_VSA_SPARSITY,
    FASTH3_VSA_TILE_ELEMENTS,
    MiniMaxH3VSAMetadata,
    build_h3_vsa_kv_list,
    compute_h3_vsa_topk,
)
from xfuser.logger import init_logger

logger = init_logger(__name__)

# VSA-H3 pools and selects over 64-token tiles, so its selection is only expressible at this
# geometry. gfx950 serves it in BF16 and FP8; gfx942's finest block-sparse row is 256x64.
VSA_H3_AITER_TILE = (FASTH3_VSA_TILE_ELEMENTS, FASTH3_VSA_TILE_ELEMENTS)

# mha_v4 rows are hd128 only, in every precision.
VSA_H3_AITER_HEAD_DIM = 128


def _keep_bf16(tensor):
    """Pass a BF16 operand through unquantized, in the (tensor, descale) shape the others return.

    The BF16 rows carry a NONE scale mode, so the kernel never reads the descale. aiter's own
    mha_v4 hands the tensor itself back as the placeholder rather than a unit scalar, and this
    matches that so what this module sends cannot disagree with what the raw entry point would.
    """
    return tensor, tensor


def _probe_aiter():
    """aiter's sorted-sparse entry points, or None when this build does not ship them.

    Resolved at import, which costs nothing for runs that never select the backend: this module is
    imported only once one of its rows is dispatched. native_fp8_format stays a function because it
    queries the device, and import time is too early to have one.
    """
    try:
        from aiter.ops.mha_v4 import (
            AttentionFormat,
            AttentionScaleMode,
            MHA_V4_SPARSE_MODE,
            mha_v4_block_tiles,
            mha_v4_operands,
            mha_v4_packed,
            native_fp8_format,
            quantize_fp8,
            quantize_fp8_rotated,
        )
    except ImportError:
        return None
    return SimpleNamespace(
        packed=mha_v4_packed,
        block_tiles=mha_v4_block_tiles,
        operands=mha_v4_operands,
        sparse_mode=MHA_V4_SPARSE_MODE,
        native_fp8_format=native_fp8_format,
        quantize_fp8=quantize_fp8,
        quantize_fp8_rotated=quantize_fp8_rotated,
        quantize_bf16=_keep_bf16,
        fmt=AttentionFormat,
        no_scale=AttentionScaleMode.NONE,
        per_tensor_scale=AttentionScaleMode.F32_PER_TENSOR,
    )


_AITER = _probe_aiter()
VSA_H3_AITER_AVAILABLE = _AITER is not None

VSA_H3_AITER_RECIPES = ("bf16", "fp8")


def _recipe_operands(recipe_id: str):
    """The six format/scale-mode codes that pick this recipe's manifest row, and its quantizers.

    Deferred rather than a module constant because the FP8 format is the device's: gfx942 and
    gfx950 disagree on which E4M3 encoding "native FP8" names, and the answer needs a device.
    """
    if recipe_id == "bf16":
        fmt = _AITER.fmt.BF16
        formats = (fmt, fmt, fmt)
        scales = (_AITER.no_scale,) * 3
        quantizers = (_AITER.quantize_bf16,) * 3
    elif recipe_id == "fp8":
        fmt = _AITER.native_fp8_format()
        formats = (fmt, fmt, fmt)
        scales = (_AITER.per_tensor_scale,) * 3
        # Q and K take the rotated quantizer and V the plain one, matching mha_v4's own recipe for
        # this row: the rotation spreads outliers across the head dimension before the cast, which
        # only the operands that meet each other in the QK GEMM can share.
        quantizers = (
            _AITER.quantize_fp8_rotated,
            _AITER.quantize_fp8_rotated,
            _AITER.quantize_fp8,
        )
    else:
        raise ValueError(
            f"VSA-H3 AITER recipe must be one of {VSA_H3_AITER_RECIPES}, got {recipe_id!r}"
        )
    return formats, scales, quantizers


@functools.lru_cache(maxsize=None)
def vsa_h3_aiter_row_available(recipe_id: str) -> bool:
    """Whether this device has a 64x64 sorted-sparse MHA v4 row for ``recipe_id``.

    Asked with the recipe's operands rather than for the geometry alone, because a geometry need
    not exist in every precision: 64x64 is BF16 and FP8 only, and the MX rows stop at 256x128.
    """
    if _AITER is None or not torch.cuda.is_available():
        return False
    formats, scales, _ = _recipe_operands(recipe_id)
    operands = _AITER.operands(*formats, *scales)
    return VSA_H3_AITER_TILE in _AITER.block_tiles(operands, _AITER.sparse_mode)


@functools.lru_cache(maxsize=None)
def _warn_once(message: str) -> None:
    """Log once per distinct message, for per-call conditions that hold for a whole run."""
    logger.warning(message)


# The two ends of this path are pure data movement over the whole sequence, and the kernel in the
# middle is the only part that is not. Both are compiled so Inductor emits one pass each instead of
# the three or four an eager expression costs -- measured 1.52 -> 0.26 ms for the gather and
# 0.75 -> 0.10 ms for the epilogue on a 125k-token render. They are compiled here, rather than by
# the caller, because the attention row itself is torch.compiler.disable: nothing outside will
# fuse them. dynamic=False because a run's geometry is fixed, so there is one shape to specialise
# for and the first call's compile amortises over every layer of every step after it.
@torch.compile(dynamic=False)
def _tile_to_bshd(packed_bshd, tiled_to_packed_index, tiled_slot_valid):
    """Gather packed rows into the kernel's padded BSHD tile buffer, in one pass.

    The caller's BHSD tensor transposed is already a BSHD view, so gathering along its sequence
    axis lands in tile order and the kernel's layout at once. Doing it as tile-then-permute, which
    is what the FlexAttention path's helpers compose to, writes the whole tensor twice.

    Padded slots are zeroed rather than left holding whatever row the gather index pointed at,
    because pooling divides by the real token count per tile and would otherwise average in a
    duplicate.
    """
    tiled = packed_bshd.index_select(1, tiled_to_packed_index)
    return torch.where(tiled_slot_valid.view(1, -1, 1, 1), tiled, torch.zeros_like(tiled))


@torch.compile(dynamic=False)
def _untile_and_mix(
    sparse_bhsd, packed_to_tiled_index, compressed, packed_token_tile, gate
):
    """Undo the tiling and add the gated compression branch, in one pass.

    Fused rather than merely faster: eager rounds the compressed-times-gate product to bf16 before
    the add, and this keeps it in fp32 until the one store, which lands within a bf16 ulp of the
    fp32 result where the eager form does not.
    """
    packed = sparse_bhsd.index_select(2, packed_to_tiled_index)
    return packed + compressed.index_select(2, packed_token_tile) * gate


def _pool_bshd(tiled_bshd: torch.Tensor, metadata: MiniMaxH3VSAMetadata):
    """Pool a padded BSHD tile buffer to fp32 ``[B, H, tiles, D]``.

    The BSHD sibling of pool_h3_vsa_tiles, needed because this path never builds the BHSD buffer
    that one takes. Same reduction over the same slots, so it returns the same tile means.
    """
    batch, _, heads, head_dim = tiled_bshd.shape
    pooled = tiled_bshd.view(
        batch, metadata.num_tiles, metadata.tile_elements, heads, head_dim
    ).sum(dim=2, dtype=torch.float32)
    return pooled.permute(0, 2, 1, 3) / metadata.variable_block_sizes.view(1, 1, -1, 1)


def _ragged_lut(kv_indices: torch.Tensor, width: int):
    """Reshape a VSA-H3 key-tile list into the sorted-sparse row's ragged LUT triple.

    ``kv_indices`` is ``[B, H, query tiles, slots]`` with the first ``width`` slots holding this
    query tile's selection and the rest a sentinel; the kernel indexes one flat list by a start and
    a count per (batch, head, query tile), in exactly that row-major order. So the list is the
    buffer viewed flat, the starts are its row stride, and the counts are constant -- the ragged
    form's generality is unused here because every query tile keeps the same number of tiles.

    The sentinel slots are never addressed, since the count stops short of them. They are why the
    starts are a stride rather than a cumulative sum of the counts.
    """
    batch, heads, query_tiles, slots = kv_indices.shape
    rows = batch * heads * query_tiles
    device = kv_indices.device
    lut_start = torch.arange(rows, dtype=torch.int32, device=device) * slots
    lut_count = torch.full((rows,), width, dtype=torch.int32, device=device)
    return kv_indices.reshape(-1), lut_start, lut_count


def aiter_h3_vsa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    metadata: MiniMaxH3VSAMetadata,
    sparsity: float = FASTH3_VSA_SPARSITY,
    *,
    recipe: str,
) -> torch.Tensor:
    """VSA-H3 attention through an AITER 64x64 sorted-sparse row, gate applied.

    Same contract as ``h3_vsa_attention``: packed ``[B, H, S, D]`` in, packed out, with the
    compression branch and the learned gate already mixed in.

    Unlike the Triton kernel this one cannot read packed rows through the tile map, so the padded
    tile buffers are built here. It also masks whole tiles only, which is what the padding warning
    below is about: a tile holding fewer than 64 real tokens has zeroed key slots, and a zeroed key
    scores exactly q.0 = 0 rather than -inf, so it takes softmax mass while adding nothing to the
    numerator.
    """
    if _AITER is None:
        raise RuntimeError(
            "VSA-H3 on AITER requires an aiter build with mha_v4_packed; please update AITER"
        )
    head_dim = query.shape[-1]
    if head_dim != VSA_H3_AITER_HEAD_DIM:
        raise ValueError(
            f"VSA-H3 on AITER needs head dimension {VSA_H3_AITER_HEAD_DIM}, got {head_dim}"
        )
    if metadata.num_prefix_partial_tiles or (
        metadata.num_full_video_tiles != metadata.num_video_tiles
    ):
        _warn_once(
            "This VSA-H3 tiling leaves padded key slots, which a per-tile block mask cannot mask "
            "out: a padded key scores 0 rather than -inf and takes softmax mass. Measured 2.8e-02 "
            "relative L2 against FlexAttention on a default render, where the same row on a "
            "tiling with no padding measures 2.7e-03."
        )

    tiled = [
        _tile_to_bshd(
            tensor.transpose(1, 2),
            metadata.tiled_to_packed_index,
            metadata.tiled_slot_valid,
        )
        for tensor in (query, key, value)
    ]

    # Pooled from the unquantized tiles, in fp32, so the retained tile set is the one the reference
    # policy picks rather than one a quantization step perturbed.
    pooled_query, pooled_key, pooled_value = (
        _pool_bshd(tensor, metadata) for tensor in tiled
    )
    kv_indices = build_h3_vsa_kv_list(pooled_query, pooled_key, metadata, sparsity)
    width = metadata.num_prefix_tiles + compute_h3_vsa_topk(
        sparsity, metadata.num_video_tiles
    )
    kv_block_indices, lut_start, lut_count = _ragged_lut(kv_indices, width)

    formats, scales, quantizers = _recipe_operands(recipe)
    # The tile buffers are already in the kernel's BSHD layout, so a rotated operand is rotated
    # along the head dimension it will meet in the GEMM. Zero rows stay zero under both
    # quantizers, so padded slots keep scoring 0 rather than drifting to some other constant.
    operands = [
        quantize(tensor) for quantize, tensor in zip(quantizers, tiled, strict=True)
    ]
    sparse_output = _AITER.packed(
        *(tensor for tensor, _ in operands),
        *(descale for _, descale in operands),
        *formats,
        *scales,
        softmax_scale=head_dim**-0.5,
        kv_block_indices=kv_block_indices,
        lut_start=lut_start,
        lut_count=lut_count,
        block_tile=VSA_H3_AITER_TILE,
    )

    # Model dtype for the compression branch, matching flex_h3_vsa_attention: that keeps the
    # [tiles, tiles] probability matrix inside a flash kernel instead of materialising it.
    compressed = F.scaled_dot_product_attention(
        pooled_query.to(query.dtype),
        pooled_key.to(query.dtype),
        pooled_value.to(query.dtype),
    )
    return _untile_and_mix(
        sparse_output.transpose(1, 2),
        metadata.packed_to_tiled_index,
        compressed.to(query.dtype),
        metadata.packed_token_tile,
        gate,
    )
