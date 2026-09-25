"""Tests for the AITER VSA-H3 attention backends.

FastH3's VSA-H3 selection is a per-query-tile list of 64-token key tiles, which is what AITER's
sorted-sparse MHA v4 rows walk at the 64x64 geometry -- so the same selection that drives
FlexAttention and the Triton kernel can drive an ASM row instead. Pure sparse (mode 1) rather than
Sol-Attn (mode 2): FastH3 already mixes a pooled branch outside the softmax with a learned gate,
and Sol-Attn would fold a second approximation of the skipped blocks inside it.

Two rows exist, BF16 and per-tensor FP8, because those are the two precisions gfx950 builds at
64x64. They are parametrized together wherever the test is about the mechanism, and separated
wherever it is about precision: BF16 passes its operands through unquantized and so should track
the reference to bf16 rounding, while FP8 carries a real quantization error of its own.

The padding case is the one to read carefully. A tile holding fewer than 64 real tokens has zeroed
key slots, and a block mask that selects whole tiles cannot drive those to -inf: a zeroed key
scores exactly q.0 = 0, so it takes softmax mass while adding nothing to the numerator. These
backends run such a tiling anyway, with a warning, which is what the last tests here pin.
"""

import math
from pathlib import Path

import pytest
import torch


def _require_aiter_vsa_h3(recipe):
    """Skip unless this build and device can run the 64x64 sorted-sparse row for recipe."""
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("VSA-H3 on AITER requires a ROCm GPU.")

    arch_name = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    if not arch_name.startswith("gfx950"):
        pytest.skip(f"The 64x64 MHA v4 rows are gfx950 only, got {arch_name}.")

    try:
        import aiter
        from aiter.ops.mha_v4 import mha_v4_packed  # noqa: F401
    except ImportError:
        pytest.skip("AITER does not expose mha_v4_packed.")

    from xfuser.core.vsa_h3_aiter import vsa_h3_aiter_row_available

    fwd_dir = Path(aiter.__file__).resolve().parent.parent / "hsa" / "gfx950" / "fmha_v4_fwd"
    code_object = f"fwd_hd128_{recipe}_sparse_64x64.co"
    if not (fwd_dir / code_object).exists():
        pytest.skip(f"AITER does not include {code_object}.")
    if not vsa_h3_aiter_row_available(recipe):
        pytest.skip(f"This device has no 64x64 sorted-sparse row for '{recipe}'.")


RECIPES = ("bf16", "fp8")

# mha_v4 is hd128 in every precision, so a VSA-H3 test that reaches the kernel cannot use the
# tiny head the rest of the FastH3 suite runs on.
HEAD_DIM = 128
HEADS = 4

# A tiling with no short tiles: both prefix segments are multiples of the 64-token tile and every
# video axis divides by the 4x4x4 tile shape. This is the geometry the rows are exact on.
#
# The video shape is also chosen so the selection width -- 3 prefix tiles plus 2 of 16 video tiles
# at the default sparsity -- is not a multiple of the index list's alignment. That is what makes
# the buffer's row length differ from the count the kernel reads, so a LUT start strided by the
# count instead of the row length lands on the wrong tiles here. At a width that happens to be
# aligned the two strides coincide and the bug is invisible.
ALIGNED_PREFIX = (128, 64)
ALIGNED_VIDEO = (8, 8, 16)
# One of each kind of straggler: 333 and 77 are not multiples of 64, and none of 7, 9, 11 divides
# by 4, so the tiling carries both partial prefix tiles and partial video tiles.
PADDED_PREFIX = (333, 77)
PADDED_VIDEO = (7, 9, 11)


def _metadata(prefix, video):
    from xfuser.core.vsa_h3_attention import build_h3_vsa_metadata

    return build_h3_vsa_metadata(prefix, video, torch.device("cuda"))


def _selection_width(metadata):
    from xfuser.core.vsa_h3_attention import FASTH3_VSA_SPARSITY, compute_h3_vsa_topk

    return metadata.num_prefix_tiles + compute_h3_vsa_topk(
        FASTH3_VSA_SPARSITY, metadata.num_video_tiles
    )


def test_both_test_geometries_leave_the_index_list_unaligned():
    """Guard the property the accuracy tests silently rest on.

    The index list is padded out to an alignment with a sentinel, so a LUT start strided by the
    selection width rather than by the buffer's row length is only wrong when the two differ.
    Every geometry here has to differ, or a test that passes says nothing about the stride.
    """
    if not torch.cuda.is_available():
        pytest.skip("VSA-H3 metadata is built on the target device.")
    from xfuser.core.vsa_h3_attention import FASTH3_VSA_KV_LIST_ALIGNMENT

    for prefix, video in (
        (ALIGNED_PREFIX, ALIGNED_VIDEO),
        (PADDED_PREFIX, PADDED_VIDEO),
    ):
        width = _selection_width(_metadata(prefix, video))
        assert width % FASTH3_VSA_KV_LIST_ALIGNMENT, (
            f"prefix={prefix} video={video} selects {width} tiles, which is already aligned"
        )


def _operands(metadata, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (1, HEADS, metadata.total_seq_length, HEAD_DIM)
    query, key, value = (
        torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
        for _ in range(3)
    )
    gate = torch.rand(
        (1, HEADS, metadata.total_seq_length, 1),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    return query, key, value, gate


def _relative_l2(actual, reference):
    return (
        (actual.float() - reference.float()).norm() / reference.float().norm()
    ).item()


@pytest.mark.parametrize(
    "prefix, video",
    [(ALIGNED_PREFIX, ALIGNED_VIDEO), (PADDED_PREFIX, PADDED_VIDEO)],
)
def test_the_fused_gather_builds_the_same_buffer_the_shared_helper_does(prefix, video):
    """The one-pass gather has to equal tile-then-permute exactly, not merely closely.

    This path builds the padded tile buffer straight in the kernel's BSHD layout rather than
    composing the shared BHSD helper with a transpose, which halves the bytes written. That is
    only a layout change, so anything other than bit equality means it is also a semantic one --
    most likely in which slots get zeroed, since the padded ones are the difference between a
    correct tile mean and one that averages in a duplicated row.
    """
    if not torch.cuda.is_available():
        pytest.skip("VSA-H3 metadata is built on the target device.")
    from xfuser.core.vsa_h3_aiter import _tile_to_bshd
    from xfuser.core.vsa_h3_attention import tile_h3_vsa_bhsd

    metadata = _metadata(prefix, video)
    query, _, _, _ = _operands(metadata)

    fused = _tile_to_bshd(
        query.transpose(1, 2),
        metadata.tiled_to_packed_index,
        metadata.tiled_slot_valid,
    )
    reference = tile_h3_vsa_bhsd(query, metadata).transpose(1, 2)

    assert fused.is_contiguous()
    assert torch.equal(fused, reference)


@pytest.mark.parametrize(
    "prefix, video",
    [(ALIGNED_PREFIX, ALIGNED_VIDEO), (PADDED_PREFIX, PADDED_VIDEO)],
)
def test_pooling_off_the_bshd_buffer_gives_the_same_tile_means(prefix, video):
    """Selection must see the same pooled tiles here as every other VSA-H3 kernel does.

    The reduction is over the same slots in a different memory order, so it should agree to the
    bit. It has to: the retained tile set is a top-k over these values, and a tie broken the
    other way is a different selection rather than a slightly different number.
    """
    if not torch.cuda.is_available():
        pytest.skip("VSA-H3 metadata is built on the target device.")
    from xfuser.core.vsa_h3_aiter import _pool_bshd, _tile_to_bshd
    from xfuser.core.vsa_h3_attention import pool_h3_vsa_tiles, tile_h3_vsa_bhsd

    metadata = _metadata(prefix, video)
    query, _, _, _ = _operands(metadata)

    pooled = _pool_bshd(
        _tile_to_bshd(
            query.transpose(1, 2),
            metadata.tiled_to_packed_index,
            metadata.tiled_slot_valid,
        ),
        metadata,
    )
    reference = pool_h3_vsa_tiles(tile_h3_vsa_bhsd(query, metadata), metadata)

    assert pooled.shape == reference.shape
    assert torch.equal(pooled, reference)


def test_the_fused_epilogue_is_no_further_from_exact_than_the_plain_one():
    """The fused untile-and-mix must not trade accuracy for the pass it saves.

    It is not bit-identical to the eager expression and is not meant to be: eager rounds the
    compressed-times-gate product to bf16 before adding, and the fused form keeps it in fp32
    until the store. That makes it the closer of the two to the exact value, which is what this
    asserts -- the direction matters, since a fusion that was merely different would be a
    regression dressed up as an optimisation.
    """
    if not torch.cuda.is_available():
        pytest.skip("VSA-H3 metadata is built on the target device.")
    from xfuser.core.vsa_h3_aiter import _untile_and_mix
    from xfuser.core.vsa_h3_attention import untile_h3_vsa_bhsd

    metadata = _metadata(PADDED_PREFIX, PADDED_VIDEO)
    sparse_bshd = torch.randn(
        (1, metadata.padded_seq_length, HEADS, HEAD_DIM),
        device="cuda",
        dtype=torch.bfloat16,
    )
    compressed = torch.randn(
        (1, HEADS, metadata.num_tiles, HEAD_DIM), device="cuda", dtype=torch.bfloat16
    )
    gate = torch.rand(
        (1, HEADS, metadata.total_seq_length, 1), device="cuda", dtype=torch.bfloat16
    )

    def plain(dtype):
        packed = untile_h3_vsa_bhsd(sparse_bshd.transpose(1, 2), metadata).to(dtype)
        return packed + compressed.to(dtype).index_select(
            2, metadata.packed_token_tile
        ) * gate.to(dtype)

    exact = plain(torch.float32)
    fused = _untile_and_mix(
        sparse_bshd.transpose(1, 2),
        metadata.packed_to_tiled_index,
        compressed,
        metadata.packed_token_tile,
        gate,
    )

    assert fused.dtype == torch.bfloat16
    fused_error = (fused.float() - exact).abs().max()
    plain_error = (plain(torch.bfloat16).float() - exact).abs().max()
    assert fused_error <= plain_error
    # One bf16 ulp is a relative 2**-8, which the fused form should be inside everywhere.
    assert ((fused.float() - exact).abs() <= exact.abs() * 2**-8).all()


def test_the_ragged_lut_addresses_the_same_tiles_the_selection_chose():
    """The LUT triple must name, per query tile, exactly the tiles the selection listed.

    This is the whole of what the AITER path adds to the shared selection, and it is pure index
    arithmetic: the kernel reads one flat list by a start and a count per (batch, head, query
    tile), so a stride that disagrees with the buffer's row length, or a count that runs past the
    real entries into the alignment sentinel, would silently address the wrong tiles rather than
    fail. Checked against the buffer it was derived from rather than against the kernel, so it
    runs without one.
    """
    pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("VSA-H3 metadata is built on the target device.")
    from xfuser.core.vsa_h3_aiter import _ragged_lut
    from xfuser.core.vsa_h3_attention import (
        build_h3_vsa_kv_list,
        pool_h3_vsa_tiles,
        tile_h3_vsa_bhsd,
    )

    metadata = _metadata(PADDED_PREFIX, PADDED_VIDEO)
    query, key, _, _ = _operands(metadata)
    pooled_query = pool_h3_vsa_tiles(tile_h3_vsa_bhsd(query, metadata), metadata)
    pooled_key = pool_h3_vsa_tiles(tile_h3_vsa_bhsd(key, metadata), metadata)
    kv_indices = build_h3_vsa_kv_list(pooled_query, pooled_key, metadata)
    width = _selection_width(metadata)

    kv_block_indices, lut_start, lut_count = _ragged_lut(kv_indices, width)

    batch, heads, query_tiles, _ = kv_indices.shape
    assert lut_start.numel() == batch * heads * query_tiles
    assert lut_count.numel() == lut_start.numel()
    assert torch.equal(lut_count, torch.full_like(lut_count, width))
    for row, (b, h, q) in enumerate(
        (b, h, q)
        for b in range(batch)
        for h in range(heads)
        for q in range(query_tiles)
    ):
        start = int(lut_start[row])
        addressed = kv_block_indices[start : start + width]
        assert torch.equal(addressed, kv_indices[b, h, q, :width])
        # Every addressed tile is a real tile, never the sentinel the list is padded with.
        assert int(addressed.max()) < metadata.num_tiles


@pytest.mark.parametrize("recipe", RECIPES)
def test_an_unpadded_tiling_tracks_the_flex_reference(recipe):
    """On a tiling with no short tiles the row computes the same attention Flex does.

    Exactly the same, up to arithmetic: the two walk the identical selection over the identical
    operands, so BF16 should differ only by the order its accumulations happen in. FP8 is held to
    a looser bound because it also quantizes, which is a real difference in the operands and not
    a difference in the walk.
    """
    _require_aiter_vsa_h3(recipe)
    from xfuser.core.vsa_h3_aiter import aiter_h3_vsa_attention
    from xfuser.core.vsa_h3_attention import h3_vsa_attention

    metadata = _metadata(ALIGNED_PREFIX, ALIGNED_VIDEO)
    query, key, value, gate = _operands(metadata)

    reference = h3_vsa_attention(
        query, key, value, gate, metadata, use_triton=False
    )
    actual = aiter_h3_vsa_attention(
        query, key, value, gate, metadata, recipe=recipe
    )

    assert actual.shape == reference.shape
    assert actual.dtype == reference.dtype
    tolerance = 5e-3 if recipe == "bf16" else 1e-1
    assert _relative_l2(actual, reference) < tolerance


@pytest.mark.parametrize("recipe", RECIPES)
def test_a_padded_tiling_runs_and_says_what_it_costs(recipe, caplog):
    """A tiling with short tiles runs rather than falling back, and warns once that it is lossy.

    Running it is the deliberate choice: the alternative kernels are available for exactness, and
    what this row offers is speed on a shape the block mask cannot express perfectly. The warning
    is the contract -- a caller must be able to tell from the log that this run took the error,
    and must not get one line per attention layer per step for it.
    """
    _require_aiter_vsa_h3(recipe)
    import xfuser.core.vsa_h3_aiter as vsa_h3_aiter
    from xfuser.core.vsa_h3_attention import h3_vsa_attention

    vsa_h3_aiter._warn_once.cache_clear()
    metadata = _metadata(PADDED_PREFIX, PADDED_VIDEO)
    query, key, value, gate = _operands(metadata)

    with caplog.at_level("WARNING", logger=vsa_h3_aiter.logger.name):
        actual = vsa_h3_aiter.aiter_h3_vsa_attention(
            query, key, value, gate, metadata, recipe=recipe
        )
        vsa_h3_aiter.aiter_h3_vsa_attention(
            query, key, value, gate, metadata, recipe=recipe
        )

    padding_warnings = [
        record for record in caplog.records if "padded key slots" in record.message
    ]
    assert len(padding_warnings) == 1

    # The padding costs accuracy but not correctness: the result is finite, the right shape, and
    # still pointed the same way as the reference. Only the magnitude moves, because the padded
    # keys inflate the softmax denominator without touching the numerator.
    reference = h3_vsa_attention(
        query, key, value, gate, metadata, use_triton=False
    )
    assert actual.shape == reference.shape
    assert torch.isfinite(actual).all()
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), reference.float().flatten(), dim=0
    )
    assert cosine > 0.99


def test_the_padding_warning_names_a_geometry_that_actually_has_padding():
    """The predicate behind the warning must agree with the tile sizes it summarises.

    It reads tile counts rather than variable_block_sizes so the answer needs no device sync, and
    that indirection is only safe while the two agree. Checked on both geometries, since a
    predicate that is always true and one that is always false both pass a one-sided test.
    """
    if not torch.cuda.is_available():
        pytest.skip("VSA-H3 metadata is built on the target device.")

    for prefix, video, expect_padding in (
        (ALIGNED_PREFIX, ALIGNED_VIDEO, False),
        (PADDED_PREFIX, PADDED_VIDEO, True),
    ):
        metadata = _metadata(prefix, video)
        from_counts = bool(
            metadata.num_prefix_partial_tiles
            or metadata.num_full_video_tiles != metadata.num_video_tiles
        )
        from_sizes = bool(
            (metadata.variable_block_sizes != metadata.tile_elements).any().item()
        )
        assert from_counts == from_sizes == expect_padding


@pytest.mark.parametrize("recipe", RECIPES)
def test_a_head_the_rows_do_not_serve_is_refused_rather_than_reshaped(recipe):
    """mha_v4 is hd128 in every precision, so a narrower head has to fail with the reason.

    The tile geometry and the head dimension are independent, and VSA-H3's own tests run a 16-wide
    head; without this the call would reach aiter and fail on a packed-width check that says
    nothing about VSA-H3.
    """
    _require_aiter_vsa_h3(recipe)
    from xfuser.core.vsa_h3_aiter import aiter_h3_vsa_attention

    metadata = _metadata(ALIGNED_PREFIX, ALIGNED_VIDEO)
    shape = (1, HEADS, metadata.total_seq_length, 64)
    query, key, value = (
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    )
    gate = torch.rand(
        (1, HEADS, metadata.total_seq_length, 1), device="cuda", dtype=torch.bfloat16
    )

    with pytest.raises(ValueError, match="head dimension 128"):
        aiter_h3_vsa_attention(query, key, value, gate, metadata, recipe=recipe)


def test_an_unknown_recipe_names_the_ones_that_exist():
    """Only two precisions are built at 64x64, so a third has to be refused by name."""
    from xfuser.core.vsa_h3_aiter import VSA_H3_AITER_AVAILABLE, _recipe_operands

    if not VSA_H3_AITER_AVAILABLE:
        pytest.skip("AITER does not expose mha_v4_packed.")
    with pytest.raises(ValueError, match="mxfp8"):
        _recipe_operands("mxfp8")


@pytest.mark.parametrize("recipe", RECIPES)
def test_the_selection_varies_per_query_tile(recipe):
    """A selection uniform across query tiles cannot tell a correct LUT walk from a broken one.

    Every query tile keeps all prefix tiles and its own top-k video tiles, so the lists differ by
    construction -- but only if the scores do. This asserts the property the accuracy test above
    silently depends on, because if the top-k came out identical for every query tile then that
    test would pass with the LUT start stuck at row zero.
    """
    _require_aiter_vsa_h3(recipe)
    from xfuser.core.vsa_h3_attention import (
        build_h3_vsa_kv_list,
        pool_h3_vsa_tiles,
        tile_h3_vsa_bhsd,
    )

    metadata = _metadata(ALIGNED_PREFIX, ALIGNED_VIDEO)
    query, key, _, _ = _operands(metadata)
    pooled_query = pool_h3_vsa_tiles(tile_h3_vsa_bhsd(query, metadata), metadata)
    pooled_key = pool_h3_vsa_tiles(tile_h3_vsa_bhsd(key, metadata), metadata)
    kv_indices = build_h3_vsa_kv_list(pooled_query, pooled_key, metadata)

    video = kv_indices[0, 0, :, metadata.num_prefix_tiles :]
    assert not torch.equal(video, video[:1].expand_as(video))


def test_the_backend_rows_are_registered_and_carry_a_recipe_each():
    """Both rows have to be selectable and each has to name the mha_v4 recipe it dispatches."""
    from xfuser.core.distributed.attention_backend import (
        ATTENTION_FUNCTION_REGISTRY,
        VSA_H3_AITER_RECIPE_BY_BACKEND,
        VSA_H3_BACKENDS,
        AttentionBackendType,
    )

    rows = {
        AttentionBackendType.AITER_BF16_VSA_H3: "bf16",
        AttentionBackendType.AITER_FP8_VSA_H3: "fp8",
    }
    assert VSA_H3_AITER_RECIPE_BY_BACKEND == rows
    for backend in rows:
        assert backend in VSA_H3_BACKENDS
        assert backend in ATTENTION_FUNCTION_REGISTRY


@pytest.mark.parametrize(
    "backend, compiles",
    [
        ("AITER_BF16_VSA_H3", False),
        ("AITER_FP8_VSA_H3", False),
        ("TRITON_VSA_H3", True),
    ],
)
def test_torch_compile_is_refused_for_the_aiter_rows_alone(backend, compiles, monkeypatch):
    """These rows disable Dynamo, and FastH3 compiles the transformer at fullgraph.

    A graph break inside a fullgraph region is an error, not a fallback, so the combination has to
    be refused at config time with something a caller can act on rather than at the first forward
    with a Dynamo traceback. The Triton row is here to hold the other side: it does compile, and a
    refusal written against the whole VSA-H3 set would take it down with the AITER pair.
    """
    from types import SimpleNamespace

    from xfuser.model_executor.models.runner_models.minimax_h3 import xFuserFastH3Model

    config = SimpleNamespace(
        attention_backend=backend,
        use_hybrid_attn_schedule=False,
        use_torch_compile=True,
    )
    monkeypatch.setattr(
        xFuserFastH3Model.__mro__[1], "_validate_config", lambda self, config: None
    )
    runner = object.__new__(xFuserFastH3Model)
    if compiles:
        xFuserFastH3Model._validate_config(runner, config)
        return
    with pytest.raises(ValueError, match="use_torch_compile"):
        xFuserFastH3Model._validate_config(runner, config)
