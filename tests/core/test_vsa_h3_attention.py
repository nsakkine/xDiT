import pytest
import torch

from xfuser.core.vsa_h3_attention import (
    FASTH3_VSA_TILE_ELEMENTS,
    build_h3_vsa_block_mask,
    build_h3_vsa_metadata,
    compute_h3_vsa_topk,
    h3_vsa_selection,
    h3_vsa_tiles_are_full,
    tile_h3_vsa_tensor,
    untile_h3_vsa_tensor,
)


def test_h3_vsa_metadata_keeps_prefix_segments_in_separate_tiles():
    metadata = build_h3_vsa_metadata(
        prefix_segments=(65, 0, 3),
        video_shape=(2, 3, 5),
        device=torch.device("cpu"),
    )

    assert metadata.num_prefix_tiles == 3
    assert metadata.num_video_tiles == 2
    assert metadata.total_seq_length == 98
    assert metadata.variable_block_sizes.tolist() == [64, 1, 3, 24, 6]
    assert metadata.padded_seq_length == 5 * FASTH3_VSA_TILE_ELEMENTS


def test_h3_vsa_tile_round_trip_with_partial_video_edges():
    metadata = build_h3_vsa_metadata(
        prefix_segments=(3, 2),
        video_shape=(5, 5, 5),
        device=torch.device("cpu"),
    )
    tensor = torch.arange(
        metadata.total_seq_length * 2,
        dtype=torch.float32,
    ).reshape(1, metadata.total_seq_length, 1, 2)

    tiled = tile_h3_vsa_tensor(tensor, metadata)
    restored = untile_h3_vsa_tensor(tiled, metadata)

    assert metadata.variable_block_sizes.tolist() == [
        3,
        2,
        64,
        16,
        16,
        4,
        16,
        4,
        4,
        1,
    ]
    assert torch.count_nonzero(
        tiled[
            :,
            torch.tensor(
                [
                    index
                    for index in range(metadata.padded_seq_length)
                    if index not in set(metadata.packed_to_tiled_index.tolist())
                ]
            ),
        ]
    ) == 0
    torch.testing.assert_close(restored, tensor)


def test_h3_vsa_exempt_mask_keeps_prefix_and_top_video_keys():
    scores = torch.zeros(1, 1, 4, 4)
    scores[0, 0, :, 2] = torch.tensor([4.0, 1.0, 3.0, 0.0])
    scores[0, 0, :, 3] = torch.tensor([1.0, 5.0, 2.0, 6.0])

    mask = build_h3_vsa_block_mask(
        scores,
        num_prefix_tiles=2,
        num_video_tiles=2,
        sparsity=0.9,
    )

    assert mask[..., :2].all()
    assert mask.sum(dim=-1).tolist() == [[[3, 3, 3, 3]]]
    assert mask[0, 0, 0, 2]
    assert mask[0, 0, 1, 3]
    assert mask[0, 0, 2, 2]
    assert mask[0, 0, 3, 3]


def test_h3_vsa_dense_mask_at_zero_sparsity():
    scores = torch.randn(2, 3, 5, 5)

    mask = build_h3_vsa_block_mask(
        scores,
        num_prefix_tiles=2,
        num_video_tiles=3,
        sparsity=0.0,
    )

    assert mask.all()


@pytest.mark.parametrize(
    ("sparsity", "tiles", "expected"),
    [(0.0, 10, 10), (0.5, 10, 5), (0.9, 10, 1), (1.0, 10, 1)],
)
def test_h3_vsa_topk(sparsity, tiles, expected):
    assert compute_h3_vsa_topk(sparsity, tiles) == expected


def test_h3_vsa_rejects_non_64_token_geometry():
    with pytest.raises(ValueError, match="64-token"):
        build_h3_vsa_metadata(
            prefix_segments=(4,),
            video_shape=(4, 4, 4),
            device=torch.device("cpu"),
            tile_shape=(2, 2, 2),
        )


@pytest.mark.parametrize(
    ("prefix_segments", "video_shape", "full"),
    [
        # whole-tile prefixes and a grid that divides by 4x4x4 leave no padded slot
        ((128, 64), (8, 16, 16), True),
        # a prefix that is not a multiple of the tile leaves a short remainder tile
        ((130, 64), (8, 16, 16), False),
        # so does a video axis that does not divide by its tile extent
        ((128, 64), (8, 16, 18), False),
    ],
)
def test_h3_vsa_tiles_are_full_tracks_every_source_of_padding(
    prefix_segments, video_shape, full
):
    """The predicate the AITER backend gates on. A kernel that masks whole tiles only cannot
    drive a padded key to -inf, so it needs to know whether any tile is short."""
    metadata = build_h3_vsa_metadata(
        prefix_segments=prefix_segments,
        video_shape=video_shape,
        device=torch.device("cpu"),
    )
    assert h3_vsa_tiles_are_full(metadata) is full
    assert full == bool(
        (metadata.variable_block_sizes == FASTH3_VSA_TILE_ELEMENTS).all()
    )


def test_h3_vsa_selection_is_kernel_independent():
    """Both VSA-H3 backends take their selection and pooled branch from this one helper, so it
    has to answer in tile units and in token units respectively, whoever calls it."""
    metadata = build_h3_vsa_metadata(
        prefix_segments=(64, 64),
        video_shape=(4, 4, 4),
        device=torch.device("cpu"),
    )
    heads, head_dim = 2, 8
    torch.manual_seed(0)
    operands = [
        torch.randn(1, heads, metadata.padded_seq_length, head_dim) for _ in range(3)
    ]

    block_map, compressed = h3_vsa_selection(*operands, metadata)

    assert block_map.shape == (1, heads, metadata.num_tiles, metadata.num_tiles)
    assert block_map.dtype == torch.bool
    # the prefix is exempt from the top-k, so every query tile keeps all of it
    assert block_map[..., : metadata.num_prefix_tiles].all()
    assert compressed.shape == (1, heads, metadata.padded_seq_length, head_dim)


def test_h3_vsa_selection_rejects_unpadded_operands():
    metadata = build_h3_vsa_metadata(
        prefix_segments=(65, 3),
        video_shape=(2, 3, 5),
        device=torch.device("cpu"),
    )
    packed = [torch.randn(1, 2, metadata.total_seq_length, 8) for _ in range(3)]
    with pytest.raises(ValueError, match="padded BHSD"):
        h3_vsa_selection(*packed, metadata)
