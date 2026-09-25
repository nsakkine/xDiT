"""AITER_VSA_H3: FastH3's 64-token tile selection over AITER's BF16 64x64 block-sparse row.

The reference throughout is the FlexAttention backend, which is the shipping implementation of
the same contract. These tests are about the kernel swap, not about VSA-H3's own accuracy.
"""
import pytest
import torch

from xfuser.core.distributed.attention_backend import (
    ATTENTION_FUNCTION_REGISTRY,
    VSA_H3_ATTN_BACKEND_SET,
    AttentionBackendType,
)
from xfuser.core.vsa_h3_attention import (
    build_h3_vsa_metadata,
    h3_vsa_tiles_are_full,
    tile_h3_vsa_tensor,
)

HEADS, HEAD_DIM = 8, 128
# what the aligned path costs against Flex is bf16 reassociation and nothing else; measured at
# 2.7e-03 across aligned geometries, so this leaves room without admitting a real divergence
_ALIGNED_TOLERANCE = 8e-3


def _require_aiter_vsa_h3():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("AITER_VSA_H3 requires a ROCm GPU.")
    from xfuser.core.distributed.attention_backend import _aiter_vsa_h3_row_available

    if not _aiter_vsa_h3_row_available():
        pytest.skip("This device has no BF16 64x64 block-sparse MHA v4 row.")


def _tiled_operands(metadata, seed=0):
    torch.manual_seed(seed)
    packed = [
        torch.randn(
            1,
            metadata.total_seq_length,
            HEADS,
            HEAD_DIM,
            device="cuda",
            dtype=torch.bfloat16,
        )
        for _ in range(3)
    ]
    return [
        tile_h3_vsa_tensor(t, metadata).transpose(1, 2).contiguous() for t in packed
    ]


def _relative_l2(got, want, metadata):
    """Compared over real rows only: padded query rows are dropped on the way out."""
    real = torch.zeros(
        metadata.padded_seq_length, dtype=torch.bool, device=got.device
    )
    real[metadata.packed_to_tiled_index] = True
    got, want = got[:, :, real].float(), want[:, :, real].float()
    return ((got - want).norm() / want.norm()).item()


def test_aiter_vsa_h3_is_registered_as_a_vsa_h3_backend():
    """Both backends serve FastH3, so anything asking "is this VSA-H3" has to see both."""
    assert AttentionBackendType.AITER_VSA_H3 in ATTENTION_FUNCTION_REGISTRY
    assert VSA_H3_ATTN_BACKEND_SET == {
        AttentionBackendType.FLEX_VSA_H3,
        AttentionBackendType.AITER_VSA_H3,
    }

    from xfuser.model_executor.models.runner_models.minimax_h3 import (
        xFuserFastH3Model,
    )

    assert VSA_H3_ATTN_BACKEND_SET <= xFuserFastH3Model._supported_attn_backends


@pytest.mark.parametrize(
    ("prefix_segments", "video_shape"),
    [((256, 128), (8, 16, 16)), ((512, 64), (4, 24, 32))],
)
def test_aiter_vsa_h3_matches_flex_on_aligned_geometry(prefix_segments, video_shape):
    """Whole-tile prefixes and a grid that divides by 4x4x4 leave no padded key, which is the
    case the AITER row can express exactly."""
    _require_aiter_vsa_h3()
    from xfuser.core.distributed.attention_backend import _aiter_h3_vsa_attention
    from xfuser.core.vsa_h3_attention import flex_h3_vsa_attention

    metadata = build_h3_vsa_metadata(
        prefix_segments, video_shape, torch.device("cuda")
    )
    assert h3_vsa_tiles_are_full(metadata), "this geometry was meant to be aligned"
    operands = _tiled_operands(metadata)

    flex_output, flex_compressed = flex_h3_vsa_attention(*operands, metadata)
    aiter_output, aiter_compressed = _aiter_h3_vsa_attention(*operands, metadata)

    # the pooled branch is shared code, so it must agree exactly rather than closely
    assert torch.equal(flex_compressed, aiter_compressed)
    assert torch.isfinite(aiter_output).all()
    assert _relative_l2(aiter_output, flex_output, metadata) < _ALIGNED_TOLERANCE


def test_aiter_vsa_h3_falls_back_to_flex_when_tiles_are_padded():
    """A per-tile block mask cannot drive a padded key to -inf, so its zeroed row would score 0
    and take softmax mass. Measured at 5.5e-01 relative L2 on this geometry, so the backend has
    to decline it rather than run it: the assertion is that the two backends agree, which they
    only can if the AITER one handed the call back to Flex.
    """
    _require_aiter_vsa_h3()

    metadata = build_h3_vsa_metadata((65, 3), (2, 3, 5), torch.device("cuda"))
    assert not h3_vsa_tiles_are_full(metadata), "this geometry was meant to be padded"
    query, key, value = _tiled_operands(metadata)
    gate = torch.zeros_like(query)
    kwargs = {"vsa_h3_metadata": metadata, "vsa_h3_gate": gate}

    flex = ATTENTION_FUNCTION_REGISTRY[AttentionBackendType.FLEX_VSA_H3]
    aiter = ATTENTION_FUNCTION_REGISTRY[AttentionBackendType.AITER_VSA_H3]
    flex_output, _ = flex(query, key, value, 0.0, False, dict(kwargs))
    aiter_output, _ = aiter(query, key, value, 0.0, False, dict(kwargs))

    assert torch.equal(aiter_output, flex_output)


def test_aiter_vsa_h3_falls_back_to_dense_without_metadata():
    """MiniMax-H3's token refiner reaches the backend with no VSA-H3 metadata at all."""
    _require_aiter_vsa_h3()

    query = torch.randn(1, HEADS, 128, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    aiter = ATTENTION_FUNCTION_REGISTRY[AttentionBackendType.AITER_VSA_H3]
    output, _ = aiter(query, query, query, 0.0, False, None)

    assert output.shape == query.shape
    assert torch.isfinite(output).all()


def test_aiter_vsa_h3_rejects_causal_and_dropout():
    _require_aiter_vsa_h3()

    metadata = build_h3_vsa_metadata((256, 128), (8, 16, 16), torch.device("cuda"))
    query, key, value = _tiled_operands(metadata)
    kwargs = {"vsa_h3_metadata": metadata, "vsa_h3_gate": torch.zeros_like(query)}
    aiter = ATTENTION_FUNCTION_REGISTRY[AttentionBackendType.AITER_VSA_H3]

    with pytest.raises(ValueError, match="AITER_VSA_H3 does not support causal"):
        aiter(query, key, value, 0.0, True, dict(kwargs))
    with pytest.raises(ValueError, match="AITER_VSA_H3 does not support attention dropout"):
        aiter(query, key, value, 0.1, False, dict(kwargs))
