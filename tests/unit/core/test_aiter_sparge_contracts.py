"""MHA v4 Sparge: the block mask must reach the kernel, at the right tile."""

from types import SimpleNamespace

import pytest
import torch

from xfuser.core.attention import registry
from xfuser.core.attention.spec import AttentionBackendType, AttnCall, Sparsity

_MHA_V4_SPARGE_BACKENDS = (
    "AITER_BF16_SPARGE",
    "AITER_BF16FP8_SPARGE",
    "AITER_I8FP8_SPARGE",
    "AITER_FP8_SPARGE",
    "AITER_MXFP8_SPARGE",
    "AITER_F8F6_SPARGE",
    "AITER_MXFP6_SPARGE",
    "AITER_F6F4_SPARGE",
    "AITER_MXFP4_SPARGE",
    "AITER_F4F4_SPARGE",
)


def _spec(name):
    return registry.get(AttentionBackendType[name])


def _require(name):
    """Skip unless this machine can run the backend, per its own spec."""
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    unavailable = _spec(name).unavailable()
    if unavailable is not None:
        pytest.skip(f"{name}: {unavailable}")


def _run(name, query, key, value, **kwargs):
    spec = _spec(name)
    spec.resolved()  # as backend selection does, before any compile
    return spec.run(query, key, value, AttnCall(**kwargs))


# ---------------------------------------------------------------------------
# what the table declares
# ---------------------------------------------------------------------------


def test_every_mha_v4_sparge_row_is_registered():
    # Ten rows: the eight that predate the BF16 pair, plus BF16 and BF16FP8.
    # Counted so that a row lost from the FORMATS table is a failure here
    # rather than a backend that quietly stops existing.
    assert len(_MHA_V4_SPARGE_BACKENDS) == 10
    for name in _MHA_V4_SPARGE_BACKENDS:
        spec = _spec(name)
        assert spec.sparsity is Sparsity.SPARGE
        assert spec.head_balanced
        assert spec.impl.target == "kernel:mha_v4_sparge"


def test_triton_sparge_backends_are_separate_from_mha_v4():
    """AITER_SPARGE/_V2 are the Sage-kernel sparge path, not MHA v4."""
    for name in ("AITER_SPARGE", "AITER_SPARGE_V2"):
        spec = _spec(name)
        assert spec.sparsity is Sparsity.SPARGE
        assert spec.impl.target.startswith("kernel:sparge")
        assert "aiter_sage" in spec.package


def test_mxfp8_sparge_is_gfx950_only():
    """Block-scaled Q/K has no gfx942 kernel, so the row declares the arch it
    needs rather than being refused once the launch fails."""
    from xfuser.core.attention.backends.aiter_mha_v4.spec import FORMATS

    mxfp8 = next(f for f in FORMATS if f.name == "MXFP8")
    assert mxfp8.sparge_on.names == ("gfx950",)

    fp8 = next(f for f in FORMATS if f.name == "FP8")
    assert fp8.sparge_on.names == ("gfx950", "gfx942")


@pytest.mark.parametrize("backend_name", _MHA_V4_SPARGE_BACKENDS)
def test_sparge_rejects_causal_and_dropout(backend_name):
    """Declared in `accepts`, so the refusal happens before the kernel runs."""
    spec = _spec(backend_name)
    tensor = torch.empty((1, 1, 1, 128))

    assert spec.rejects(tensor, tensor, tensor, AttnCall(is_causal=True)) is not None
    assert spec.rejects(tensor, tensor, tensor, AttnCall(dropout_p=0.1)) is not None


# ---------------------------------------------------------------------------
# what the kernel does with the mask
# ---------------------------------------------------------------------------


@pytest.mark.accelerator
@pytest.mark.rocm
def test_sparge_passes_the_block_mask_and_tile_to_the_kernel(monkeypatch):
    _require("AITER_FP8_SPARGE")
    from xfuser.core.attention.backends.aiter_mha_v4 import kernel

    captured = {}

    def fake_build(
        query,
        key,
        value,
        *,
        is_causal,
        config,
        block_m,
        block_n,
        ulysses_world_size,
        cost_sink,
        pad_block_divisible=False,
    ):
        captured["tile"] = (block_m, block_n)
        captured["pad_block_divisible"] = pad_block_divisible
        mask = torch.ones((query.shape[0], query.shape[1], 2, 4), dtype=torch.bool)
        return query, key, value, SimpleNamespace(), mask

    def fake_mha_v4(query, key, value, *formats, block_mask=None, **kwargs):
        captured["layout"] = tuple(query.shape)
        captured["block_mask"] = block_mask
        return torch.zeros_like(query)

    monkeypatch.setattr(kernel, "build_block_mask", fake_build)
    monkeypatch.setattr(kernel, "restore_sparge_output", lambda output, state: output)
    monkeypatch.setattr(kernel, "mha_v4", fake_mha_v4)

    query = torch.zeros((1, 2, 512, 128), device="cuda", dtype=torch.bfloat16)
    output, lse = _run("AITER_FP8_SPARGE", query, query, query)

    assert lse is None
    assert output.shape == query.shape
    geometry = kernel.SPARSE_GEOMETRY["FP8"]
    assert captured["tile"] == (256, geometry.kv_tile)
    assert captured["pad_block_divisible"] is not geometry.ragged_kv
    assert captured["layout"] == (1, 512, 2, 128)  # BSHD for the kernel
    assert tuple(captured["block_mask"].shape) == (1, 2, 2, 4)


@pytest.mark.accelerator
@pytest.mark.rocm
@pytest.mark.parametrize("ragged_kv", [False, True])
def test_sparge_tile_and_pad_follow_the_rows_geometry(monkeypatch, ragged_kv):
    """The mask is built at the row's own sparse geometry; a mismatch would
    mask the wrong keys rather than fail. Only a row that cannot bound a short
    last block has its keys padded."""
    _require("AITER_FP8_SPARGE")
    from xfuser.core.attention.backends.aiter_mha_v4 import kernel

    captured = {}

    def fake_build(query, key, value, *, block_m, block_n, pad_block_divisible=False, **kwargs):
        captured["tile"] = (block_m, block_n)
        captured["pad_block_divisible"] = pad_block_divisible
        mask = torch.ones((query.shape[0], query.shape[1], 2, 4), dtype=torch.bool)
        return query, key, value, SimpleNamespace(), mask

    geometry = {**kernel.SPARSE_GEOMETRY, "FP8": kernel.SparseGeometry(64, ragged_kv)}
    monkeypatch.setattr(kernel, "SPARSE_GEOMETRY", geometry)
    monkeypatch.setattr(kernel, "build_block_mask", fake_build)
    monkeypatch.setattr(kernel, "restore_sparge_output", lambda output, state: output)
    monkeypatch.setattr(kernel, "mha_v4", lambda q, k, v, *a, **kw: torch.zeros_like(q))

    query = torch.zeros((1, 2, 512, 128), device="cuda", dtype=torch.bfloat16)
    _run("AITER_FP8_SPARGE", query, query, query)

    assert captured["tile"] == (256, 64)
    assert captured["pad_block_divisible"] is not ragged_kv


@pytest.mark.accelerator
@pytest.mark.rocm
@pytest.mark.parametrize(
    "backend_name, expected_kv_tile",
    [("AITER_BF16_SPARGE", 64), ("AITER_BF16FP8_SPARGE", 64), ("AITER_FP8_SPARGE", 128)],
)
def test_sparge_cuts_its_mask_at_its_own_recipes_kv_tile(monkeypatch, backend_name, expected_kv_tile):
    """gfx950 routes BF16 Q/K on 64-key blocks and the rest on 128, so one tile
    cannot serve all. A mask cut at the other recipe's tile has the wrong number
    of KV columns and AITER refuses it."""
    _require(backend_name)
    from xfuser.core.attention.backends.aiter_mha_v4 import kernel
    from xfuser.core.attention.requirements import device_arch

    if "gfx950" not in device_arch():
        pytest.skip("the per-recipe KV tiles differ on gfx950 only")
    captured = {}

    def fake_build(query, key, value, *, block_n, **kwargs):
        captured["block_n"] = block_n
        mask = torch.ones((query.shape[0], query.shape[1], 2, 4), dtype=torch.bool)
        return query, key, value, SimpleNamespace(), mask

    monkeypatch.setattr(kernel, "build_block_mask", fake_build)
    monkeypatch.setattr(kernel, "restore_sparge_output", lambda output, state: output)
    monkeypatch.setattr(kernel, "mha_v4", lambda q, k, v, *a, **kw: torch.zeros_like(q))

    query = torch.zeros((1, 2, 512, 128), device="cuda", dtype=torch.bfloat16)
    _run(backend_name, query, query, query)

    assert captured["block_n"] == expected_kv_tile


# Every block selected and nothing reordered, so the launch should be exact
# attention up to its own quantisation.
_DENSE_SPARGE = {
    "spargeattn_simthreshold": 2.0,
    "spargeattn_cdfthreshold": 1.0,
    "spargeattn_reorder_sequence": False,
    "use_spargeattn_static_block_mask": False,
}


def _ragged_qkv(seq_len=1030):
    """A length no block size divides, with a short last block on every tile."""
    torch.manual_seed(0)
    return [torch.randn((1, 2, seq_len, 128), device="cuda", dtype=torch.bfloat16) for _ in range(3)]


@pytest.mark.accelerator
@pytest.mark.rocm
@pytest.mark.parametrize("backend_name", _MHA_V4_SPARGE_BACKENDS)
def test_sparge_runs_at_a_ragged_length(backend_name):
    """Each row is cut at its own KV tile. gfx950's rows differ (BF16 on 64
    keys, the rest on 128), and AITER refuses a mask cut at another row's."""
    _require(backend_name)
    query, key, value = _ragged_qkv()
    expected = torch.nn.functional.scaled_dot_product_attention(query.float(), key.float(), value.float())

    output, _ = _run(backend_name, query, key, value, attention_kwargs=dict(_DENSE_SPARGE))

    assert output.shape == query.shape
    cosine = torch.nn.functional.cosine_similarity(output.float().flatten(), expected.flatten(), dim=0)
    assert cosine > 0.97


@pytest.mark.accelerator
@pytest.mark.rocm
def test_ragged_sparge_attends_no_padding_keys():
    """A zero pad key scores 0 and takes softmax mass. BF16 has no quantisation
    to hide that behind, so its output matches exact attention closely only
    when the tail goes unpadded."""
    _require("AITER_BF16_SPARGE")
    from xfuser.core.attention.backends.aiter_mha_v4 import kernel

    if not kernel.SPARSE_GEOMETRY["BF16"].ragged_kv:
        pytest.skip("this GPU's BF16 sparse row needs its keys padded")
    query, key, value = _ragged_qkv()
    expected = torch.nn.functional.scaled_dot_product_attention(query.float(), key.float(), value.float())

    output, _ = _run("AITER_BF16_SPARGE", query, key, value, attention_kwargs=dict(_DENSE_SPARGE))

    # Padded to the tile, this is about 3e-2.
    assert (output.float() - expected).norm() / expected.norm() < 1e-2
