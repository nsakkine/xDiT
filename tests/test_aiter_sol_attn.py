"""Tests for the AITER Sol-Attn attention backends.

Sol-Attn (arXiv 2607.24027) runs an AITER MHA v4 mode-2 row over the KV blocks its routing selects
and recovers the rest from pooled per-block K/V under the same online softmax, so a dropped block
costs its higher-order terms rather than all of its mass.

Four rows are built -- fp8, i8fp8, mxfp8 and mxfp4 -- differing only in how Q/K/V are quantized.
Where a test is about the mechanism rather than one row's precision it is parametrized over all
four, because the MX rows reach the kernel by a different route: they have no raw entry point and
their block-granular scales do not survive pooling, so they pool in dequantized space and hand the
kernel a pooled scale of its own.

It differs from the Sparge backends in ways these tests pin down: it routes for itself instead of
through a Sparge block mask, it is gfx950-only, and it rejects ring parallelism outright because
merging partial outputs by LSE stops being valid once each rank has added its own correction.
"""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


def _require_sol_attn():
    """Skip unless this build and device can actually run the Sol-Attn kernel."""
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("Sol-Attn requires a ROCm GPU.")

    arch_name = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    arch = next((a for a in ("gfx942", "gfx950") if arch_name.startswith(a)), None)
    if arch is None:
        pytest.skip(f"Sol-Attn ships gfx942 and gfx950 kernels, got {arch_name}.")

    try:
        import aiter
        from aiter.ops.mha_v4 import mha_v4_sol_attn  # noqa: F401
        from aiter.ops.triton.attention.utils import sol_attn_prepare  # noqa: F401
    except ImportError:
        pytest.skip("AITER does not expose the Sol-Attn API.")

    # gfx942 keeps its code objects one level down, in an MI300 subdirectory.
    fwd_dir = Path(aiter.__file__).resolve().parent.parent / "hsa" / arch / "fmha_v4_fwd"
    if arch == "gfx942":
        fwd_dir = fwd_dir / "MI300"
    if not (fwd_dir / "fwd_hd128_fp8_sol_attn.co").exists():
        pytest.skip(f"AITER does not include the {arch} Sol-Attn FMHA kernel.")


def _require_sol_recipe(recipe):
    """Skip unless this device has a Sol-Attn row for recipe.

    A plain skip, not an expected failure: aiter's MX quantizer aborts the process rather than
    raising on an unsupported device, so reaching one of those rows here takes the whole session
    down instead of failing a test.
    """
    from xfuser.core.sparse_attention.sol import SolAttnUnsupported, check_sol_attn_recipe

    try:
        check_sol_attn_recipe(recipe)
    except SolAttnUnsupported as error:
        pytest.skip(str(error))


def _recipes_on_this_device():
    """The Sol-Attn recipe ids this arch builds rows for, in declaration order."""
    from xfuser.core.sparse_attention import sol

    allowed = sol._ARCH_RECIPES.get(sol._device_arch())
    return [r for r in sol.SOL_ATTN_RECIPES if allowed is None or r in allowed]


def _serves(recipe, tile):
    """Whether this GPU has a Sol-Attn kernel at `tile` for `recipe`'s operands."""
    from xfuser.core.sparse_attention import sol

    operands = sol._recipe_operands(sol._RECIPES[recipe])
    return sol._kv_tile_for_q_tile(tile[0], operands, sol._AITER.sol_attn_mode) == tile[1]


def _default_tile(recipe):
    """The (q_tile, kv_tile) `recipe`'s kernel defaults to, whatever the override currently says.

    Asked per recipe because the arch has no one answer to give: on gfx950 the BF16 rows route on
    a 64-token block and every other recipe on 128. Asked of aiter rather than through
    sol_attn_block_tile because that one honours the override, which the tests below set and
    unset -- reading the default through it would return whatever was last forced.
    """
    from xfuser.core.sparse_attention import sol

    operands = sol._recipe_operands(sol._RECIPES[recipe])
    return sol._default_block_tile(operands, sol._AITER.sol_attn_mode)


def test_sol_attn_drops_a_caller_alignment_pad():
    """key_seqlen must reproduce attention over the real tokens, whatever the pad holds.

    MiniMax-H3 pads its packed sequence to 64 rows and reports the real count as varlen metadata.
    Sol-Attn takes no mask, so the pad is dropped rather than attended. With a zero pad this
    changes nothing, because zero is what the tile pad would have added anyway; the test pins
    both that and the non-zero case, which is the one a bias on the QKV projection would produce.
    """
    _require_sol_attn()
    import torch

    from xfuser.core.sparse_attention.sol import sol_attn_bhsd

    real, align = 3970, 64
    padded = -(-real // align) * align
    torch.manual_seed(0)

    def bhsd(seq):
        return torch.randn(1, 4, seq, 128, device="cuda", dtype=torch.bfloat16)

    query = bhsd(512)
    for fill in ("zeros", "nonzero"):
        key, value = bhsd(padded), bhsd(padded)
        if fill == "zeros":
            key[:, :, real:] = 0
            value[:, :, real:] = 0
        dropped, _ = sol_attn_bhsd(query, key, value, beta=0.5, key_seqlen=real)
        unpadded, _ = sol_attn_bhsd(
            query, key[:, :, :real], value[:, :, :real], beta=0.5
        )
        assert torch.equal(dropped, unpadded), (
            f"dropping a {fill} pad did not reproduce attention over the real tokens"
        )


def test_forced_blocks_are_added_to_what_routing_picked():
    """A named token's block must be computed exactly however routing scored it.

    This is the lever for a packed multimodal sequence, where a small modality is thresholded
    against statistics the large one writes. Forcing can only add blocks, never remove them, so
    the result moves toward dense rather than away.
    """
    _require_sol_attn()
    import torch

    from xfuser.core.sparse_attention.sol import (
        _force_blocks_from_tokens,
        _quantize,
        _RECIPES,
        sol_attn_block_tile,
        sol_attn_routing_for,
    )

    recipe = _RECIPES["fp8"]
    tile = sol_attn_block_tile(recipe)[1]
    query, key, value = _operands(seqlen=8 * tile, heads=2)
    q, k, v = _quantize(
        recipe,
        *(x.permute(0, 2, 1, 3).contiguous() for x in (query, key, value)),
        query.shape[-1] ** -0.5,
    )

    # A band the size of one block, as a minority modality would be.
    exact_tokens = torch.zeros(8 * tile, dtype=torch.bool, device="cuda")
    exact_tokens[3 * tile : 4 * tile] = True
    forced = _force_blocks_from_tokens(exact_tokens, None, 8 * tile, recipe)
    assert forced.tolist() == [False, False, False, True, False, False, False, False]

    routed = sol_attn_routing_for(q, k, v, 0.5, recipe)["block_attn_mask"]
    pinned = sol_attn_routing_for(q, k, v, 0.5, recipe, force_blocks=forced)[
        "block_attn_mask"
    ]

    assert pinned[..., 3].all(), "the named block must be exact for every query tile"
    assert (pinned | routed).equal(pinned), "forcing must only ever add blocks"
    assert pinned.sum() >= routed.sum()


def test_forced_blocks_follow_the_same_trim_and_pad_as_kv():
    """The token mask has to stay in step with K/V or it names the wrong blocks."""
    # sol_attn_block_tile() reads the manifest for the device, so this needs one even though the
    # check below is pure tensor bookkeeping.
    _require_sol_attn()

    from xfuser.core.sparse_attention.sol import (
        _force_blocks_from_tokens,
        _RECIPES,
        sol_attn_block_tile,
    )

    import torch

    recipe = _RECIPES["fp8"]
    tile = sol_attn_block_tile(recipe)[1]
    # Two real blocks plus a partial third, trimmed from a caller pad, then padded back to tile.
    tokens = torch.zeros(3 * tile, dtype=torch.bool)
    tokens[2 * tile : 2 * tile + 5] = True
    forced = _force_blocks_from_tokens(tokens, 2 * tile + 5, 3 * tile, recipe)
    assert forced.tolist() == [False, False, True]


def test_the_backend_hands_named_tokens_to_the_routing(monkeypatch):
    """The last link: the dispatch has to pass the model's mask on to sol_attn_bhsd.

    Tested because the bug this pairs with was of exactly this shape -- every piece correct on its
    own, and one connection between them never made.
    """
    import torch

    from xfuser.core.distributed import attention_backend as backend_module
    from xfuser.core.distributed.attention_backend import (
        SOL_EXACT_TOKENS_KEY,
        _aiter_sol_attn_call,
    )
    from xfuser.core.sparse_attention import sol as sol_module

    seen = {}

    def _record(query, key, value, **kwargs):
        seen.update(kwargs)
        return torch.zeros_like(query), None

    monkeypatch.setattr(sol_module, "sol_attn_bhsd", _record)
    monkeypatch.setattr(backend_module, "get_ring_parallel_world_size", lambda: 1)

    exact_tokens = torch.zeros(256, dtype=torch.bool)
    exact_tokens[:8] = True
    q = torch.zeros(1, 2, 256, 8)
    _aiter_sol_attn_call(
        q, q, q, 0.0, False,
        {SOL_EXACT_TOKENS_KEY: exact_tokens, "solattn_beta": 0.25},
        "fp8",
    )

    assert seen.get("exact_tokens") is exact_tokens, (
        "the named tokens did not reach the routing"
    )
    # The beta the caller asked for has to arrive too, rather than the default.
    assert seen.get("beta") == 0.25


def test_sol_attn_refuses_multi_sequence_varlen():
    """One packed sequence is fine; several in one call are not, and must not be approximated."""
    import torch

    from xfuser.core.distributed.attention_backend import _sol_attn_key_seqlen
    from xfuser.core.sparse_attention.sol import SolAttnUnsupported

    assert _sol_attn_key_seqlen({}) is None
    assert (
        _sol_attn_key_seqlen(
            {"cu_seqlens_k": torch.tensor([0, 4000]), "max_seqlen_k": 4000}
        )
        == 4000
    )
    with pytest.raises(SolAttnUnsupported, match="one sequence per call"):
        _sol_attn_key_seqlen(
            {"cu_seqlens_k": torch.tensor([0, 1000, 4000]), "max_seqlen_k": 3000}
        )


def _kv_block(key, recipe="fp8"):
    """Number of pooled KV blocks in a BHSD key, at the tile this recipe's manifest row uses."""
    return -(-key.shape[2] // _default_tile(recipe)[1])


def _operands(seqlen=1024, heads=2, head_dim=128, seed=1234, sharpness=2.0, cluster=128):
    """BHSD bf16 operands with real block structure, which is the layout the backends take.

    Independent Gaussian noise is the wrong input for judging any block-sparse kernel: attention
    over it is close to uniform, so every block carries similar mass, no selection can exploit
    anything, and the measured gap to dense says more about the data than the kernel. Here the keys
    sit in contiguous per-block clusters and each query tile targets one of them, so there really is
    a block to find.

    sharpness sets cluster separation relative to the noise, and 2.0 is deliberate. Push it higher
    and the per-tensor fp8 scale is set by the cluster centers, which coarsens everything else --
    measured at 2048 tokens, fp8 falls from 0.98 cosine against fp32 at sharpness 2 to 0.82 at 8,
    and MXFP4 from 0.82 to 0.12. So a test that reads an absolute cosine wants the default, and one
    that raises sharpness has to compare each row against its own dense sibling instead, which
    moves with it.

    cluster is how wide one group of related tokens is, and it defaults to the default kernel's KV
    tile so that the structure lands on tile boundaries. Lowering it below a tile is what separates
    the geometries: selection is per tile, so a cluster narrower than one drags in its neighbours.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    # Round the cluster counts up so a seqlen that is not a whole number of clusters still gets a
    # center for its short last one; _spread trims the overhang. Exact for an aligned seqlen.
    num_clusters = -(-seqlen // cluster)
    centers = torch.randn(num_clusters, head_dim, generator=g, device="cuda") * sharpness

    def _spread(rows):
        x = rows[:seqlen] + torch.randn(seqlen, head_dim, generator=g, device="cuda")
        return x.unsqueeze(0).unsqueeze(0).expand(1, heads, seqlen, head_dim).contiguous()

    key = _spread(centers.repeat_interleave(cluster, 0))
    # Queries change target half as often as the keys change cluster, so that a run of queries has
    # one block to find rather than every query wanting its own.
    targets = torch.arange(-(-seqlen // (2 * cluster)), device="cuda") % num_clusters
    query = _spread(centers[targets].repeat_interleave(2 * cluster, 0))
    value = torch.randn(1, heads, seqlen, head_dim, generator=g, device="cuda")
    return query.bfloat16(), key.bfloat16(), value.bfloat16()


def _fp32_attention(query, key, value):
    q, k, v = query.float(), key.float(), value.float()
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / (q.shape[-1] ** 0.5)
    return torch.einsum("bhqk,bhkd->bhqd", scores.softmax(dim=-1), v)


def _cosine(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _packed_modalities(seqlen, heads, band, majority_gain, seed=99):
    """BHSD operands shaped like a packed multimodal sequence, plus the minority's token mask.

    Clustered per block as _operands is, so there is a block to find, with two departures that
    together reproduce what a small modality runs into. The minority band sits inside a query tile
    rather than on its boundary, so the one averaged query that decides the tile's selection is
    mostly majority rows; and the majority carries the larger activations, which is what lets it
    outvote the band in that average. majority_gain is that imbalance.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    tile = 128
    centers = torch.randn(-(-seqlen // tile), 128, generator=g, device="cuda") * 2.0

    def _spread(rows):
        x = rows[:seqlen] + torch.randn(seqlen, 128, generator=g, device="cuda")
        return x.unsqueeze(0).unsqueeze(0).expand(1, heads, seqlen, 128).contiguous()

    key = _spread(centers.repeat_interleave(tile, 0))
    value = torch.randn(1, heads, seqlen, 128, generator=g, device="cuda")

    # Every query targets its own block, so the band's rows do want the band's keys.
    query = _spread(centers.repeat_interleave(tile, 0))
    gain = torch.full((seqlen,), majority_gain, device="cuda")
    gain[band] = 1.0
    query = query * gain.view(1, 1, seqlen, 1)

    exact_tokens = torch.zeros(seqlen, dtype=torch.bool, device="cuda")
    exact_tokens[band] = True
    return query.bfloat16(), key.bfloat16(), value.bfloat16(), exact_tokens


def test_pinning_recovers_a_band_that_routing_outvotes():
    """The point of forcing blocks, measured: a minority modality gets its own keys back.

    Selection for a 256-row query tile is decided by one averaged query and a threshold taken over
    every KV block, so a band that is a fraction of its tile and quieter than its neighbours loses
    the blocks it most needed, and falls back to pooled means covering the whole sequence. Naming
    it restores it. The majority is checked too: forcing only ever adds blocks, so it must not
    move.
    """
    _require_sol_attn()

    from xfuser.core.sparse_attention.sol import sol_attn_bhsd

    seqlen, heads = 4096, 4
    # One 128-row band, a few percent of the sequence, buried mid query tile.
    band = slice(2048 + 128, 2048 + 256)
    query, key, value, exact_tokens = _packed_modalities(
        seqlen, heads, band, majority_gain=8.0
    )
    reference = _fp32_attention(query, key, value)

    routed, _ = sol_attn_bhsd(query, key, value, beta=1.0)
    pinned, _ = sol_attn_bhsd(query, key, value, beta=1.0, exact_tokens=exact_tokens)

    band_routed = _cosine(routed[:, :, band], reference[:, :, band])
    band_pinned = _cosine(pinned[:, :, band], reference[:, :, band])
    assert band_pinned > band_routed + 0.01, (
        f"pinning the band did not improve it: {band_routed:.5f} -> {band_pinned:.5f}"
    )

    rest = slice(3072, 3584)
    rest_routed = _cosine(routed[:, :, rest], reference[:, :, rest])
    rest_pinned = _cosine(pinned[:, :, rest], reference[:, :, rest])
    assert rest_pinned >= rest_routed - 1e-4, (
        f"pinning a band cost the majority accuracy: {rest_routed:.5f} -> {rest_pinned:.5f}"
    )


def test_sol_attn_backends_are_registered():
    """One backend per built row, each naming its recipe the way the rest of mha_v4 does.

    The names matter beyond taste: --attention_backend resolves by enum MEMBER name, so these are
    the CLI strings. They read recipe-then-variant (aiter_fp8_sol) to match aiter_fp8_sparge rather
    than inventing a second convention for the same family.
    """
    from xfuser.core.distributed.attention_backend import (
        AITER_MHA_V4_SOL_BACKENDS,
        ATTENTION_FUNCTION_REGISTRY,
        AttentionBackendType,
    )
    from xfuser.core.sparse_attention.sol import SOL_ATTN_RECIPES

    assert len(AITER_MHA_V4_SOL_BACKENDS) == len(SOL_ATTN_RECIPES)
    for backend in AITER_MHA_V4_SOL_BACKENDS:
        assert backend in ATTENTION_FUNCTION_REGISTRY
        recipe = backend.name.removeprefix("AITER_").removesuffix("_SOL").lower()
        assert recipe in SOL_ATTN_RECIPES, f"{backend.name} names no known recipe"
    assert AttentionBackendType.AITER_FP8_SOL.value == "AITER FP8 Sol"


def test_sol_attn_backends_are_head_balanced():
    """They route for themselves, but publish the same per-head cost the balancer consumes."""
    from xfuser.core.distributed.attention_backend import AITER_MHA_V4_SOL_BACKENDS
    from xfuser.model_executor.layers.usp import _HEAD_BALANCE_BACKENDS

    assert set(AITER_MHA_V4_SOL_BACKENDS) <= _HEAD_BALANCE_BACKENDS


def test_sol_attn_rejects_causal_and_dropout(monkeypatch):
    """Both are refused before any kernel work, so neither needs a GPU to check.

    Causal is not a missing feature but a conflict: the pooled correction assumes every skipped
    block is fully attendable, which a causal mask breaks.
    """
    from xfuser.core.distributed import attention_backend as ab
    from xfuser.core.distributed.attention_backend import (
        ATTENTION_FUNCTION_REGISTRY,
        AttentionBackendType,
    )

    monkeypatch.setattr(ab, "get_ring_parallel_world_size", lambda: 1)
    call = ATTENTION_FUNCTION_REGISTRY[AttentionBackendType.AITER_FP8_SOL]
    tensor = torch.zeros((1, 2, 8, 128), dtype=torch.bfloat16)

    with pytest.raises(NotImplementedError):
        call(tensor, tensor, tensor, dropout_p=0.1, is_causal=False)

    from xfuser.core.sparse_attention.sol import SolAttnUnsupported

    with pytest.raises(SolAttnUnsupported, match="causal"):
        call(tensor, tensor, tensor, dropout_p=0.0, is_causal=True)


def test_sol_attn_rejects_ring_parallelism(monkeypatch):
    """LSE merging is not valid once each rank has added a pooled correction of its own."""
    from xfuser.core.distributed import attention_backend as ab
    from xfuser.core.distributed.attention_backend import (
        ATTENTION_FUNCTION_REGISTRY,
        AttentionBackendType,
    )
    from xfuser.core.sparse_attention.sol import SolAttnUnsupported

    monkeypatch.setattr(ab, "get_ring_parallel_world_size", lambda: 2)
    call = ATTENTION_FUNCTION_REGISTRY[AttentionBackendType.AITER_FP8_SOL]
    tensor = torch.zeros((1, 2, 8, 128), dtype=torch.bfloat16)

    with pytest.raises(SolAttnUnsupported, match="ring"):
        call(tensor, tensor, tensor, dropout_p=0.0, is_causal=False)


def test_sol_attn_rejects_non_bf16(monkeypatch):
    from xfuser.core.distributed import attention_backend as ab
    from xfuser.core.distributed.attention_backend import (
        ATTENTION_FUNCTION_REGISTRY,
        AttentionBackendType,
    )
    from xfuser.core.sparse_attention.sol import SolAttnUnsupported

    monkeypatch.setattr(ab, "get_ring_parallel_world_size", lambda: 1)
    call = ATTENTION_FUNCTION_REGISTRY[AttentionBackendType.AITER_FP8_SOL]
    tensor = torch.zeros((1, 2, 8, 128), dtype=torch.float16)

    with pytest.raises(SolAttnUnsupported, match="bf16"):
        call(tensor, tensor, tensor, dropout_p=0.0, is_causal=False)


def test_sol_attn_tracks_the_dense_fp8_sibling(monkeypatch):
    """Against AITER_FP8, which is the same fp8 row over every block.

    That sibling is the right target rather than an fp32 reference: it shares the quantization and
    the rotation, so what is left in the gap is the routing plus the pooled correction.
    """
    _require_sol_attn()

    from xfuser.core.distributed import attention_backend as ab
    from xfuser.core.distributed.attention_backend import (
        ATTENTION_FUNCTION_REGISTRY,
        AttentionBackendType,
    )

    monkeypatch.setattr(ab, "get_ulysses_parallel_world_size", lambda: 1)
    monkeypatch.setattr(ab, "get_ring_parallel_world_size", lambda: 1)

    query, key, value = _operands()
    with torch.no_grad():
        sol, sol_lse = ATTENTION_FUNCTION_REGISTRY[AttentionBackendType.AITER_FP8_SOL](
            query, key, value, dropout_p=0.0, is_causal=False,
            attention_kwargs={"solattn_beta": 0.5},
        )
        dense, _ = ATTENTION_FUNCTION_REGISTRY[AttentionBackendType.AITER_FP8](
            query, key, value, dropout_p=0.0, is_causal=False
        )

    assert sol_lse is None, "Sol-Attn cannot return an LSE; ring parallelism is refused instead"
    assert sol.shape == dense.shape
    assert sol.dtype == torch.bfloat16
    assert torch.isfinite(sol).all()
    cosine = _cosine(sol, dense)
    assert cosine > 0.99, f"cosine to the dense fp8 sibling {cosine}"


@pytest.mark.parametrize("seqlen", [1023, 1008, 960, 897])
def test_sol_attn_takes_a_seqlen_that_is_not_a_whole_number_of_kv_blocks(seqlen):
    """Wan is ragged at every standard size, so this is the common case rather than a corner.

    720p is 21 latent frames x 80 x 45 = 75600 tokens, i.e. 590.6 KV blocks, and aiter refuses a
    ragged seqlen_k on the LUT rows outright. sol_attn_bhsd pads K/V up to the tile; what this
    pins is that the pad stays invisible -- the output keeps Q's real length, and accuracy does not
    sag as the pad grows, which it would if the zero keys were taking real softmax mass. 897 pads
    by 127 of 1024, far past anything Wan asks for, and still has to hold.
    """
    _require_sol_attn()

    from xfuser.core.sparse_attention.sol import sol_attn_bhsd

    query, key, value = _operands(seqlen=seqlen)
    with torch.no_grad():
        out, _ = sol_attn_bhsd(query, key, value, beta=0.5)

    assert out.shape == query.shape, "the pad must not reach the output"
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
    cosine = _cosine(out, _fp32_attention(query, key, value))
    assert cosine > 0.95, f"cosine to fp32 over the real tokens {cosine} at seqlen {seqlen}"


# Per-format distance a row is allowed from its dense sibling, measured at the geometry below.
# BF16's is four orders tighter than the rest because nothing rounds on that row: with no
# quantization the pooled correction reproduces the dense result outright, so anything it loses is
# float accumulation order and not the algorithm. The quantized rows are bounded by how far their
# pooled means round away from the blocks they stand in for, which is why MXFP4 -- eight magnitude
# levels, and the only row that also quantizes V into them -- gets an order more room.
_DENSE_SIBLING_TOLERANCE = {
    "bf16": 1e-6,
    "bf16fp8": 2e-3,
    "fp8": 2e-3,
    "i8fp8": 2e-3,
    "mxfp8": 2e-3,
    "mxfp4": 2e-2,
}


@pytest.mark.parametrize(
    ("recipe", "dense_backend"),
    [("bf16", "AITER_BF16"), ("bf16fp8", "AITER_BF16FP8"),
     ("fp8", "AITER_FP8"), ("i8fp8", "AITER_I8FP8"),
     ("mxfp8", "AITER_MXFP8"), ("mxfp4", "AITER_MXFP4")],
)
def test_every_sol_row_holds_its_dense_sibling_across_cluster_separations(
    recipe, dense_backend, monkeypatch
):
    """Each row against its own dense sibling as the clusters pull apart, not at one separation.

    The single-separation test above is the wiring check; this is the conditioning one. Cluster
    separation is what sets the per-tensor scales, so raising it coarsens everything outside the
    cluster centers and drives each format toward the regime where its pooled means stop
    representing the blocks they stand in for. A row that is wired correctly and still degrades
    only there would pass above and fail here.

    BF16 is the row that makes this readable. It has no quantization to blame, and it reproduces
    its dense sibling to within a bf16 ulp at every separation, which says the selection and the
    correction are together exact on these operands -- so the distance every other row shows is
    its format rounding and nothing else. That is also why this asserts a two-sided band against
    fp32 rather than that Sol-Attn beats dense: measured, fp8 and i8fp8 come out slightly ahead of
    their dense siblings here and bf16fp8, mxfp8 and mxfp4 slightly behind, so a one-sided claim
    would be asserting a coincidence.

    Run at 2048 tokens rather than the 1024 the rest of the file uses. That is 16 KV blocks, which
    is _MIN_USEFUL_KV_BLOCKS: below it sol.py itself warns that a mean-plus-sigma threshold over
    the blocks says little, and a comparison there grades the threshold's luck rather than the
    kernel.
    """
    _require_sol_attn()
    _require_sol_recipe(recipe)

    from xfuser.core.distributed import attention_backend as ab
    from xfuser.core.distributed.attention_backend import (
        ATTENTION_FUNCTION_REGISTRY,
        AttentionBackendType,
    )
    from xfuser.core.sparse_attention.sol import sol_attn_bhsd

    monkeypatch.setattr(ab, "get_ulysses_parallel_world_size", lambda: 1)
    monkeypatch.setattr(ab, "get_ring_parallel_world_size", lambda: 1)

    tolerance = _DENSE_SIBLING_TOLERANCE[recipe]
    for sharpness in (2.0, 4.0, 8.0):
        query, key, value = _operands(seqlen=2048, sharpness=sharpness)
        reference = _fp32_attention(query, key, value)
        with torch.no_grad():
            sol, _ = sol_attn_bhsd(query, key, value, beta=0.5, recipe=recipe)
            dense, _ = ATTENTION_FUNCTION_REGISTRY[
                AttentionBackendType[dense_backend]
            ](query, key, value, dropout_p=0.0, is_causal=False)

        assert sol.shape == dense.shape and sol.dtype == torch.bfloat16
        assert torch.isfinite(sol).all()

        sibling = _cosine(sol, dense)
        assert 1.0 - sibling < tolerance, (
            f"at sharpness {sharpness} the {recipe} Sol-Attn row tracks its dense sibling at "
            f"{sibling:.8f}"
        )

        sol_cosine, dense_cosine = _cosine(sol, reference), _cosine(dense, reference)
        assert abs(sol_cosine - dense_cosine) < tolerance, (
            f"at sharpness {sharpness} the {recipe} Sol-Attn row tracks fp32 at {sol_cosine:.6f} "
            f"against its dense sibling's {dense_cosine:.6f}"
        )


def test_sol_attn_publishes_a_head_cost_without_changing_the_output(monkeypatch):
    """Requesting the head cost moves the call onto aiter's packed API.

    That is a different entry point reached with operands quantized here rather than inside the raw
    one, so the two have to be shown to agree -- otherwise turning head balancing on would silently
    change what the model computes.
    """
    _require_sol_attn()

    from xfuser.core.distributed import attention_backend as ab
    from xfuser.core.distributed.attention_backend import (
        ATTENTION_FUNCTION_REGISTRY,
        AttentionBackendType,
    )
    from xfuser.core.sparse_attention.head_balance import COST_SINK_KEY

    monkeypatch.setattr(ab, "get_ulysses_parallel_world_size", lambda: 1)
    monkeypatch.setattr(ab, "get_ring_parallel_world_size", lambda: 1)

    heads = 2
    query, key, value = _operands(heads=heads)
    call = ATTENTION_FUNCTION_REGISTRY[AttentionBackendType.AITER_FP8_SOL]
    cost_sink = torch.zeros(heads, device="cuda", dtype=torch.float32)

    with torch.no_grad():
        plain, _ = call(
            query, key, value, dropout_p=0.0, is_causal=False,
            attention_kwargs={"solattn_beta": 0.5},
        )
        balanced, _ = call(
            query, key, value, dropout_p=0.0, is_causal=False,
            attention_kwargs={"solattn_beta": 0.5, COST_SINK_KEY: cost_sink},
        )

    assert torch.equal(plain, balanced), (
        "the packed path taken for head cost must reproduce the raw path exactly"
    )
    assert (cost_sink > 0).all(), "every head should select at least one block"
    num_kv_blocks = _kv_block(key, "fp8")
    num_q_tiles = (query.shape[2] + 255) // 256
    assert (cost_sink <= num_q_tiles * num_kv_blocks).all()


@pytest.mark.parametrize(
    ("recipe", "dense_backend"),
    [("bf16", "AITER_BF16"), ("bf16fp8", "AITER_BF16FP8"),
     ("fp8", "AITER_FP8"), ("i8fp8", "AITER_I8FP8"),
     ("mxfp8", "AITER_MXFP8"), ("mxfp4", "AITER_MXFP4")],
)
def test_every_sol_recipe_tracks_its_own_dense_sibling(recipe, dense_backend):
    """Each row against the dense row of the SAME recipe, which is the only fair target.

    Comparing every row to fp32 would grade the quantization, not the wiring: MXFP4 carries eight
    magnitude levels, so it sits near 0.79 of fp32 no matter how correct Sol-Attn is -- and dense
    MXFP4 sits there too. Against its own dense sibling each row has to be near-identical, because
    Sol-Attn computes the selected blocks exactly and recovers the others, so the only difference
    left is the correction's error rather than anything about the format.

    This is what catches a miswired recipe. Handing the kernel the wrong format, scale mode or
    pooled scale still produces finite output that correlates with the reference; it just quietly
    loses accuracy, and only the comparison against the matched dense row makes that visible.
    """
    _require_sol_attn()
    _require_sol_recipe(recipe)

    from xfuser.core.distributed import attention_backend as ab
    from xfuser.core.distributed.attention_backend import (
        ATTENTION_FUNCTION_REGISTRY,
        AttentionBackendType,
    )
    from xfuser.core.sparse_attention.sol import sol_attn_bhsd

    for name in ("get_ulysses_parallel_world_size", "get_ring_parallel_world_size"):
        setattr(ab, name, lambda: 1)

    query, key, value = _operands()
    with torch.no_grad():
        sol, _ = sol_attn_bhsd(query, key, value, beta=0.5, recipe=recipe)
        dense, _ = ATTENTION_FUNCTION_REGISTRY[AttentionBackendType[dense_backend]](
            query, key, value, dropout_p=0.0, is_causal=False
        )

    assert sol.shape == dense.shape and sol.dtype == torch.bfloat16
    assert torch.isfinite(sol).all()
    cosine = _cosine(sol, dense)
    assert cosine > 0.99, f"{recipe} Sol-Attn to its dense sibling {cosine}"

    # And it must not be losing ground to the format's own dense row against the truth.
    reference = _fp32_attention(query, key, value)
    assert _cosine(sol, reference) > _cosine(dense, reference) - 0.01


@pytest.mark.parametrize("recipe", ["bf16", "bf16fp8", "fp8", "i8fp8", "mxfp8", "mxfp4"])
def test_every_sol_recipe_publishes_a_head_cost(recipe):
    """The balancer consumes this for every row, not just the one with a raw entry point."""
    _require_sol_attn()
    _require_sol_recipe(recipe)

    from xfuser.core.sparse_attention.sol import sol_attn_bhsd

    heads = 4
    query, key, value = _operands(heads=heads)
    with torch.no_grad():
        out, cost = sol_attn_bhsd(query, key, value, beta=0.5, recipe=recipe,
                                  return_head_cost=True)

    assert cost.shape == (heads,) and cost.dtype == torch.float32
    assert (cost > 0).all(), "every head should select at least one block"
    num_q_tiles = (query.shape[2] + 255) // 256
    assert (cost <= num_q_tiles * (key.shape[2] // 128)).all()
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("recipe", ["bf16", "bf16fp8", "fp8", "i8fp8", "mxfp8", "mxfp4"])
@pytest.mark.parametrize("seqlen", [1024, 1008])
@pytest.mark.parametrize("head_cost", [False, True])
def test_sol_attn_holds_one_graph(recipe, seqlen, head_cost):
    """fullgraph=True, because a break here lands in the middle of every attention layer.

    xDiT compiles without fullgraph, so a break does not fail the run -- it splits the block into
    two compiled regions and quietly gives back the fusion, which is only visible in a profile.
    Every row is covered because they take different branches: the ragged seqlen reaches the K/V
    padding, the head cost swaps the raw entry point for the packed one, and the MX rows carry
    their own quantizers and pooled scales through it.

    The last regression here was aiter decorating a helper with functools.cache and this module
    reaching it through a SimpleNamespace attribute, where Dynamo binds the owner as self. Eager
    could not see it, which is exactly why it is worth a test.
    """
    _require_sol_attn()
    _require_sol_recipe(recipe)

    from xfuser.core.sparse_attention.sol import sol_attn_bhsd

    query, key, value = _operands(seqlen=seqlen)

    def run(q, k, v):
        out, _ = sol_attn_bhsd(q, k, v, beta=0.5, return_head_cost=head_cost, recipe=recipe)
        return out

    torch._dynamo.reset()
    with torch.no_grad():
        compiled = torch.compile(run, fullgraph=True)(query, key, value)
        eager = run(query, key, value)

    assert torch.equal(compiled, eager), "compiling must not change the result"


def test_sol_attn_is_equivariant_under_a_head_permutation():
    """Permuting heads must permute the cost and the output and change nothing else.

    This is the contract head balancing rests on. It reorders heads before the input all-to-all so
    each rank gets a cost-balanced subset, then inverts that on the output, which is only sound if
    the permutation is a pure relabelling: the cost the balancer plans next step from has to follow
    its head, and the inverse has to put the output back exactly. Sol-Attn routes per head and the
    routing never mixes them, so this should hold to the bit rather than approximately.
    """
    _require_sol_attn()

    from xfuser.core.sparse_attention.sol import sol_attn_bhsd

    heads = 8
    query, key, value = _operands(heads=heads)
    perm = torch.randperm(heads, device="cuda")
    inverse = torch.argsort(perm)

    with torch.no_grad():
        out, cost = sol_attn_bhsd(query, key, value, beta=0.5, return_head_cost=True)
        permuted, permuted_cost = sol_attn_bhsd(
            query.index_select(1, perm), key.index_select(1, perm),
            value.index_select(1, perm), beta=0.5, return_head_cost=True,
        )

    assert cost.shape == (heads,) and cost.dtype == torch.float32, (
        "the cost sink apply_head_balance allocates is float32 (local_nheads_q,)"
    )
    assert torch.equal(cost[perm], permuted_cost), "cost must follow its head"
    assert torch.equal(out.index_select(1, perm), permuted), "output must follow its head"
    assert torch.equal(permuted.index_select(1, inverse), out), "revert must be exact"


def test_sol_attn_beta_controls_sparsity(monkeypatch):
    """A higher threshold has to select strictly fewer blocks, or the knob is not wired through.

    Deliberately run on UNCLUSTERED operands, the one place they are the right input. The clustered
    ones the accuracy tests use are so cleanly separated that every beta lands on the same single
    matching block, which is the floor sol_attn_prepare enforces anyway, leaving the threshold
    nothing to move. A smooth score distribution is what makes the knob observable.
    """
    _require_sol_attn()

    from xfuser.core.distributed import attention_backend as ab
    from xfuser.core.distributed.attention_backend import (
        ATTENTION_FUNCTION_REGISTRY,
        AttentionBackendType,
    )
    from xfuser.core.sparse_attention.head_balance import COST_SINK_KEY

    monkeypatch.setattr(ab, "get_ulysses_parallel_world_size", lambda: 1)
    monkeypatch.setattr(ab, "get_ring_parallel_world_size", lambda: 1)

    heads = 2
    query, key, value = _operands(heads=heads, sharpness=0.0)
    call = ATTENTION_FUNCTION_REGISTRY[AttentionBackendType.AITER_FP8_SOL]

    costs = {}
    for beta in (0.0, 1.5):
        sink = torch.zeros(heads, device="cuda", dtype=torch.float32)
        with torch.no_grad():
            call(
                query, key, value, dropout_p=0.0, is_causal=False,
                attention_kwargs={"solattn_beta": beta, COST_SINK_KEY: sink},
            )
        costs[beta] = sink.sum().item()

    assert costs[1.5] < costs[0.0], f"beta did not tighten the selection: {costs}"


def test_a_scheduled_beta_routes_exactly_as_the_same_number_would(monkeypatch):
    """--solattn_beta_schedule hands the backends a 0-d tensor where --solattn_beta hands a float.

    It has to be the same threshold to the last bit, or a schedule would silently mean something
    other than the betas it was given, and the flat-beta runs it gets compared against would not be
    a baseline. A tensor is what the schedule passes because reading it as a number happens inside
    the compiled forward, where it costs a graph break and a recompile per distinct beta; routing
    consumes it as a scalar operand of tau = mean_j(proxy) + beta * std_j(proxy), which is why the
    substitution is possible at all.
    """
    _require_sol_attn()

    from types import SimpleNamespace

    from xfuser.core.distributed import attention_backend as ab
    from xfuser.core.distributed import runtime_state
    from xfuser.core.distributed.attention_backend import (
        ATTENTION_FUNCTION_REGISTRY,
        AttentionBackendType,
    )
    from xfuser.core.sparse_attention.head_balance import COST_SINK_KEY

    monkeypatch.setattr(ab, "get_ulysses_parallel_world_size", lambda: 1)
    monkeypatch.setattr(ab, "get_ring_parallel_world_size", lambda: 1)

    heads = 2
    query, key, value = _operands(heads=heads, sharpness=0.0)
    call = ATTENTION_FUNCTION_REGISTRY[AttentionBackendType.AITER_FP8_SOL]

    def run(scheduled):
        """One call, routed either from the flag or from a schedule standing in for it."""
        monkeypatch.setattr(runtime_state, "_RUNTIME",
                            SimpleNamespace(scheduled_solattn_beta=scheduled), raising=False)
        sink = torch.zeros(heads, device="cuda", dtype=torch.float32)
        with torch.no_grad():
            out, _ = call(
                query, key, value, dropout_p=0.0, is_causal=False,
                attention_kwargs={"solattn_beta": 0.25, COST_SINK_KEY: sink},
            )
        return out, sink

    # Before comparing anything: the first Sol-Attn call of a process autotunes the pooling kernels,
    # and a config picked on timing noise moves a near-threshold block, which reads as a difference
    # between the two betas when it is a difference between a cold and a warm cache.
    run(None)

    from_flag, cost_from_flag = run(None)
    from_schedule, cost_from_schedule = run(torch.tensor(0.25, dtype=torch.float32))

    assert torch.equal(cost_from_flag, cost_from_schedule), (
        "the scheduled beta selected a different number of blocks than the float: "
        f"{cost_from_flag.tolist()} vs {cost_from_schedule.tolist()}")
    assert torch.equal(from_flag, from_schedule), "same threshold, so the output must be identical"


def _override_tile(monkeypatch, tile):
    """Point the module at one geometry, as XFUSER_SOL_ATTN_BLOCK_TILE would have at import.

    Setting the environment variable in-process would do nothing: it is parsed once at import into
    _BLOCK_TILE_OVERRIDE, deliberately, so that a traced call never reads os.environ. Patching the
    parsed constant is therefore what exercises the same path a real override takes.
    """
    from xfuser.core.sparse_attention import sol

    monkeypatch.setattr(sol, "_BLOCK_TILE_OVERRIDE", tile)


@pytest.mark.parametrize("spec,expected", [("64x64", (64, 64)), ("256X128", (256, 128))])
def test_the_block_tile_env_var_parses_to_a_tile(monkeypatch, spec, expected):
    """Both cases, because the value is lowercased before splitting on the x."""
    from xfuser.core.sparse_attention import sol

    monkeypatch.setenv("XFUSER_SOL_ATTN_BLOCK_TILE", spec)
    assert sol._read_block_tile_override() == expected


def test_a_malformed_block_tile_env_var_says_what_it_wanted(monkeypatch):
    """A typo here would otherwise land as an unpacking error with no mention of the variable."""
    from xfuser.core.sparse_attention import sol

    monkeypatch.setenv("XFUSER_SOL_ATTN_BLOCK_TILE", "64,64")
    with pytest.raises(ValueError, match="must be QxKV"):
        sol._read_block_tile_override()


def test_an_unset_block_tile_takes_the_kernel_default(monkeypatch):
    """Unset is the shipped configuration, and it must not pin a geometry of its own.

    Checked once per recipe, because the kernel default is now per recipe: gfx950's BF16 rows
    route on a 64-token block and the rest on 128. Deferring to aiter per recipe is the whole
    property -- a geometry pinned here would be right for at most one of them.
    """
    _require_sol_attn()

    from aiter.ops.mha_v4 import mha_v4_block_tile
    from xfuser.core.sparse_attention import sol

    _override_tile(monkeypatch, None)
    for recipe in _recipes_on_this_device():
        operands = sol._recipe_operands(sol._RECIPES[recipe])
        assert sol.sol_attn_block_tile(sol._RECIPES[recipe]) == mha_v4_block_tile(
            operands, sol._AITER.sol_attn_mode
        ), f"the '{recipe}' recipe does not take aiter's default for its own row"


def test_a_block_tile_with_no_kernel_is_rejected_by_name(monkeypatch):
    """Named at the variable that set it, not as a missing manifest row several frames down."""
    _require_sol_attn()

    from xfuser.core.sparse_attention.sol import SolAttnUnsupported, check_sol_attn_supported

    _override_tile(monkeypatch, (128, 64))
    query, key, value = _operands(seqlen=256)
    with pytest.raises(SolAttnUnsupported, match="XFUSER_SOL_ATTN_BLOCK_TILE"):
        check_sol_attn_supported(query, key, value, is_causal=False)


def test_the_64x64_override_routes_and_dispatches_at_64(monkeypatch):
    """End to end at the finer geometry: it has to route at 64 and hold accuracy.

    Accuracy alone would pass with the override ignored and the default kernel running, so the
    block count carries the proof. It is reported in (query tile x KV block) pairs, so naming the
    same content on the finer grid costs strictly more of them; ignoring the override would return
    the default run's number exactly. Routing, pooling and dispatch therefore all moved together.

    The output is deliberately not part of that proof, because for a recipe whose default already
    routes at 64 keys it is bit-identical to the default run: refining only the query tile leaves
    every row accumulating over the same KV tiles in the same order, so the two kernels agree to
    the last bit. That is a fact about the arithmetic, not evidence the override did nothing --
    the block counts, which come out four times larger, are what separate the two.

    What is deliberately NOT asserted is that 64x64 is more accurate. On clustered operands the
    pooled correction already recovers most of what coarse selection drops, so both geometries land
    at the same cosine and the finer one keeps more tokens getting there; the case for it is models
    whose routing is genuinely finer than a 128-token block, which is a different input than this.

    Every backend whose recipe has a 64x64 row is exercised, read off the manifest rather than
    named, so a precision that gains one is covered the day it lands instead of leaving the
    geometry proven in FP8 alone.
    """
    _require_sol_attn()

    from xfuser.core.distributed import attention_backend as ab
    from xfuser.core.distributed.attention_backend import (
        AITER_MHA_V4_SOL_RECIPE,
        ATTENTION_FUNCTION_REGISTRY,
    )
    from xfuser.core.sparse_attention import sol
    from xfuser.core.sparse_attention.head_balance import COST_SINK_KEY

    # Per recipe, since a blind enumeration would raise rather than answer: it lists one KV tile
    # per Q tile, and at 256 rows this arch serves two depending on the recipe.
    if not any(_serves(recipe, (64, 64)) for recipe in _recipes_on_this_device()):
        pytest.skip("this build has no 64x64 Sol-Attn row.")

    monkeypatch.setattr(ab, "get_ulysses_parallel_world_size", lambda: 1)
    monkeypatch.setattr(ab, "get_ring_parallel_world_size", lambda: 1)

    heads, seqlen = 2, 2048
    # Clustered at 64, which only the finer geometry can resolve; see _operands.
    query, key, value = _operands(seqlen=seqlen, heads=heads, cluster=64)
    reference = _fp32_attention(query, key, value)

    on_device = _recipes_on_this_device()
    serving = [(backend, recipe) for backend, recipe in AITER_MHA_V4_SOL_RECIPE.items()
               if recipe in on_device and _serves(recipe, (64, 64))]
    assert serving, "this build has a 64x64 Sol-Attn row but no backend reaches it"

    for backend, recipe in serving:
        call = ATTENTION_FUNCTION_REGISTRY[backend]
        # Per recipe, since that is how fine the defaults are now: BF16 defaults to a 64-key
        # block here and the rest to 128.
        default_tile = _default_tile(recipe)
        q_tile, kv_tile = default_tile

        def run(tile):
            _override_tile(monkeypatch, tile)
            assert sol.sol_attn_block_tile(sol._RECIPES[recipe]) == (tile or default_tile)
            sink = torch.zeros(heads, device="cuda", dtype=torch.float32)
            with torch.no_grad():
                out, _ = call(query, key, value, dropout_p=0.0, is_causal=False,
                              attention_kwargs={"solattn_beta": 0.25, COST_SINK_KEY: sink})
            return out, sink.sum().item()

        default, default_blocks = run(None)
        fine, fine_blocks = run((64, 64))

        assert fine_blocks > default_blocks, (
            f"{recipe}: 64x64 reported {fine_blocks:.0f} selected blocks and "
            f"{q_tile}x{kv_tile} reported {default_blocks:.0f}; the finer grid has to cost more "
            "pairs for the same content, so an equal or smaller count means the override never "
            "reached routing")

        cosine, baseline = _cosine(fine, reference), _cosine(default, reference)
        assert cosine > 0.9, f"{recipe}: 64x64 Sol-Attn fell to {cosine:.4f} against fp32 dense"
        assert cosine > baseline - 0.02, (
            f"{recipe}: 64x64 lost ground to {q_tile}x{kv_tile}: {cosine:.4f} vs {baseline:.4f}")


def test_a_block_tile_not_every_recipe_serves_is_rejected_at_setup(monkeypatch):
    """gfx950's 64x64 rows are FP8 and BF16, and the MX/INT8 recipes have to say so by name.

    Which recipes serve 64x64 is read off the manifest rather than named here, so adding a row
    extends the test instead of breaking it. What is asserted is the partition: the tile has to be
    served by some recipe and not by all of them, or neither half below would be checking anything.

    At setup rather than at the first attention call, which is the point: this is the check that
    stands between a mistyped launch and a run that loads a model for minutes before dying. The
    dispatch would catch it too, but only after the load and without naming the variable.
    """
    _require_sol_attn()

    from xfuser.core.sparse_attention import sol
    from xfuser.core.sparse_attention.sol import SolAttnUnsupported, check_sol_attn_recipe

    # Per recipe, since a blind enumeration would raise rather than answer: it lists one KV tile
    # per Q tile, and at 256 rows this arch serves two depending on the recipe.
    if not any(_serves(recipe, (64, 64)) for recipe in _recipes_on_this_device()):
        pytest.skip("this build has no 64x64 Sol-Attn row.")

    _override_tile(monkeypatch, (64, 64))
    recipes = _recipes_on_this_device()
    served = [recipe for recipe in recipes if _serves(recipe, (64, 64))]
    unserved = [recipe for recipe in recipes if recipe not in served]
    assert served, "no recipe serves 64x64, so the acceptance half below checks nothing"
    assert unserved, "every recipe serves 64x64, so the rejection half below checks nothing"

    for recipe in served:
        check_sol_attn_recipe(recipe)
    for recipe in unserved:
        with pytest.raises(SolAttnUnsupported, match=f"XFUSER_SOL_ATTN_BLOCK_TILE.*{recipe}"):
            check_sol_attn_recipe(recipe)


def test_every_recipe_serves_the_default_block_tile(monkeypatch):
    """The unset default has to work everywhere, which is what makes it the safe default.

    Without this the geometry filtering above could narrow to nothing for some recipe and only the
    64x64 tests would notice, since an override is what they set.
    """
    _require_sol_attn()

    from xfuser.core.sparse_attention import sol

    _override_tile(monkeypatch, None)
    for recipe in _recipes_on_this_device():
        tile = _default_tile(recipe)
        assert _serves(recipe, tile), (
            f"the '{recipe}' recipe has no {tile[0]}x{tile[1]} Sol-Attn row, so its default "
            "geometry is not one it serves")
