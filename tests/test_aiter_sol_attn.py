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
        _kv_tile,
        _quantize,
        _RECIPES,
        sol_attn_routing_for,
    )

    tile = _kv_tile()
    query, key, value = _operands(seqlen=8 * tile, heads=2)
    recipe = _RECIPES["fp8"]
    q, k, v = _quantize(
        recipe,
        *(x.permute(0, 2, 1, 3).contiguous() for x in (query, key, value)),
        query.shape[-1] ** -0.5,
    )

    # A band the size of one block, as a minority modality would be.
    exact_tokens = torch.zeros(8 * tile, dtype=torch.bool, device="cuda")
    exact_tokens[3 * tile : 4 * tile] = True
    forced = _force_blocks_from_tokens(exact_tokens, None, 8 * tile)
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
    # _kv_tile() reads the manifest for the device, so this needs one even though the check below
    # is pure tensor bookkeeping.
    _require_sol_attn()

    from xfuser.core.sparse_attention.sol import _force_blocks_from_tokens, _kv_tile

    import torch

    tile = _kv_tile()
    # Two real blocks plus a partial third, trimmed from a caller pad, then padded back to tile.
    tokens = torch.zeros(3 * tile, dtype=torch.bool)
    tokens[2 * tile : 2 * tile + 5] = True
    forced = _force_blocks_from_tokens(tokens, 2 * tile + 5, 3 * tile)
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


def _kv_block(key):
    """Number of pooled KV blocks in a BHSD key, at the tile this device's manifest row uses."""
    from xfuser.core.sparse_attention.sol import _kv_tile

    tile = _kv_tile()
    return -(-key.shape[2] // tile)


def _operands(seqlen=1024, heads=2, head_dim=128, seed=1234, sharpness=2.0):
    """BHSD bf16 operands with real block structure, which is the layout the backends take.

    Independent Gaussian noise is the wrong input for judging any block-sparse kernel: attention
    over it is close to uniform, so every block carries similar mass, no selection can exploit
    anything, and the measured gap to dense says more about the data than the kernel. Here the keys
    sit in contiguous per-block clusters and each query tile targets one of them, so there really is
    a block to find.

    sharpness sets cluster separation relative to the noise, and 2.0 is deliberate. Push it higher
    and the per-tensor fp8 scale is set by the cluster centers, which coarsens everything else --
    measured, the DENSE fp8 sibling falls to 0.37 cosine against fp32 at sharpness 4 while Sol-Attn
    holds 0.92, so a dense comparison up there would be measuring the reference falling apart.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    # Round the cluster counts up so a seqlen that is not a whole number of blocks still gets a
    # center for its short last block; _spread trims the overhang. Exact for an aligned seqlen.
    num_kv_blocks = -(-seqlen // 128)
    centers = torch.randn(num_kv_blocks, head_dim, generator=g, device="cuda") * sharpness

    def _spread(rows):
        x = rows[:seqlen] + torch.randn(seqlen, head_dim, generator=g, device="cuda")
        return x.unsqueeze(0).unsqueeze(0).expand(1, heads, seqlen, head_dim).contiguous()

    key = _spread(centers.repeat_interleave(128, 0))
    targets = torch.arange(-(-seqlen // 256), device="cuda") % num_kv_blocks
    query = _spread(centers[targets].repeat_interleave(256, 0))
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


def test_sol_attn_is_no_less_accurate_than_dense_fp8(monkeypatch):
    """Across cluster separations, including where the dense fp8 sibling itself breaks down.

    This is the claim that survives outside the well-conditioned regime the test above needs. Once
    the per-tensor fp8 scale is set by widely separated clusters, dense fp8 loses the rest of the
    distribution, while Sol-Attn computes only the blocks its routing picked and recovers the others
    from pooled K/V -- measured at sharpness 4, dense fp8 holds 0.37 cosine to fp32 and Sol-Attn
    0.92. So the correction is not merely cheaper than attending densely, it is also better
    conditioned, and that should not silently regress.
    """
    _require_sol_attn()

    from xfuser.core.distributed import attention_backend as ab
    from xfuser.core.distributed.attention_backend import (
        ATTENTION_FUNCTION_REGISTRY,
        AttentionBackendType,
    )

    monkeypatch.setattr(ab, "get_ulysses_parallel_world_size", lambda: 1)
    monkeypatch.setattr(ab, "get_ring_parallel_world_size", lambda: 1)

    for sharpness in (2.0, 4.0, 8.0):
        query, key, value = _operands(sharpness=sharpness)
        reference = _fp32_attention(query, key, value)
        with torch.no_grad():
            sol, _ = ATTENTION_FUNCTION_REGISTRY[AttentionBackendType.AITER_FP8_SOL](
                query, key, value, dropout_p=0.0, is_causal=False,
                attention_kwargs={"solattn_beta": 0.5},
            )
            dense, _ = ATTENTION_FUNCTION_REGISTRY[AttentionBackendType.AITER_FP8](
                query, key, value, dropout_p=0.0, is_causal=False
            )
        sol_cosine, dense_cosine = _cosine(sol, reference), _cosine(dense, reference)
        assert sol_cosine > dense_cosine - 1e-3, (
            f"at sharpness {sharpness} Sol-Attn tracks fp32 at {sol_cosine:.5f} against dense fp8's "
            f"{dense_cosine:.5f}"
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
    num_kv_blocks = _kv_block(key)
    num_q_tiles = (query.shape[2] + 255) // 256
    assert (cost_sink <= num_q_tiles * num_kv_blocks).all()


@pytest.mark.parametrize(
    ("recipe", "dense_backend"),
    [("fp8", "AITER_FP8"), ("i8fp8", "AITER_I8FP8"),
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


@pytest.mark.parametrize("recipe", ["fp8", "i8fp8", "mxfp8", "mxfp4"])
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


@pytest.mark.parametrize("recipe", ["fp8", "i8fp8", "mxfp8", "mxfp4"])
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
