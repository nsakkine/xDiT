"""Sol-Attn's host-side bookkeeping: the block tile, the K/V tail, and forced blocks.

The tile is pinned through the parsed override so these need no AITER manifest:
it is the same constant XFUSER_SOL_ATTN_BLOCK_TILE sets, and every site reads it
through sol_attn_block_tile.
"""

import pytest
import torch

from xfuser.core.sparse_attention import sol

TILE = (256, 128)


@pytest.fixture
def pinned_tile(monkeypatch):
    monkeypatch.setattr(sol, "_BLOCK_TILE_OVERRIDE", TILE)
    return TILE[1]


@pytest.mark.parametrize("spec,expected", [("64x64", (64, 64)), ("256X128", (256, 128))])
def test_the_block_tile_env_var_parses_to_a_tile(monkeypatch, spec, expected):
    """Both cases, because the value is lowercased before splitting on the x."""
    monkeypatch.setenv("XFUSER_SOL_ATTN_BLOCK_TILE", spec)
    assert sol._read_block_tile_override() == expected


def test_a_malformed_block_tile_env_var_says_what_it_wanted(monkeypatch):
    """A typo here would otherwise land as an unpacking error with no mention of the variable."""
    monkeypatch.setenv("XFUSER_SOL_ATTN_BLOCK_TILE", "64,64")
    with pytest.raises(ValueError, match="must be QxKV"):
        sol._read_block_tile_override()


@pytest.mark.parametrize("ragged_kv", [True, False])
def test_kv_is_padded_to_the_tile_only_for_a_row_that_cannot_bound_its_last_block(monkeypatch, pinned_tile, ragged_kv):
    """A zero key scores 0 rather than -inf, so a pad the row does not need takes softmax mass.
    gfx950's rows mask a short last block themselves; gfx942's index KV in whole tiles."""
    monkeypatch.setattr(sol, "_takes_ragged_kv", lambda recipe: ragged_kv)
    key = torch.randn(1, 2 * pinned_tile + 5, 2, 128)

    padded_key, padded_value = sol._pad_kv_to_tile(key, key, sol._RECIPES["f6f4"])

    if ragged_kv:
        assert padded_key is key and padded_value is key
    else:
        assert padded_key.shape[1] == 3 * pinned_tile
        torch.testing.assert_close(padded_key[:, : key.shape[1]], key)
        assert not padded_key[:, key.shape[1] :].any()


def test_forced_blocks_follow_kv_through_a_trimmed_pad_onto_a_short_last_block(pinned_tile):
    """The token mask has to stay in step with K/V or it names the wrong blocks. The caller's own
    pad is trimmed first, and what remains need not be a whole number of blocks."""
    tokens = torch.zeros(3 * pinned_tile, dtype=torch.bool)
    tokens[2 * pinned_tile : 2 * pinned_tile + 5] = True
    valid = 2 * pinned_tile + 5

    forced = sol._force_blocks_from_tokens(tokens, valid, valid, sol._RECIPES["fp8"])

    assert forced.tolist() == [False, False, True]


def test_a_flag_only_in_the_trimmed_pad_forces_nothing(pinned_tile):
    """The last real block shares its tile with the trimmed pad, so a flag there must go with it."""
    tokens = torch.zeros(3 * pinned_tile, dtype=torch.bool)
    tokens[-1] = True
    valid = 2 * pinned_tile + 5

    forced = sol._force_blocks_from_tokens(tokens, valid, valid, sol._RECIPES["fp8"])

    assert not forced.any()
