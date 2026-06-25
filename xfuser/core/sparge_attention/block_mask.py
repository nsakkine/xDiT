#   Copyright 2025 Jintao Zhang, Chendong Xiang, Haofeng Huang
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#   This file has been modified from the upstream Apache-2.0 source at
#   https://github.com/thu-ml/SpargeAttn (spas_sage_attn/utils.py).

from typing import Optional

import torch
import triton
import triton.language as tl


# Module-level cache of broadcast threshold tensors keyed on a hashable
# Python tuple (so torch.compile / dynamo can constant-fold the lookup
# and bake the resulting tensor into the graph).
#
# Key layout: (float_value, num_heads, device_type, device_index)
#   - float_value is the scalar broadcast across all heads
#   - num_heads is the target shape[0]
#   - device_type / device_index distinguish per-GPU copies in multi-GPU runs
_HYPER_TENSOR_CACHE: dict[tuple, torch.Tensor] = {}


def _device_key(device: torch.device) -> tuple:
    # `torch.device` instances are not always hashable across versions; use
    # explicit (type, index) so the key is stable and dynamo-friendly.
    return (device.type, device.index if device.index is not None else -1)

def hyperparameter_check(
    hyper: float | torch.Tensor, H: int, device: torch.device
) -> torch.Tensor:
    if isinstance(hyper, (float, int)):
        key = (float(hyper), H, *_device_key(device))
        cached = _HYPER_TENSOR_CACHE.get(key)
        if cached is None:
            cached = torch.full((H,), float(hyper), device=device)
            _HYPER_TENSOR_CACHE[key] = cached
        return cached
    if isinstance(hyper, torch.Tensor):
        if hyper.dim() == 0:
            key = (float(hyper.item()), H, *_device_key(device))
            cached = _HYPER_TENSOR_CACHE.get(key)
            if cached is None:
                cached = torch.full((H,), hyper.item(), device=device)
                _HYPER_TENSOR_CACHE[key] = cached
            return cached
        assert hyper.dim() == 1 and hyper.numel() == H, (
            f"Hyperparameter tensor must have {H} elements, got shape "
            f"{tuple(hyper.shape)}"
        )
        return hyper.to(device)
    raise ValueError(
        f"Hyperparameter must be a float, int, or 0-D/1-D tensor; got {type(hyper)}"
    )


@triton.jit
def triton_bmm_pool_sim_simmean(
    x_ptr,
    pool_ptr,
    sim_ptr,
    simthreshd1_ptr,
    N: tl.constexpr,
    D: tl.constexpr,
    BS: tl.constexpr
):
    b, h, nb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    B, H, NB = tl.num_programs(0), tl.num_programs(1), tl.num_programs(2)

    block_offset = b * H * N * D + h * N * D + nb * BS * D
    xmask = (nb*BS + tl.arange(0, BS)[:, None]) < N
    x_ptrs = x_ptr + block_offset + tl.arange(0, BS)[:, None] * D + tl.arange(0, D)[None, :]
    # Load the input block, xmask will return nan for out-of-bound elements
    x = tl.load(x_ptrs, mask = xmask)
    BS_ = BS if (N - nb*BS) >= BS else (N - nb*BS)

    cur_h1 = tl.load(simthreshd1_ptr + h)
    x_fp32 = x.to(tl.float32)
    # Check for NaN values
    is_nan = x_fp32 != x_fp32
    x_fp32 = tl.where(is_nan, 0.0, x_fp32)

    pool = (tl.sum(x_fp32, axis=0) / BS_)
    x_norm = tl.sqrt(tl.sum(x_fp32 * x_fp32, axis=1, keep_dims=True))
    x = (x / x_norm).to(tl.float16)  # norm at D dim
    # Check for NaN values after normalization
    is_nan = x != x
    x = tl.where(is_nan, 0.0, x)

    grams = tl.dot(x, tl.trans(x))
    sum_value = tl.sum(grams).to(tl.float32)
    cur_sim = (sum_value / (BS_ * BS_)) > cur_h1

    pool_block_offset = b * H * NB * D + h * NB * D + nb * D
    tl.store(pool_ptr + pool_block_offset + tl.arange(0, D), pool)
    sim_offset = b * H * NB + h * NB + nb
    tl.store(sim_ptr + sim_offset, cur_sim)


def get_pool_sim_triton_simmean(
    x, block_size, simthreshd1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
    x: (B, H, N, D)
    block_size: int
    simthreshd1: (H,) tensor

    Steps:
    1. Pooling within each block
    2. Compute similarity within each block
    3. Return pooled tensor and similarity mask

    Note how 3rd dimension N is reduced to nblock = N // block_size.
    This way later in the algorithm we don't compute the full attention ( O(N^2) ), but only O(nblock^2).

    Returns:
    pool: (B, H, nblock, D) tensor
    sim_blocks: (B, H, nblock) bool tensor
    """
    x = x.contiguous()
    B, H, N, D = x.shape
    nblock = (N + block_size - 1) // block_size  # Number of blocks per feature map
    pool = torch.empty((B, H, nblock, D), device=x.device, dtype=x.dtype)
    sim_blocks = torch.empty((B, H, nblock), device=x.device, dtype=torch.bool)
    grid = (B, H, nblock)
    # Launch kernel
    triton_bmm_pool_sim_simmean[grid](x, pool, sim_blocks, simthreshd1, N=N, D=D, BS=block_size)
    return pool, sim_blocks


@triton.jit
def triton_fill_causal_mask(mask, BqdivBk):
    q, k = tl.program_id(0), tl.program_id(1)
    Q, K = tl.num_programs(0), tl.num_programs(1)
    if k >= (q + 1) * BqdivBk:
        tl.store(mask + q * K + k, 0)
    else:
        tl.store(mask + q * K + k, 1)


def fill_causal_mask_triton(mask: torch.Tensor, BqdivBk:float) -> torch.Tensor:
    assert mask.dim() == 2
    triton_fill_causal_mask[mask.shape](mask, BqdivBk)
    return mask


@triton.jit
def triton_fill_block_map_kernel(final_map, num_to_select, sorted_indices, NK: tl.constexpr):
    b, h, q = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    B, H, Q = tl.num_programs(0), tl.num_programs(1), tl.num_programs(2)
    cur_num_to_select = tl.load(num_to_select + b * H * Q + h * Q + q)
    cur_sorted_idx_ptr = sorted_indices + b * H * Q * NK + h * Q * NK + q * NK
    cur_final_map_ptr = final_map + b * H * Q * NK + h * Q * NK + q * NK
    cur_num_to_select = (cur_num_to_select + 1) if cur_num_to_select == 0 else cur_num_to_select
    for i in range(cur_num_to_select):
        cur_idx = tl.load(cur_sorted_idx_ptr + i)
        tl.store(cur_final_map_ptr + cur_idx, 1)


def fill_block_map_triton(final_map, num_to_select, sorted_indices):
    final_map = final_map.contiguous()
    num_to_select = num_to_select.contiguous()
    sorted_indices = sorted_indices.contiguous()
    B, H, Q, K = final_map.shape
    grid = (B, H, Q)
    triton_fill_block_map_kernel[grid](final_map, num_to_select, sorted_indices, K)
    return final_map


def get_block_map_meansim(
    q: torch.Tensor,
    k: torch.Tensor,
    is_causal: bool = False,
    BLKQ: int = 64,
    BLKK: int = 64,
    simthreshd1: float = 0.1,
    cdfthreshd: float = 0.9,
    attention_sink: bool = False,
    return_ordering_aux: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the per-(B, H, q_block) block-sparse map via mean-pooled
    similarity.

    When ``return_ordering_aux`` is True, three extra tensors are returned
    alongside the boolean map so the caller can build an *ordered*
    (descending-probability, optionally with a static prefix) ragged LUT:

      * ``pooled_score`` ``(B, H, Q, K)`` float: the per-block-pair softmax
        probability used as the descending-priority key,
      * ``nonsim_mask`` ``(B, H, Q, K)`` bool: the structurally forced
        ("non-similar") block pairs, which are never frozen and belong to the
        static prefix of the ordered LUT, and
      * ``sorted_kidx`` ``(B, H, Q, K)`` int: the descending-``pooled_score``
        permutation of KV-block indices (reused from the CDF sort, so the LUT
        builder does not need a second argsort).
    """
    Headnum = q.size(1)
    simthreshd1 = hyperparameter_check(simthreshd1, Headnum, q.device)
    cdfthreshd = hyperparameter_check(cdfthreshd, Headnum, q.device)
    nq = (q.shape[-2] + BLKQ - 1) // BLKQ
    nk = (k.shape[-2] + BLKK - 1) // BLKK
    pooled_qblocks, sim_qblocks = get_pool_sim_triton_simmean(q, BLKQ, simthreshd1)
    pooled_kblocks, sim_kblocks = get_pool_sim_triton_simmean(k, BLKK, simthreshd1)

    sim_kblocks = sim_kblocks.unsqueeze(-2).expand(-1, -1, nq, -1)  # faster than repeat
    sim_qblocks = sim_qblocks.unsqueeze(-1).expand(-1, -1, -1, nk)
    pooled_score = pooled_qblocks @ pooled_kblocks.transpose(-1, -2) * q.shape[-1] ** -0.5

    neg_inf = pooled_score.new_full((), float("-inf"))
    pooled_score = torch.where(sim_kblocks, pooled_score, neg_inf)
    if is_causal:
        nq = pooled_qblocks.shape[-2]
        nk = pooled_kblocks.shape[-2]
        empty_mask = torch.empty(nq, nk, device=q.device, dtype=torch.bool)
        causal_mask = fill_causal_mask_triton(empty_mask, BLKQ / BLKK)
        pooled_score = torch.where(causal_mask[None, None, ...], pooled_score, neg_inf)
    pooled_score = pooled_score.softmax(-1)
    sorted_score = torch.sort(pooled_score, dim=-1, descending=True)
    cdf = torch.cumsum(sorted_score.values, dim=-1)
    B, H, Q, K = cdf.shape

    cdfthreshd_ts = cdfthreshd.view(1, H, 1, 1)

    ge = cdf >= cdfthreshd_ts

    idx = ge.to(torch.uint8).argmax(dim=-1)
    any_ge = ge.any(dim=-1)
    # 0-D fallback value broadcasts in `where`; avoids a (B, H, Q) alloc.
    num_to_select = torch.where(any_ge, idx, idx.new_full((), K))
 
    # Structurally forced ("non-similar") blocks: a block pair is forced into
    # the map when either its K block or its Q block failed the intra-block
    # similarity test. These are never frozen and form the static prefix.
    nonsim_mask = (~sim_kblocks) | (~sim_qblocks)

    final_map = torch.zeros_like(pooled_score, dtype=torch.bool)
    final_map = fill_block_map_triton(final_map, num_to_select, sorted_score.indices)
    final_map = final_map | nonsim_mask
    if is_causal:
        final_map = final_map * causal_mask[None, None, ...]
        nonsim_mask = nonsim_mask & causal_mask[None, None, ...]

    if attention_sink:
        final_map[:, :, :, 0] = 1

    if not return_ordering_aux:
        return final_map

    return final_map, pooled_score, nonsim_mask, sorted_score.indices


@triton.jit
def triton_fill_compact_lut_kernel(
    order_ptr, mask_ptr, lut_start_ptr, kv_out_ptr,
    NK: tl.constexpr, BLOCK_KB: tl.constexpr,
):
    """Stream-compact the selected blocks of a presorted candidate order.

    ``order_ptr[row]`` is a length-``NK`` permutation of KV-block indices in the
    desired (descending-priority) order. For each candidate we look up
    ``mask_ptr[row, candidate]`` and, when selected, append it to
    ``kv_out[lut_start[row] : ...]`` preserving the candidate order. This is the
    vectorized analogue of aiter's ``block_attn_mask_to_lut_kernel`` but it
    iterates in priority order rather than ascending KV index.
    """
    row = tl.program_id(0)
    base = tl.load(lut_start_ptr + row)
    order_row = order_ptr + row * NK
    mask_row = mask_ptr + row * NK
    write_offset = 0
    for start in range(0, NK, BLOCK_KB):
        offs = start + tl.arange(0, BLOCK_KB)
        in_bounds = offs < NK
        cand = tl.load(order_row + offs, mask=in_bounds, other=0)
        sel = tl.load(mask_row + cand, mask=in_bounds, other=0)
        sel_i = tl.where(in_bounds & (sel != 0), 1, 0).to(tl.int32)
        cumsum = tl.cumsum(sel_i, axis=0)
        tl.store(kv_out_ptr + base + write_offset + cumsum - 1, cand, mask=sel_i != 0)
        write_offset += tl.sum(sel_i)


def fill_compact_lut_triton(order, mask, lut_start, kv_out, BLOCK_KB: int = 128):
    order = order.contiguous()
    mask = mask.contiguous()
    lut_start = lut_start.contiguous()
    kv_out = kv_out.contiguous()
    B, H, Q, K = order.shape
    grid = (B * H * Q,)
    triton_fill_compact_lut_kernel[grid](
        order, mask, lut_start, kv_out, NK=K, BLOCK_KB=BLOCK_KB,
    )
    return kv_out


@triton.jit
def triton_fill_partitioned_lut_kernel(
    mask_ptr, live_ptr, lut_start_ptr, lut_freeze_ptr, kv_out_ptr,
    NK: tl.constexpr, BLOCK_KB: tl.constexpr,
):
    """Emit, per row, the live blocks then the frozen blocks, each in ascending
    KV-block index order. Both the mask and live reads are contiguous (the
    candidate index is the position itself), so the resulting LUT yields two
    monotonic, coalesced KV runs while still placing the priority-selected live
    blocks in the leading ``lut_freeze`` slots.
    """
    row = tl.program_id(0)
    base = tl.load(lut_start_ptr + row)
    n_live = tl.load(lut_freeze_ptr + row)
    mask_row = mask_ptr + row * NK
    live_row = live_ptr + row * NK

    # Region 1: live (selected & live) blocks, ascending index.
    write_offset = 0
    for start in range(0, NK, BLOCK_KB):
        offs = start + tl.arange(0, BLOCK_KB)
        in_bounds = offs < NK
        m = tl.load(mask_row + offs, mask=in_bounds, other=0)
        lv = tl.load(live_row + offs, mask=in_bounds, other=0)
        sel = tl.where(in_bounds & (m != 0) & (lv != 0), 1, 0).to(tl.int32)
        cumsum = tl.cumsum(sel, axis=0)
        tl.store(kv_out_ptr + base + write_offset + cumsum - 1, offs.to(tl.int32), mask=sel != 0)
        write_offset += tl.sum(sel)

    # Region 2: frozen (selected & not live) blocks, ascending index, placed
    # right after the live region (which has exactly n_live entries).
    write_offset = n_live
    for start in range(0, NK, BLOCK_KB):
        offs = start + tl.arange(0, BLOCK_KB)
        in_bounds = offs < NK
        m = tl.load(mask_row + offs, mask=in_bounds, other=0)
        lv = tl.load(live_row + offs, mask=in_bounds, other=0)
        sel = tl.where(in_bounds & (m != 0) & (lv == 0), 1, 0).to(tl.int32)
        cumsum = tl.cumsum(sel, axis=0)
        tl.store(kv_out_ptr + base + write_offset + cumsum - 1, offs.to(tl.int32), mask=sel != 0)
        write_offset += tl.sum(sel)


def fill_partitioned_lut_triton(mask, live, lut_start, lut_freeze, kv_out, BLOCK_KB: int = 128):
    mask = mask.contiguous()
    live = live.contiguous()
    lut_start = lut_start.contiguous()
    lut_freeze = lut_freeze.contiguous()
    kv_out = kv_out.contiguous()
    B, H, Q, K = mask.shape
    grid = (B * H * Q,)
    triton_fill_partitioned_lut_kernel[grid](
        mask, live, lut_start, lut_freeze, kv_out, NK=K, BLOCK_KB=BLOCK_KB,
    )
    return kv_out


def build_ordered_block_lut(
    full_mask: torch.Tensor,
    *,
    full_order: Optional[torch.Tensor] = None,
    static_mask: Optional[torch.Tensor] = None,
    priority: Optional[torch.Tensor] = None,
    freeze_ratio: float = 0.0,
    freeze_include_static: bool = True,
    static_first: bool = False,
    lut_freeze_ones: bool = True,
    coalesced: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build an ordered ragged block-sparse LUT plus the per-row freeze count.

    Ordering of the KV blocks for every ``(b, h, q)`` row:

      * ``static_first=False`` (default): all selected blocks sorted purely by
        **descending** ``priority``, so position 0 is the peak-probability
        block. The descending order is supplied via ``full_order`` (reused from
        ``get_block_map_meansim``'s sort) and the selected blocks are
        stream-compacted out of it on the GPU -- no second ``argsort``.
      * ``static_first=True``: the *static* non-frozen blocks
        (``static_mask & full_mask``) come first (ascending index), then the
        remaining selected blocks in descending ``priority`` order. This branch
        builds the order with an ``argsort`` (requires ``static_mask`` and
        ``priority``).

    Args:
        full_mask:   ``(B, H, Q, K)`` bool, all selected block pairs.
        full_order:  ``(B, H, Q, K)`` int, per-row permutation of KV-block
            indices in descending-priority order (required when
            ``static_first`` is False).
        static_mask: ``(B, H, Q, K)`` bool prefix mask (required when
            ``static_first`` is True, or when ``lut_freeze_ones`` is False and
            ``freeze_include_static``).
        priority:    ``(B, H, Q, K)`` float order key (required when
            ``static_first`` is True or ``lut_freeze_ones`` is False).
        freeze_ratio / freeze_include_static: control the computed freeze count
            when ``lut_freeze_ones`` is False.
        lut_freeze_ones: if True, ``lut_freeze`` is all ones (clamped to
            ``lut_count``): only the first ordered (peak) block stays live.

    Returns ``(kv_block_indices, lut_start, lut_count, lut_freeze)`` as int32,
    matching ``block_attn_mask_to_ragged_lut``'s ``(b, h, q)`` row layout.
    """
    B, H, Q, K = full_mask.shape
    device = full_mask.device
    selected = full_mask

    if coalesced and not static_first:
        # Coalesced freeze: pick the live (un-frozen) set by priority, but emit
        # the LUT as [live ascending] ++ [frozen ascending] so both KV runs are
        # monotonic. Online softmax is order-invariant within each region, so
        # only the live *set* (the leading lut_freeze slots) must be the top
        # blocks -- their internal order is free.
        if priority is None:
            raise ValueError("coalesced freeze requires `priority` to pick the live set")
        # Only carve static blocks out of the dynamic pool when we actually fold
        # them back into the live set; otherwise treat all selected blocks as
        # dynamic so the peak is taken over the full selected set.
        static_sel = (
            (static_mask & selected)
            if (static_mask is not None and freeze_include_static)
            else None
        )
        if lut_freeze_ones:
            # Single live block: the peak-priority selected block per row.
            neg = priority.new_full((), -1e30)
            masked_prio = torch.where(selected, priority, neg)
            argmax_idx = masked_prio.argmax(dim=-1, keepdim=True)
            live_mask = torch.zeros_like(selected)
            live_mask.scatter_(-1, argmax_idx, True)
            live_mask &= selected
        else:
            dynamic = selected & ~static_sel if static_sel is not None else selected
            dyn_prob = torch.where(dynamic, priority, priority.new_zeros(()))
            max_prob = dyn_prob.amax(dim=-1, keepdim=True)
            thresh = (1.0 - freeze_ratio) * max_prob
            live_mask = dynamic & (priority >= thresh)
            live_mask &= dynamic.any(dim=-1, keepdim=True)
            if freeze_include_static and static_sel is not None:
                live_mask = live_mask | static_sel

        lut_count = selected.to(torch.int32).sum(dim=-1).reshape(-1).to(torch.int32)
        lut_start = (torch.cumsum(lut_count, dim=0) - lut_count).to(torch.int32)
        lut_freeze = live_mask.to(torch.int32).sum(dim=-1).reshape(-1).to(torch.int32)

        kv_block_indices = torch.empty(B * H * Q * K, dtype=torch.int32, device=device)
        fill_partitioned_lut_triton(
            selected, live_mask, lut_start, lut_freeze, kv_block_indices,
        )
        return kv_block_indices, lut_start, lut_count, lut_freeze

    if static_first:
        # Static blocks sort first (score in (2, 3], lower index first), dynamic
        # blocks by their probability in (0, 1], unselected blocks last.
        static_sel = static_mask & selected
        idx = torch.arange(K, device=device, dtype=priority.dtype)
        static_score = 2.0 + (K - idx) / K
        order_score = torch.where(static_sel, static_score, priority)
        order_score = torch.where(selected, order_score, order_score.new_full((), -1e30))
        order = order_score.argsort(dim=-1, descending=True).to(torch.int32)
    else:
        # Reuse the descending-priority permutation produced upstream.
        order = full_order.to(torch.int32)

    # NB: torch.sum / torch.cumsum promote integer dtypes to int64; the fp8
    # sparse kernel asserts the whole LUT is int32, so cast both back.
    lut_count = selected.to(torch.int32).sum(dim=-1).reshape(-1).to(torch.int32)
    lut_start = (torch.cumsum(lut_count, dim=0) - lut_count).to(torch.int32)

    # Over-allocate like block_attn_mask_to_ragged_lut (avoids a data-dependent
    # .sum() that would graph-break torch.compile); the kernel only reads the
    # [lut_start, lut_start + lut_count) segments. The compaction kernel walks
    # `order` and emits the selected blocks in that order.
    kv_block_indices = torch.empty(B * H * Q * K, dtype=torch.int32, device=device)
    fill_compact_lut_triton(order, selected, lut_start, kv_block_indices)

    if lut_freeze_ones:
        # One leading (peak-probability) block live, m frozen for the rest.
        # Clamp to lut_count so a (degenerate) empty row stays 0.
        ones = torch.ones_like(lut_count)
        lut_freeze = torch.minimum(ones, lut_count).reshape(-1).to(torch.int32)
        return kv_block_indices, lut_start, lut_count, lut_freeze

    # Freeze count: static prefix + leading high-probability dynamic blocks
    # within `freeze_ratio` relative drop of that row's peak dynamic prob.
    static_sel = static_mask & selected
    dynamic = selected & ~static_sel
    dyn_prob = torch.where(dynamic, priority, priority.new_zeros(()))
    max_prob = dyn_prob.amax(dim=-1, keepdim=True)
    thresh = (1.0 - freeze_ratio) * max_prob
    n_ratio = (dynamic & (priority >= thresh)).sum(dim=-1)
    n_ratio = torch.where(dynamic.any(dim=-1), n_ratio, torch.zeros_like(n_ratio))

    if freeze_include_static:
        n_static = static_sel.to(torch.int32).sum(dim=-1)
    else:
        n_static = torch.zeros_like(n_ratio)
    lut_freeze = (n_static + n_ratio).reshape(-1).to(torch.int32)

    return kv_block_indices, lut_start, lut_count, lut_freeze
