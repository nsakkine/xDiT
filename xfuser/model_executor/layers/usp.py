# This file implements USP with torch version >= '2.5.0'
import torch
import functools
from torch.nn import functional as F

import torch.distributed._functional_collectives as ft_c

from torch.distributed.tensor.experimental._attention import _templated_ring_attention
import xfuser.envs as envs

if torch.cuda.is_available() or envs._is_npu():
    from yunchang.globals import PROCESS_GROUP
else:
    PROCESS_GROUP = None

from xfuser.core.distributed import (
    get_sequence_parallel_world_size,
    get_ulysses_parallel_world_size,
    get_ring_parallel_world_size,
    get_sequence_parallel_rank,
    get_ulysses_parallel_rank,
    get_runtime_state,
)

from packaging.version import parse
from xfuser.core.cache_manager.cache_manager import get_cache_manager
from xfuser.core.distributed.attention_backend import ATTENTION_FUNCTION_REGISTRY
from xfuser.core.sparge_attention.block_mask import (
    hyperparameter_check,
    get_pool_sim_triton_simmean,
    fill_causal_mask_triton,
    fill_block_map_triton,
)
try:
    from aiter.ops.triton.attention.fav3_sage import get_sage_fwd_configs
except ImportError:
    get_sage_fwd_configs = None  # required only when using sparge with Ulysses


def ring_attn(attention_function, query, key, value, dropout_p=0.0, is_causal=False, attn_func_kwargs=None, joint_attn_kwargs=None):
    kwargs = {
        "dropout_p": dropout_p,
        "is_causal": is_causal,
        "attn_func_kwargs": attn_func_kwargs,
        "joint_attn_kwargs": joint_attn_kwargs,
    }
    if parse(torch.__version__).release >= parse("2.6.0").release:
        from torch.distributed.tensor.experimental._attention import _cp_options
        _cp_options.enable_load_balance = False
        out, *_ = _templated_ring_attention(
            PROCESS_GROUP.RING_PG,
            1,
            attention_function,
            query,
            key,
            value,
            **kwargs,
        )
    else:
        out, *_ = _templated_ring_attention(
            PROCESS_GROUP.RING_PG,
            attention_function,
            query,
            key,
            value,
            **kwargs,
        )
    return out


def _maybe_wait(tensor: torch.Tensor) -> torch.Tensor:
    """
    When tracing the code, the result tensor is not an AsyncCollectiveTensor,
    so we cannot call ``wait()``.
    """
    if isinstance(tensor, ft_c.AsyncCollectiveTensor):
        return tensor.wait()
    return tensor


def _sdpa_all_to_all_single(x):
    x_shape = x.shape
    x = x.flatten()
    x = ft_c.all_to_all_single(x, output_split_sizes=None, input_split_sizes=None, group=PROCESS_GROUP.ULYSSES_PG)
    x = _maybe_wait(x)
    x = x.reshape(x_shape)
    return x


def _ft_c_input_all_to_all(x):
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    ndim = x.ndim
    if ndim == 3:
        x = x.unsqueeze(-1)

    assert x.ndim == 4, "x must have 4 dimensions, got {}".format(x.ndim)
    b, h, s, d = x.shape
    assert h % world_size == 0, "h must be divisible by world_size, got {} and {}".format(h, world_size)

    x = x.permute(1, 0, 2, 3).contiguous()
    x = _sdpa_all_to_all_single(x)
    x = x.reshape(world_size, h // world_size, b, -1, d).permute(2, 1, 0, 3, 4).reshape(b, h // world_size, -1, d)

    if ndim == 3:
        x = x.squeeze(-1)

    return x


def _ft_c_input_all_to_all_with_plan(x, head_partition_plan):
    """All-to-all over heads according to head_partition_plan (rank -> list of global head indices)."""
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    assert x.ndim == 4, "x must have 4 dimensions, got {}".format(x.ndim)
    b, h, s, d = x.shape
    rank = get_ulysses_parallel_rank()
    Hp = h

    send_counts = [
        len(head_partition_plan[j])
        for j in range(world_size)
    ]
    recv_counts = [
        len(head_partition_plan[rank])
        for j in range(world_size)
    ]

    reorder = []
    for j in range(world_size):
        for g in head_partition_plan[j]:
            reorder.append(g)

    reorder_t = torch.tensor(reorder, device=x.device, dtype=torch.long)
    x = x[:, reorder_t, :, :]

    x = x.permute(1, 0, 2, 3).contiguous()
    element_size = b * s * d
    input_split_sizes = [send_counts[j] * element_size for j in range(world_size)]
    output_split_sizes = [recv_counts[j] * element_size for j in range(world_size)]

    x_flat = x.reshape(-1)
    x_out = ft_c.all_to_all_single(
        x_flat,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=PROCESS_GROUP.ULYSSES_PG,
    )
    x_out = _maybe_wait(x_out)

    num_heads_recv = len(head_partition_plan[rank])
    x_out = x_out.reshape(num_heads_recv, b, -1, d).permute(1, 0, 2, 3).contiguous()

    return x_out


def _combined_qkv_all_to_all(q, k, v):
    """Concatenate query, key, value tensors and perform a single all-to-all communication."""
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return q, k, v

    assert q.ndim == 4, f"q must have 4 dimensions, got {q.ndim}"
    b, h, s, d = q.shape
    assert h % world_size == 0, f"h must be divisible by world_size, got {h} and {world_size}"

    # [3, b, h, s, d]
    qkv = torch.stack([q, k, v], dim=0)
    # [3, b, P, h/P, s, d]
    qkv = qkv.view(3, b, world_size, h // world_size, s, d)
    # [P, 3, b, h/P, s, d]
    qkv = qkv.permute(2, 0, 1, 3, 4, 5).contiguous()

    qkv = _sdpa_all_to_all_single(qkv)

    # [3, b, h/P, P, s, d]
    qkv = qkv.permute(1, 2, 3, 0, 4, 5).contiguous()
    # [3, b, h/P, P*s, d]
    qkv = qkv.view(3, b, h // world_size, -1, d)

    q, k, v = torch.unbind(qkv, dim=0)
    return q, k, v


def _combined_qkv_all_to_all_with_plan(q, k, v, head_partition_plan):
    """All-to-all over heads for Q, K, V according to head_partition_plan (three separate collectives)."""
    query = _ft_c_input_all_to_all_with_plan(q, head_partition_plan)
    key = _ft_c_input_all_to_all_with_plan(k, head_partition_plan)
    value = _ft_c_input_all_to_all_with_plan(v, head_partition_plan)
    return query, key, value


def _ft_c_output_all_to_all(x):
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    assert x.ndim == 4, "x must have 4 dimensions, got {}".format(x.ndim)
    b, h, s, d = x.shape
    assert s % world_size == 0, "s must be divisible by world_size, got {} and {}".format(s, world_size)

    x = x.permute(2, 0, 1, 3).contiguous()
    x = _sdpa_all_to_all_single(x)
    x = x.reshape(world_size, s // world_size, b, -1, d).permute(2, 0, 3, 1, 4).reshape(b, -1, s // world_size, d)
    return x


def _ft_c_output_all_to_all_with_plan(x, head_partition_plan):
    """
    Output all-to-all over the sequence dimension when head counts differ per rank.
    Each rank sends the j-th sequence chunk to rank j and receives the j-th chunk from every rank.
    Output is (b, H, s // world_size, d) with heads in global order 0..H-1.
    """
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    assert x.ndim == 4, "x must have 4 dimensions, got {}".format(x.ndim)
    b, h_local, s, d = x.shape
    rank = get_ulysses_parallel_rank()
    assert s % world_size == 0, "s must be divisible by world_size, got {} and {}".format(s, world_size)

    P = world_size
    sp = s // P
    H = sum(len(head_partition_plan[j]) for j in range(world_size))

    element_size = b * d
    input_split_sizes = [sp * h_local * element_size for _ in range(world_size)]
    output_split_sizes = [sp * len(head_partition_plan[j]) * element_size for j in range(world_size)]

    x = x.permute(1, 0, 2, 3).contiguous()
    x_flat = x.reshape(-1)
    x_out = ft_c.all_to_all_single(
        x_flat,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=PROCESS_GROUP.ULYSSES_PG,
    )
    x_out = _maybe_wait(x_out)

    x_out = x_out.reshape(H, b, sp, d)

    inv_perm = torch.empty(H, dtype=torch.long, device=x_out.device)
    recv_idx = 0
    for j in range(world_size):
        for g in head_partition_plan[j]:
            inv_perm[g] = recv_idx
            recv_idx += 1

    x_out = x_out[inv_perm, ...].permute(1, 0, 2, 3).contiguous()

    return x_out


def _preprocess_joint_tensors(joint_key, joint_value):
    """
    Preprocess the joint key and value tensors for Ulysses parallelism.
    """
    ulysses_world_size = get_ulysses_parallel_world_size()
    ulysses_rank = get_ulysses_parallel_rank()
    attn_heads_per_ulysses_rank = (
        joint_key.shape[1] // ulysses_world_size
    )
    joint_key = joint_key.transpose(1,2)
    joint_value = joint_value.transpose(1,2)
    joint_key = joint_key[
        ...,
        attn_heads_per_ulysses_rank
        * ulysses_rank : attn_heads_per_ulysses_rank
        * (ulysses_rank + 1),
        :, ].transpose(1,2)
    joint_value = joint_value[
        ...,
        attn_heads_per_ulysses_rank
        * ulysses_rank : attn_heads_per_ulysses_rank
        * (ulysses_rank + 1),
        :,
    ].transpose(1,2)
    return joint_key, joint_value

def _concat_joint_tensor(tensor, joint_tensor, joint_strategy, dim):
    """
    Concatenate the joint tensor to the main tensor based on the joint strategy.
    """
    if joint_strategy == "rear":
        tensor = torch.cat([tensor, joint_tensor], dim=dim)
    elif joint_strategy == "front":
        tensor = torch.cat([joint_tensor, tensor], dim=dim)
    else:
        raise ValueError(f"Invalid joint_strategy: {joint_strategy}")
    return tensor

def _update_and_get_kv_cache(key, value, attn_layer):
    """
    Update and get the key and value cache for pipeline parallelism.
    """
    key, value = get_cache_manager().update_and_get_kv_cache(
        new_kv=[key.transpose(1, 2), value.transpose(1, 2)],
        layer=attn_layer,
        slice_dim=1,
        layer_type="attn",
    )
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()
    return key, value

def _get_attention_function(backend=None):
    """
    Get the attention function based on the runtime state or from a given explicit backend.
    """
    if backend is not None:
        attention_backend = backend
    else:
        attention_backend = get_runtime_state().attention_backend
    func = ATTENTION_FUNCTION_REGISTRY.get(attention_backend, None)
    if func is None:
        raise NotImplementedError(f"Attention backend {attention_backend} not registered.")
    return concat_joint_tensors_decorator(func)

def concat_joint_tensors_decorator(func):
    """
    Decorator to handle joint tensor concatenation
    This is needed for ring attention with 'rear' joint_strategy, as it
    needs to concat the joint tensors before calling the attention function
    but only on the last step.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        query, key, value = args[0:3]
        is_causal = kwargs.get("is_causal")
        dropout_p = kwargs.get("dropout_p")
        joint_attn_kwargs = kwargs.get("joint_attn_kwargs", None)
        attn_func_kwargs = kwargs.get("attn_func_kwargs", {})

        if joint_attn_kwargs is not None:
            joint_strategy = joint_attn_kwargs.get("joint_strategy", None)
            joint_key = joint_attn_kwargs.get("joint_key", None)
            joint_value = joint_attn_kwargs.get("joint_value", None)
            step = joint_attn_kwargs.get("step", 0)
            total_steps = joint_attn_kwargs.get("total_steps", 1)
            if (joint_strategy == "front" and step == 0) or (joint_strategy == "rear" and step == total_steps - 1):
                key = _concat_joint_tensor(key, joint_key, joint_strategy, dim=2)
                value = _concat_joint_tensor(value, joint_value, joint_strategy, dim=2)
            joint_attn_kwargs["step"] = step + 1 # In place increment step

        return func(query, key, value, dropout_p=dropout_p, is_causal=is_causal, **attn_func_kwargs)

    return wrapper

def USP(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        joint_query: torch.Tensor | None = None,
        joint_key: torch.Tensor | None = None,
        joint_value: torch.Tensor | None = None,
        joint_strategy: str | None = None,
        attn_layer=None,
        combine_qkv_a2a: bool | None = None,
        backend=None,
        **attn_func_kwargs,
    ):
    """
    Unified Sequence Parallelism (USP) attention call, supporting combinations of Ulysses and
    Ring attention. Also supports joint tensors and key-value caching for pipeline parallelism.
    Explicit backend can be provided to specify the attention backend to use.
    """
    if combine_qkv_a2a is None:
        combine_qkv_a2a = False

    attention_function = _get_attention_function(backend=backend)


    joint_attn_kwargs = None
    if joint_strategy:
        query = _concat_joint_tensor(query, joint_query, joint_strategy, dim=2)
        joint_key, joint_value = _preprocess_joint_tensors(joint_key, joint_value)
        joint_attn_kwargs = {
            "joint_value": joint_value,
            "joint_key": joint_key,
            "joint_strategy": joint_strategy,
            "step": 0,
            "total_steps": get_ring_parallel_world_size(),

        }

    if get_ulysses_parallel_world_size() > 1:
        head_partition_plan = None
        if ("simthreshold" in attn_func_kwargs) and ("cdfthreshold" in attn_func_kwargs):
            Headnum = query.size(1)
            simthreshd1 = hyperparameter_check(attn_func_kwargs.get("simthreshold"), Headnum, query.device)

            config = get_sage_fwd_configs()
            block_m, block_n = config["BLOCK_M"], config["BLOCK_N"]
            pooled_query, query_sim_blocks = get_pool_sim_triton_simmean(query, block_m, simthreshd1)
            pooled_key, key_sim_blocks = get_pool_sim_triton_simmean(key, block_n, simthreshd1)

            pooled_query = _ft_c_input_all_to_all(pooled_query)
            pooled_key = _ft_c_input_all_to_all(pooled_key)
            query_sim_blocks = _ft_c_input_all_to_all(query_sim_blocks)
            key_sim_blocks = _ft_c_input_all_to_all(key_sim_blocks)

            nq = pooled_query.shape[-2]
            nk = pooled_key.shape[-2]
            sim_kblocks = key_sim_blocks.unsqueeze(-2).expand(-1, -1, nq, -1)  # faster than repeat
            sim_qblocks = query_sim_blocks.unsqueeze(-1).expand(-1, -1, -1, nk)
            pooled_score = pooled_query @ pooled_key.transpose(-1, -2) * pooled_query.shape[-1] ** -0.5
            # Mask out the blocks that are not similar
            pooled_score[~sim_kblocks] = -torch.inf
            if is_causal:
                nq = pooled_query.shape[-2]
                nk = pooled_key.shape[-2]
                empty_mask = torch.empty(nq, nk, device=pooled_query.device, dtype=torch.bool)
                causal_mask = fill_causal_mask_triton(empty_mask, block_m / block_n)
                pooled_score = pooled_score.masked_fill(~causal_mask[None, None, ...], -torch.inf)
            pooled_score = pooled_score.softmax(-1)
            sorted_score = torch.sort(pooled_score, dim=-1, descending=True)
            cdf = torch.cumsum(sorted_score.values, dim=-1)
            B, H, Q, K = cdf.shape
            cdfthreshd = hyperparameter_check(attn_func_kwargs.get("cdfthreshold"), H, query.device)
            cdfthreshd_ts = cdfthreshd.view(1, H, 1, 1)
            cdfthreshd_ts = cdfthreshd_ts.expand(B, -1, Q, 1).contiguous()
            # searchsorted(cdf, v, right=True) equivalent using compilable ops (avoids torch.compile graph break)
            ge = (cdf >= cdfthreshd_ts)
            idx = ge.long().argmax(dim=-1)
            num_to_select = torch.where(ge.any(dim=-1), idx, cdf.new_full(idx.shape, K, dtype=torch.long))
            final_map = torch.zeros_like(pooled_score, dtype=torch.bool)
            final_map[~sim_kblocks] = 1
            final_map[~sim_qblocks] = 1
            final_map = fill_block_map_triton(final_map, num_to_select, sorted_score.indices)
            if is_causal:
                final_map = final_map * causal_mask[None, None, ...]

            n_dense_blocks = final_map.sum(dim=-1).sum(dim=-1).sum(dim=0).contiguous()
            output = torch.empty(
                get_ulysses_parallel_world_size() * n_dense_blocks.numel(),
                dtype=n_dense_blocks.dtype,
                device=n_dense_blocks.device,
            )
            torch.distributed.all_gather_into_tensor(output, n_dense_blocks, group=PROCESS_GROUP.ULYSSES_PG)
            n_dense_blocks = output

            head_partition_plan = {j: [] for j in range(get_ulysses_parallel_world_size())}
            n_dense_blocks_per_gpu = torch.zeros(get_ulysses_parallel_world_size(), device=query.device)
            _, dense_blocks_order = torch.sort(n_dense_blocks, descending=True, stable=True)
            for head_idx in dense_blocks_order:
                gpu_rank = torch.argmin(n_dense_blocks_per_gpu).item()
                head_partition_plan[gpu_rank] = head_partition_plan[gpu_rank] + [head_idx.item()]
                n_dense_blocks_per_gpu[gpu_rank] += n_dense_blocks[head_idx]

        if head_partition_plan is not None and any(head_partition_plan):
            if combine_qkv_a2a and query.shape == key.shape == value.shape:
                query, key, value = _combined_qkv_all_to_all_with_plan(query, key, value, head_partition_plan)
            else:
                query = _ft_c_input_all_to_all_with_plan(query, head_partition_plan)
                key = _ft_c_input_all_to_all_with_plan(key, head_partition_plan)
                value = _ft_c_input_all_to_all_with_plan(value, head_partition_plan)
        else:
            if combine_qkv_a2a and query.shape == key.shape == value.shape:
                query, key, value = _combined_qkv_all_to_all(query, key, value)
            else:
                query = _ft_c_input_all_to_all(query)
                key = _ft_c_input_all_to_all(key)
                value = _ft_c_input_all_to_all(value)

    if attn_layer:
        key, value = _update_and_get_kv_cache(key, value, attn_layer)

    if get_sequence_parallel_world_size() == 1: # No SP
        out, _ = attention_function(
            query, key, value, dropout_p=dropout_p, is_causal=is_causal, attn_func_kwargs=attn_func_kwargs,
            joint_attn_kwargs=joint_attn_kwargs
        )

    elif get_ulysses_parallel_world_size() == 1: # Ring only
        out = ring_attn(
            attention_function,
            query, key, value, dropout_p=dropout_p, is_causal=is_causal, attn_func_kwargs=attn_func_kwargs,
            joint_attn_kwargs=joint_attn_kwargs
        )

    else:
        if get_ring_parallel_world_size() == 1: # Ulysses only
            out, _ = attention_function(
                query, key, value, dropout_p=dropout_p, is_causal=is_causal, attn_func_kwargs=attn_func_kwargs,
                joint_attn_kwargs=joint_attn_kwargs)
        else: # USP
            out = ring_attn(attention_function, query, key, value, dropout_p=dropout_p, is_causal=is_causal, joint_attn_kwargs=joint_attn_kwargs)
        if head_partition_plan is not None and any(head_partition_plan):
            out = _ft_c_output_all_to_all_with_plan(out, head_partition_plan)
        else:
            out = _ft_c_output_all_to_all(out)

    return out


def attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        backend=None,
        **attn_func_kwargs,
    ):
    """
    Runs attention call without any parallelism.
    This can be used when the logic necessitates no Ulysses or Ring parallelism in any case.
    Explicit backend can be provided to specify the attention backend to use.
    """
    attention_function = _get_attention_function(backend=backend)
    out, _ = attention_function(
        query,
        key,
        value,
        dropout_p=dropout_p,
        is_causal=is_causal,
        attn_func_kwargs=attn_func_kwargs,
    )
    return out

