import torch
from torch import nn
import triton
import triton.language as tl
import torch.distributed as dist

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context
from nanovllm.layers.sparse_manager import KVManager
from nanovllm.layers.flash_sparse_decoding import flash_block_sparse_decoding

def get_slopes(n):
    import math
    def get_slopes_power_of_2(n):
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start*ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)                   #In the paper, we only train models that have 2^a heads for some a. This function has
    else:                                                 #some good properties that only occur when the input is a power of 2. To maintain that even
        closest_power_of_2 = 2**math.floor(math.log2(n))  #when the number of heads is not a power of 2, we use this workaround. 
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    
@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0).to(tl.int64)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx).to(tl.int64)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        layer_id,
        num_heads,
        head_dim,
        is_self_layer,
        window_size,
        alibi,
        scale,
        num_kv_heads,
        sparse_decoding,
        sparse_block_size,
        sparse_block_ratio,
        sparse_min_num_blocks,
        sparse_local_num_blocks,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.is_self_layer = is_self_layer
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.alibi_slopes = torch.tensor(get_slopes(self.num_heads), dtype=torch.float32).chunk(self.tp_size, dim=0)[self.tp_rank] if alibi else None
        self.kv_manager = KVManager(num_kv_heads, sparse_block_size, sparse_block_ratio, sparse_local_num_blocks, sparse_min_num_blocks) if sparse_decoding and self.window_size > 0 else None

    def forward(self, q: torch.Tensor, k: torch.Tensor | None, v: torch.Tensor | None):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        if self.is_self_layer:
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel() and self.is_self_layer:
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill and self.is_self_layer:
            if self.kv_manager and self.kv_manager.sparse_max_table.numel():
                self.kv_manager.init_centeroids(k, context.cu_seqlens_q, context.slot_mapping)
            block_table = context.block_tables if context.cu_seqlens_k[-1] > context.cu_seqlens_q[-1] else None
            if block_table is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=block_table, window_size=(self.window_size, 0) if self.window_size > 0 else (-1, -1), alibi_slopes=self.alibi_slopes)
        elif k_cache.numel() and v_cache.numel():    # decode
            if self.kv_manager:
                if self.is_self_layer:
                    self.kv_manager.update_centeroids(k, context.slot_mapping)
                sparse_indices, num_selected_blocks = self.kv_manager.get_kv_cache_indices_fast(q, context.context_lens, context.block_tables)
                o = flash_block_sparse_decoding(q, k_cache, v_cache,
                                                cache_seqlens=context.context_lens,
                                                block_indices=sparse_indices,
                                                num_selected_blocks=num_selected_blocks,
                                                block_tables=context.block_tables,
                                                sm_scale=self.scale,
                                                block_size=self.kv_manager.block_size)
            else:
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                            softmax_scale=self.scale, causal=True, window_size=(self.window_size, 0) if self.window_size > 0 else (-1, -1), alibi_slopes=self.alibi_slopes)
        else:
            o = q # only for yoco warmup
        o = o.view(-1, self.num_heads * self.head_dim)
        return o
    
    def fill_kvcache(self, k: torch.Tensor, v: torch.Tensor):
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if self.kv_manager and self.kv_manager.sparse_max_table.numel():
                self.kv_manager.init_centeroids(k, context.cu_seqlens_q, context.slot_mapping)
        else:    # decode
            if self.kv_manager:
                self.kv_manager.update_centeroids(k, context.slot_mapping)