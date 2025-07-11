import math
import torch

import triton
import triton.language as tl

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def num_splits_heuristic(total_mblocks, max_splits):
    props = torch.cuda.get_device_properties(torch.device("cuda:0"))
    num_sm = props.multi_processor_count
    if total_mblocks >= 0.8 * num_sm:
        return 1
    
    max_efficiency = 0.0
    efficiency = []

    # Compute efficiency for different splits
    for num_splits in range(1, max_splits + 1):
        n_waves = (total_mblocks * num_splits) / num_sm
        eff = n_waves / math.ceil(n_waves)
        # Track max efficiency
        if eff > max_efficiency:
            max_efficiency = eff

        efficiency.append(eff)

    # Find the smallest number of splits that achieves at least 85% of max efficiency
    for num_splits in range(1, max_splits + 1):
        if efficiency[num_splits - 1] >= 0.9 * max_efficiency:
            return num_splits

    return 1

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
    ],
    key=['gqa_group_size', 'BLOCK_H', 'BLOCK_N', 'BLOCK_D', 'BLOCK_V'],
)
@triton.jit
def _fwd_kernel_decoding(
    Q, K, V, Out, L,
    sm_scale,
    cache_seqlens,
    block_indices_ptr,
    num_selected_blocks,
    stride_qz, stride_qh, stride_qd,
    stride_kz, stride_vz,
    stride_oz, stride_oh, stride_os, stride_od,
    stride_lz, stride_lh, stride_ls,
    stride_bz, stride_bn, stride_bd,
    block_tables,
    max_num_pages: tl.constexpr,
    page_block_size: tl.constexpr,
    num_splits: tl.constexpr,
    gqa_group_size: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    BLOCK_H: tl.constexpr, 
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    off_z = tl.program_id(0).to(tl.int64)
    off_h_for_kv = tl.program_id(1).to(tl.int64)
    off_split = tl.program_id(2).to(tl.int64)

    off_h_q = off_h_for_kv * gqa_group_size
    
    offs_m = tl.arange(0, BLOCK_H) ## head 
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    offs_v = tl.arange(0, BLOCK_V)

    seqlen_k = tl.load(cache_seqlens + off_z)
    num_selected_blocks = tl.load(num_selected_blocks + off_z * NUM_KV_HEADS + off_h_for_kv)
    blocks_per_split = num_selected_blocks // num_splits
    remaining_blocks = num_selected_blocks % num_splits
    loop_range = blocks_per_split + (1 if off_split < remaining_blocks else 0)
    start = blocks_per_split * off_split + min(off_split, remaining_blocks)

    Q += off_z * stride_qz + off_h_q * stride_qh
    K += off_h_for_kv * BLOCK_D
    V += off_h_for_kv * BLOCK_V
    L += off_z * stride_lz + off_h_q * stride_lh + off_split * stride_ls
    Out += off_z * stride_oz + off_h_q * stride_oh + off_split * stride_os
    block_indices_ptr += off_z * stride_bz + off_h_for_kv * stride_bn

    q = tl.load(Q + offs_m[:, None] * stride_qh + offs_d[None, :] * stride_qd,
                mask=(offs_m[:, None] < gqa_group_size)) ## padding to min 16

    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_H], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_V], dtype=tl.float32)

    k_ptrs = K + offs_d[:, None]
    v_ptrs = V + offs_v[None, :]
    if block_tables is not None:
        block_tables += off_z * max_num_pages
    else:
        k_ptrs += off_z * stride_kz
        v_ptrs += off_z * stride_vz

    for block_ptr_idx in range(start, start + loop_range):
        block_idx = tl.load(block_indices_ptr + block_ptr_idx * stride_bd).to(tl.int64)
        if block_idx >= 0 and block_idx * BLOCK_N < seqlen_k:
            start_n = block_idx * BLOCK_N
            mask = offs_n + start_n < seqlen_k
            if block_tables is not None:
                page_idx, page_offset = start_n // page_block_size, start_n % page_block_size
                page_ptrs = tl.load(block_tables + page_idx).to(tl.int64) * page_block_size
                k = tl.load(k_ptrs + (page_ptrs + page_offset + offs_n[None, :]) * NUM_KV_HEADS * BLOCK_D, mask=mask[None, :])
                v = tl.load(v_ptrs + (page_ptrs + page_offset + offs_n[:, None]) * NUM_KV_HEADS * BLOCK_V, mask=mask[:, None])
            else:
                k = tl.load(k_ptrs + (start_n + offs_n[None, :]) * NUM_KV_HEADS * BLOCK_D, mask=mask[None, :])
                v = tl.load(v_ptrs + (start_n + offs_n[:, None]) * NUM_KV_HEADS * BLOCK_V, mask=mask[:, None])
    
            qk = tl.dot(q, k)
            qk = tl.where(mask, qk, -1e6)
            qk *= sm_scale

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
            p = tl.exp(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            
            p = p.to(v.type.element_ty)

            acc += tl.dot(p, v)
            m_i = m_ij

    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip
    if num_selected_blocks > 0:
        m_i += tl.math.log(l_i)
    else:
        m_i = tl.zeros([BLOCK_H], dtype=tl.float32)

    l_ptrs = L + offs_m * stride_lh
    tl.store(l_ptrs, m_i, mask=(offs_m < gqa_group_size))
    O_ptrs = Out + offs_m[:, None] * stride_oh + offs_v[None, :] * stride_od
    tl.store(O_ptrs, acc, mask=(offs_m[:, None] < gqa_group_size))

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
    ],
    key=['BLOCK_V'],
)
@triton.jit
def combine(
    out_partial, out, L,
    stride_op_z, stride_op_h, stride_op_s, stride_op_d,
    stride_o_z, stride_o_h, stride_o_d,
    stride_l_z, stride_l_h, stride_l_s,
    num_splits: tl.constexpr,
    num_splits_pow2: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    off_z = tl.program_id(0).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)

    split = tl.arange(0, num_splits_pow2)
    split_mask = split < num_splits

    lse_local = tl.load(L + off_z * stride_l_z + off_h * stride_l_h + split * stride_l_s, mask=split_mask, other=float("-inf"))
    lse_max_local = tl.max(lse_local, axis=0)

    lse_logsum_local = tl.sum(tl.exp(lse_local - lse_max_local), axis=0)
    lse_logsum_local = tl.log(lse_logsum_local) + lse_max_local
    
    po_local = tl.load(out_partial + off_z * stride_op_z + off_h * stride_op_h + split[:, None] * stride_op_s + tl.arange(0, BLOCK_V) * stride_op_d, mask=split_mask[:, None])
    scale_local = tl.exp(lse_local - lse_logsum_local)
    accum_local = tl.sum(po_local * scale_local[:, None], axis=0)
    tl.store(out + off_z * stride_o_z + off_h * stride_o_h + tl.arange(0, BLOCK_V) * stride_o_d, accum_local.to(out.type.element_ty))

def flash_block_sparse_decoding(
    q, k, v,
    cache_seqlens,
    block_indices,
    num_selected_blocks,
    block_tables=None,
    sm_scale=None,
    block_size=64,
    num_splits=None
):
    # split q to blocks
    batch, n_heads, key_dim = q.shape
    _, _, n_kv_heads, head_dim = v.shape
    gqa_group_size = n_heads // n_kv_heads
    block_h = max(triton.next_power_of_2(gqa_group_size), 16)
    if len(num_selected_blocks.shape) == 1:
        num_selected_blocks = num_selected_blocks.unsqueeze(1).repeat(1, n_kv_heads)

    assert k.size(0) == v.size(0)
    assert q.size(2) == k.size(3)
    assert k.size(1) == v.size(1)
    assert key_dim in {64, 128, 256}
    assert head_dim in {64, 128, 256}
    assert k.is_contiguous() and v.is_contiguous(), "k cache and v cache must be contiguous"
    assert triton.next_power_of_2(block_size) == block_size, "block size must be power of 2"
    assert num_selected_blocks.shape == (batch, n_kv_heads), "num_selected_blocks must be a 2D tensor of shape (batch, n_kv_heads)"

    total_mblocks = batch * n_kv_heads
    if num_splits is None:
        num_splits = num_splits_heuristic(
            total_mblocks, max_splits=32)

    out_partial = torch.empty((batch, n_heads, num_splits, head_dim), device=q.device, dtype=torch.float32)
    out = torch.empty((batch, n_heads, head_dim), device=q.device, dtype=q.dtype)
    L = torch.empty((batch, n_heads, num_splits), device=q.device, dtype=torch.float32)

    if is_hip():
        extra_kern_args = {"waves_per_eu": 1}
    else:
        extra_kern_args = {}

    with torch.cuda.device(q.device.index): 
        grid = (batch, n_kv_heads, num_splits)
        _fwd_kernel_decoding[grid](
            q, k, v, out_partial, L,
            sm_scale if sm_scale is not None else key_dim ** -0.5,
            cache_seqlens.contiguous(),
            block_indices.contiguous(),
            num_selected_blocks.contiguous(),
            *q.stride(),
            k.stride(0) if block_tables is None else None,
            v.stride(0) if block_tables is None else None,
            *out_partial.stride(),
            *L.stride(),
            *block_indices.stride(),
            block_tables=block_tables.contiguous() if block_tables is not None else None,
            NUM_KV_HEADS=n_kv_heads,
            max_num_pages=block_tables.shape[1] if block_tables is not None else 0,
            page_block_size=k.shape[1] if block_tables is not None else 0,
            num_splits=num_splits,
            gqa_group_size=gqa_group_size,
            BLOCK_H = block_h,
            BLOCK_N = block_size,
            BLOCK_D = key_dim,
            BLOCK_V = head_dim,
            **extra_kern_args
        )
        grid = (batch, n_heads)
        combine[grid](
            out_partial, out, L,
            *out_partial.stride(),
            *out.stride(),
            *L.stride(),
            num_splits=num_splits,
            num_splits_pow2=triton.next_power_of_2(num_splits),
            BLOCK_V = head_dim,
            **extra_kern_args
        )
    return out

def main():
    from torch.nn import functional as F
    from einops import rearrange
    import time
    torch.cuda.manual_seed(0)
    bsz, n_head, key_dim = 7, 4, 256
    n_kv_seq = 8192
    page_size = 256
    pages = bsz * n_kv_seq // page_size
    head_dim = 128
    gqa_size = 6
    block_size = 32
    dtype = torch.float16
    xq = torch.randn((bsz, n_head * gqa_size, key_dim), device='cuda', dtype=dtype)
    xk = torch.randn((pages, page_size, n_head, key_dim), device='cuda', dtype=dtype)
    xv = torch.randn((pages, page_size, n_head, head_dim), device='cuda', dtype=dtype)
    cache_seqlens = torch.randint(100, n_kv_seq, (bsz,), device='cuda', dtype=torch.int32)
    num_pages = torch.ceil(cache_seqlens / page_size).int()
    num_blocks = torch.ceil(cache_seqlens / block_size).int()

    cu_num_blocks = torch.cat([torch.zeros(1, device='cuda', dtype=torch.int32), torch.cumsum(num_blocks, dim=0)])
    sparse_mask = torch.zeros((bsz, n_head, (n_kv_seq + block_size - 1) // block_size), device='cuda', dtype=torch.bool)
    sparse_mask_compact = torch.rand((n_head, num_blocks.sum()), device='cuda') < 0.2
    for i in range(bsz):
        sparse_mask[i, :, :num_blocks[i]] = sparse_mask_compact[:, cu_num_blocks[i]:cu_num_blocks[i + 1]]
    print(cache_seqlens, sparse_mask.shape, sparse_mask.sum(dim=-1))

    cu_num_pages = torch.cat([torch.zeros(1, device='cuda', dtype=torch.int32), torch.cumsum(num_pages, dim=0)])
    block_tables_compact = torch.randperm(pages, device='cuda', dtype=torch.int32)[:num_pages.sum().item()]
    block_tables = torch.full((bsz, num_pages.max().item()), -1, device='cuda', dtype=torch.int32)
    for i in range(bsz):
        block_tables[i, :num_pages[i]] = block_tables_compact[cu_num_pages[i]:cu_num_pages[i + 1]]

    num_selected_blocks = sparse_mask.sum(dim=-1).to(torch.int32)
    sparse_indices = torch.full((bsz * 2, n_head, (n_kv_seq + block_size - 1) // block_size), -1, device='cuda', dtype=torch.int32)
    for i in range(bsz):
        for j in range(n_head):
            valid_blocks = torch.where(sparse_mask[i, j])[0]
            sparse_indices[i, j, :len(valid_blocks)] = valid_blocks
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(100):
        triton_output = flash_block_sparse_decoding(xq, xk, xv, cache_seqlens, sparse_indices, num_selected_blocks=num_selected_blocks, block_tables=block_tables, block_size=block_size)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"Triton Time taken: {end_time - start_time} seconds")
    
    naive_mask = torch.zeros((bsz, n_head, 1, bsz * n_kv_seq), device=xq.device, dtype=torch.bool)
    for i in range(bsz):
        for j in range(n_head):
            for k in range(num_blocks[i]):
                if sparse_mask[i, j, k]:
                    page_idx = (k * block_size) // page_size
                    page_offset = (k * block_size) % page_size
                    begin_ptr = block_tables[i, page_idx] * page_size + page_offset
                    end_ptr = begin_ptr + block_size
                    causal_mask = torch.arange(k * block_size, (k + 1) * block_size, device=xq.device) < cache_seqlens[i]
                    naive_mask[i, j, 0, begin_ptr:end_ptr] = causal_mask
    
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(100):
        output = F.scaled_dot_product_attention(xq.unsqueeze(2), rearrange(xk, 'b n h d -> 1 h (b n) d'), rearrange(xv, 'b n h d -> 1 h (b n) d'), attn_mask=naive_mask.repeat_interleave(gqa_size, dim=1), enable_gqa=True)
        output = output.view(bsz, n_head * gqa_size, head_dim)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"Torch SDPA Time taken: {end_time - start_time} seconds")
    print((output - triton_output).abs().max(), (output - triton_output).abs().mean())
    
if __name__ == "__main__":
    main()