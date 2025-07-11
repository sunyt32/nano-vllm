import torch
import math
import triton
import triton.language as tl

torch._dynamo.config.capture_scalar_outputs = True

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
        if efficiency[num_splits - 1] >= 0.95 * max_efficiency:
            return num_splits

    return 1

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=2)
        for num_warps in [1, 2, 4, 8]
    ],
    key=['head_dim', 'num_kv_heads', 'num_splits'],
)
@triton.jit
def block_attention_kernel(
    q,
    k_min,
    k_max,
    num_blocks,
    block_tables,
    local_num_blocks: tl.constexpr,
    num_splits: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    indices_per_page: tl.constexpr,
    stride_qb, stride_qh, stride_qd,
    stride_ab, stride_ah, stride_ad,
    max_num_pages,
    attn_score,
    BLOCK_N: tl.constexpr,
    BLOCK_PAGE: tl.constexpr
):
    batch_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)
    split_idx = tl.program_id(2).to(tl.int64)

    query_ptr = q + batch_idx * stride_qb + head_idx * stride_qh + tl.arange(0, head_dim) * stride_qd
    query = tl.load(query_ptr).to(tl.float32)
    POS_INF = 1e38

    total_n_blocks = tl.load(num_blocks + batch_idx)
    total_n_split_blocks = (total_n_blocks + BLOCK_N - 1) // BLOCK_N
    blocks_per_split = total_n_split_blocks // num_splits
    remaining_blocks = total_n_split_blocks % num_splits
    loop_range = blocks_per_split + (1 if split_idx < remaining_blocks else 0)
    start = blocks_per_split * split_idx + min(split_idx, remaining_blocks)
    end = start + loop_range
    for i in range(start, end):
        start_page_idx = i * BLOCK_PAGE
        page_idx = start_page_idx + tl.arange(0, BLOCK_PAGE)
        page_ptr = block_tables + batch_idx * max_num_pages + page_idx
        page_block_idx = tl.load(page_ptr, mask=page_idx < max_num_pages)
        key_block_idx = page_block_idx[:, None] * indices_per_page + tl.arange(0, indices_per_page)
        key_block_idx = tl.reshape(key_block_idx, (BLOCK_N))

        block_block_idx = start_page_idx * indices_per_page + tl.arange(0, BLOCK_N)
        block_mask = block_block_idx < total_n_blocks
        k_min_ptr = k_min + key_block_idx[:, None] * num_kv_heads * head_dim + head_idx * head_dim + tl.arange(0, head_dim)
        k_max_ptr = k_max + key_block_idx[:, None] * num_kv_heads * head_dim + head_idx * head_dim + tl.arange(0, head_dim)
        k_min_val = tl.load(k_min_ptr, mask=block_mask[:, None]).to(tl.float32)
        k_max_val = tl.load(k_max_ptr, mask=block_mask[:, None]).to(tl.float32)
        score = tl.maximum(k_min_val * query, k_max_val * query)
        score = tl.sum(score, axis=1)
        score = tl.where(block_block_idx >= total_n_blocks - local_num_blocks, POS_INF, score)
        score_u32 = tl.cast(score, tl.uint32, bitcast=True)
        sign_bit = score_u32 >> 31
        flip_mask = sign_bit * 0xFFFFFFFF
        scores_ord = score_u32 ^ (flip_mask | 0x80000000)
        attn_score_ptr = attn_score + batch_idx * stride_ab + head_idx * stride_ah + block_block_idx * stride_ad
        tl.store(attn_score_ptr, scores_ord, mask=block_mask)

def block_attention(q, k_min, k_max, num_blocks, local_num_blocks, block_tables, max_num_blocks):
    batch = q.shape[0]
    _, indices_per_page, n_kv_heads, head_dim = k_min.shape
    attn_score = torch.zeros((batch, n_kv_heads, max_num_blocks), device=q.device, dtype=torch.uint32)
    BLOCK_N = 32
    BLOCK_PAGE = BLOCK_N // indices_per_page
    num_splits = num_splits_heuristic(batch * n_kv_heads, max_splits=8)
    assert BLOCK_N % indices_per_page == 0, "BLOCK_N should be divisible by indices_per_page"
    grid = (batch, n_kv_heads, num_splits)
    with torch.cuda.device(q.device.index): 
        block_attention_kernel[grid](q, k_min, k_max, 
            num_blocks, block_tables, local_num_blocks, num_splits,
            n_kv_heads, head_dim, indices_per_page,
            *q.stride(),
            *attn_score.stride(),
            block_tables.shape[1],
            attn_score,
            BLOCK_N=BLOCK_N,
            BLOCK_PAGE=BLOCK_PAGE
        )
    return attn_score

def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int
) -> torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(cu_seqlens[1:] - cu_seqlens[:-1], chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).contiguous().to(cu_seqlens)

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=1)
        for num_warps in [1, 2, 4, 8]
    ],
    key=['H', 'D', 'block_size'],
)
@triton.jit
def block_init_kernel(
    k,
    k_max,
    k_min,
    cu_seqlens,
    chunk_indices,
    slot_mapping,
    block_size: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
):
    i_h = tl.program_id(0).to(tl.int64)
    i_m = tl.program_id(1).to(tl.int64)

    i_n, i_t = tl.load(chunk_indices + i_m * 2).to(tl.int64), tl.load(chunk_indices + i_m * 2 + 1).to(tl.int64)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    seqlen = eos - bos
    k_idx = i_t * block_size + tl.arange(0, block_size)
    mask = k_idx < seqlen
    p_k = k + (bos + k_idx[:, None]) * D * H + i_h * D + tl.arange(0, D)
    s_k = tl.load(slot_mapping + bos + i_t * block_size) // block_size
    b_k = tl.load(p_k, mask=mask[:, None])
    b_k_min = tl.where(mask[:, None], b_k, 1e6)
    b_k_max = tl.where(mask[:, None], b_k, -1e6)
    b_k_min = tl.min(b_k_min, axis=0)
    b_k_max = tl.max(b_k_max, axis=0)
    tl.store(k_min + H * D * s_k + i_h * D + tl.arange(0, D), b_k_min)
    tl.store(k_max + H * D * s_k + i_h * D + tl.arange(0, D), b_k_max)

def block_init(k, k_max, k_min, cu_seqlens, slot_mapping, block_size):
    L, H, D = k.shape
    assert k.is_contiguous() and k_max.is_contiguous() and k_min.is_contiguous(), "k, k_min and k_max should be contiguous"
    chunk_indices = prepare_chunk_indices(cu_seqlens, block_size)
    grid = (H, chunk_indices.shape[0])
    with torch.cuda.device(k.device.index): 
        block_init_kernel[grid](k, k_max, k_min, 
            cu_seqlens, chunk_indices, slot_mapping,
            block_size, H, D
        )

@triton.jit
def _compare_and_swap(
    x,
    ids,
    flip,
    i: tl.constexpr,
    n_dims: tl.constexpr,
):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2**i, 2, 2**(n_dims - i - 1)]
    y = tl.reshape(x, shape)
    # slice left/right with 'stride' 2**(n_dims - i - 1)
    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape).to(y.dtype)
    right = tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape).to(y.dtype)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)
    # idx
    y_idx = tl.reshape(ids, shape)
    left_idx = tl.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = tl.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = tl.reshape(left_idx, x.shape).to(y_idx.dtype)
    right_idx = tl.reshape(right_idx, x.shape).to(y_idx.dtype)
    # actual compare-and-swap
    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    cond = (left > right) != flip
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    new_ids = ids ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(ids))
    return ret.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge(
    x,
    ids,
    stage: tl.constexpr,
    order: tl.constexpr,
    n_dims: tl.constexpr,
):
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    # flip denotes whether to re-arrange sub-sequences of elements in ascending or
    # descending order.
    # if flip = 00000000... then all elements will be re-arranged ascendingly at this stage
    # if flip = 00110011... then all the elements will be re-arranged alternatingly (with
    # a stride of 2) at this stage
    if order == 2:
        shape: tl.constexpr = [n_outer * 2**(n_dims - 1 - stage), 2, 2**stage]
        flip = tl.reshape(tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape)
    else:
        flip = order
    # perform `stage` rounds of `compare-and-swap`
    for i in tl.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids

@triton.jit
def sort_topk_indices_kernel(
    block_attn_score,       # [B, H, L]
    indices_out,            # [B, H, K], int32 output indices
    num_blocks,             # total number of blocks = N // BLOCK_SIZE
    num_select_blocks,      # total number of select blocks = K // BLOCK_SIZE
    N_SEL_POW2: tl.constexpr,
    stride_bb, stride_bh, stride_bl,
    stride_mbb, stride_mbh, stride_mbl,
):
    bid = tl.program_id(0).to(tl.int64)
    hid = tl.program_id(1).to(tl.int64)

    N = tl.load(num_blocks + bid)
    N_SEL = tl.load(num_select_blocks + bid)
    scores = tl.zeros([N_SEL_POW2 * 2], tl.uint32)
    ids = tl.full([N_SEL_POW2 * 2], N, tl.int32)
    space_mask = tl.arange(0, N_SEL_POW2 * 2) >= N_SEL_POW2
    for i in range(0, N, N_SEL_POW2):
        index = i + tl.arange(0, N_SEL_POW2 * 2) - N_SEL_POW2
        mask = (index < N) & space_mask
        local_scores = tl.load(block_attn_score + bid * stride_bb + hid * stride_bh + index * stride_bl, mask=mask)
        scores = tl.where(mask, local_scores, scores)
        ids = tl.where(mask, index, ids)
        n_dims: tl.constexpr = tl.standard._log2(N_SEL_POW2 * 2)
        for j in tl.static_range(1, n_dims + 1):
            scores, ids = _bitonic_merge(scores, ids, j, 2 if j < n_dims else 1, n_dims)

    final_mask = tl.arange(0, N_SEL_POW2 * 2) < N_SEL
    ids = tl.where(final_mask, ids, N)
    ids = tl.sort(ids, dim=0, descending=False)
    tl.store(indices_out + bid * stride_mbb + hid * stride_mbh + tl.arange(0, N_SEL_POW2 * 2) * stride_mbl, ids, mask=final_mask)

@triton.jit
def radix_topk_indices_kernel(
    block_attn_score,       # [B, H, L]
    indices_out,            # [B, H, K], int32 output indices
    candidate,              # [B, H, N], bool output topk mask
    current_candidate,      # [B, H, N], bool output current topk mask
    num_blocks,             # total number of blocks = N // BLOCK_SIZE
    num_select_blocks,      # total number of select blocks = K // BLOCK_SIZE
    BLOCK_SIZE: tl.constexpr,
    RADIX_BITS: tl.constexpr,
    stride_bb, stride_bh, stride_bl,
    stride_mbb, stride_mbh, stride_mbl,
    stride_sb, stride_sh, stride_sl,
):
    bid = tl.program_id(0).to(tl.int64)
    hid = tl.program_id(1).to(tl.int64)

    N = tl.load(num_blocks + bid)
    N_SEL = tl.load(num_select_blocks + bid)

    # === Step 1: 初始化候选 mask ===
    for i in range(0, N, BLOCK_SIZE):
        is_candidate = tl.full([BLOCK_SIZE], False, tl.int1)
        index = i + tl.arange(0, BLOCK_SIZE)
        is_current_candidate = index < N
        tl.store(candidate + bid * stride_sb + hid * stride_sh + index * stride_sl, is_candidate, mask=is_current_candidate)
        tl.store(current_candidate + bid * stride_sb + hid * stride_sh + index * stride_sl, is_current_candidate, mask=is_current_candidate)

    # === Step 2: 32轮 Bit 筛选 ===
    for bit in range(32 // RADIX_BITS):
        bit_count = tl.zeros([2 ** RADIX_BITS], tl.int32)
        bit_range = (2 ** RADIX_BITS - 1) - tl.arange(0, 2 ** RADIX_BITS)
        for i in range(0, N, BLOCK_SIZE):
            index = i + tl.arange(0, BLOCK_SIZE)
            scores = tl.load(block_attn_score + bid * stride_bb + hid * stride_bh + index * stride_bl, mask=index < N)
            is_current_candidate = tl.load(current_candidate + bid * stride_sb + hid * stride_sh + index * stride_sl, mask=index < N)
            scores = (scores >> (32 // RADIX_BITS - 1 - bit) * RADIX_BITS) & (2 ** RADIX_BITS - 1)
            bit_mask = bit_range[:, None] == scores[None, :]
            bit_mask &= is_current_candidate[None, :]
            bit_count += tl.sum(bit_mask.to(tl.int32), axis=1)
        
        sel_bit_count = tl.cumsum(bit_count) < N_SEL
        bit_rank = tl.sum(sel_bit_count.to(tl.int32))
        bit_count = tl.where(tl.arange(0, 2 ** RADIX_BITS) < bit_rank, bit_count, 0)
        N_SEL = N_SEL - tl.sum(bit_count)
        for i in range(0, N, BLOCK_SIZE):
            index = i + tl.arange(0, BLOCK_SIZE)
            mask = index < N
            scores = tl.load(block_attn_score + bid * stride_bb + hid * stride_bh + index * stride_bl, mask=mask)
            is_candidate = tl.load(candidate + bid * stride_sb + hid * stride_sh + index * stride_sl, mask=mask)
            is_current_candidate = tl.load(current_candidate + bid * stride_sb + hid * stride_sh + index * stride_sl, mask=mask)
            scores = (scores >> (32 // RADIX_BITS - 1 - bit) * RADIX_BITS) & (2 ** RADIX_BITS - 1)
            bit_mask = bit_range[:, None] == scores[None, :]
            bit_mask &= is_current_candidate[None, :]
            is_next_candidate = bit_mask & (tl.arange(0, 2 ** RADIX_BITS) < bit_rank)[:, None]
            is_next_candidate = tl.sum(is_next_candidate.to(tl.int32), axis=0)
            is_next_current_candidate = bit_mask & (tl.arange(0, 2 ** RADIX_BITS) == bit_rank)[:, None]
            is_next_current_candidate = tl.sum(is_next_current_candidate.to(tl.int32), axis=0)
            is_candidate |= is_next_candidate.to(tl.int1)
            is_current_candidate &= is_next_current_candidate.to(tl.int1)
            tl.store(candidate + bid * stride_sb + hid * stride_sh + index * stride_sl, is_candidate, mask=mask)
            tl.store(current_candidate + bid * stride_sb + hid * stride_sh + index * stride_sl, is_current_candidate, mask=mask)

    for i in range(0, N, BLOCK_SIZE):
        index = i + tl.arange(0, BLOCK_SIZE)
        mask = index < N
        is_candidate = tl.load(candidate + bid * stride_sb + hid * stride_sh + index * stride_sl, mask=mask)
        is_current_candidate = tl.load(current_candidate + bid * stride_sb + hid * stride_sh + index * stride_sl, mask=mask)
        is_current_candidate &= (tl.cumsum(is_current_candidate.to(tl.int32)) <= N_SEL)
        is_candidate |= is_current_candidate
        N_SEL = N_SEL - tl.sum(is_current_candidate.to(tl.int32))
        tl.store(candidate + bid * stride_sb + hid * stride_sh + index * stride_sl, is_candidate, mask=mask)

    current_ptr = 0
    for i in range(0, N, BLOCK_SIZE):
        index = i + tl.arange(0, BLOCK_SIZE)
        is_candidate = tl.load(candidate + bid * stride_sb + hid * stride_sh + index * stride_sl, mask=index < N)
        is_candidate_ptr = current_ptr + tl.cumsum(is_candidate.to(tl.int32)) - 1
        tl.store(indices_out + bid * stride_mbb + hid * stride_mbh + is_candidate_ptr * stride_mbl, index, mask=is_candidate)
        current_ptr += tl.sum(is_candidate.to(tl.int32))

def get_topk_indices(
    block_attn_score,
    num_blocks,
    num_selected_blocks,
    max_num_blocks,
    max_num_selected_blocks,
):
    B, H = num_blocks.shape[0], block_attn_score.shape[1]
    topk_indices = torch.full((B, H, max_num_selected_blocks), -1, device=block_attn_score.device, dtype=torch.int32)
    candidate = torch.zeros((B, H, max_num_blocks), device=block_attn_score.device, dtype=torch.bool)
    current_candidate = torch.zeros((B, H, max_num_blocks), device=block_attn_score.device, dtype=torch.bool)
    grid = (B, H)
    # with torch.cuda.device(block_attn_score.device.index): 
    #     sort_topk_indices_kernel[grid](block_attn_score, topk_indices, num_blocks, num_selected_blocks, triton.next_power_of_2(max_num_selected_blocks), *block_attn_score.stride(), *topk_indices.stride(), num_stages=3, num_warps=4)

    BLOCK_SIZE, RADIX_BITS = 2048, 4
    with torch.cuda.device(block_attn_score.device.index): 
        radix_topk_indices_kernel[grid](block_attn_score, topk_indices, candidate, current_candidate, num_blocks, num_selected_blocks, BLOCK_SIZE, RADIX_BITS, *block_attn_score.stride(), *topk_indices.stride(), *candidate.stride(), num_stages=3, num_warps=8)
    return topk_indices

@torch.compile(fullgraph=True)
def get_topk_indices_fast(
    block_attn_score,
    num_blocks,
    num_selected_blocks,
    max_num_blocks,
    max_num_selected_blocks,
):
    topk_indices = torch.topk(block_attn_score.to(torch.int64), k=max_num_selected_blocks, dim=-1, sorted=True).indices
    topk_indices = torch.masked_fill(topk_indices, torch.arange(max_num_selected_blocks, device=topk_indices.device) >= num_selected_blocks[:, None, None], max_num_blocks)
    topk_indices = torch.sort(topk_indices, dim=-1, descending=False).values
    return topk_indices

@torch.compile(fullgraph=True)
def get_num_blocks(cache_seqlens, block_size, sparse_ratio, min_block_num):
    num_blocks = (cache_seqlens + (block_size - 1)) // block_size
    num_selected_blocks = torch.ceil(num_blocks * sparse_ratio)
    num_selected_blocks = torch.maximum(num_selected_blocks, num_blocks.clamp(max=min_block_num)).long()
    return num_blocks, num_selected_blocks

@triton.jit
def update_centeroids_kernel(
    key,
    k_max,
    k_min,
    slot_mapping,
    block_size: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
):
    i_b = tl.program_id(0)
    i_h = tl.program_id(1)

    p_k = key + i_b * D * H + i_h * D + tl.arange(0, D)
    s_k = tl.load(slot_mapping + i_b)
    new_blocks = s_k % block_size == 0
    block_ids = s_k // block_size
    b_k = tl.load(p_k)
    p_k_max = k_max + block_ids * D * H + i_h * D + tl.arange(0, D)
    p_k_min = k_min + block_ids * D * H + i_h * D + tl.arange(0, D)
    if new_blocks:
        tl.store(p_k_max, b_k)
        tl.store(p_k_min, b_k)
    else:
        old_k_max = tl.load(p_k_max)
        old_k_min = tl.load(p_k_min)
        tl.store(p_k_max, tl.maximum(old_k_max, b_k))
        tl.store(p_k_min, tl.minimum(old_k_min, b_k))

class KVManager:
    def __init__(self, num_heads, block_size, sparse_ratio, local_num_blocks, min_block_num):
        self.num_heads = num_heads
        self.block_size = block_size
        self.sparse_ratio = sparse_ratio
        self.local_num_blocks = local_num_blocks
        self.min_block_num = min_block_num
        self.max_num_blocks = None
        self.max_num_selected_blocks = None
        self.sparse_max_table = self.sparse_min_table = torch.tensor([])
        assert self.local_num_blocks <= self.min_block_num, "local_num_blocks should be less than or equal to min_block_num"

    def init_centeroids(self, key, cu_seqlens, slot_mapping):
        assert self.sparse_max_table.numel() > 0 and self.sparse_min_table.numel() > 0, "sparse_max_table and sparse_min_table should not be empty"
        block_init(key, self.sparse_max_table, self.sparse_min_table, cu_seqlens, slot_mapping, self.block_size)
        
    def update_centeroids(self, key, slot_mapping):
        batch, num_kv_heads, head_dim = key.shape
        with torch.cuda.device(key.device.index): 
            update_centeroids_kernel[(batch, num_kv_heads)](key, self.sparse_max_table, self.sparse_min_table, slot_mapping, self.block_size, num_kv_heads, head_dim, num_stages=1, num_warps=4)

    def get_kv_cache_indices(self, query, cache_seqlens, block_tables):
        bsz, num_heads, head_dim = query.shape
        num_blocks, num_selected_blocks = get_num_blocks(cache_seqlens, self.block_size, self.sparse_ratio, self.min_block_num)
        blocks_per_page = self.sparse_max_table.shape[1]
        query = query.view(bsz, self.num_heads, num_heads // self.num_heads, head_dim).mean(dim=2) * (head_dim ** -0.5)
        topk_indices = torch.full((bsz, self.num_heads, self.max_num_selected_blocks), device=query.device, dtype=torch.int32, fill_value=-1)
        for i in range(bsz):
            local_query = query[i]
            local_block_index = torch.arange(num_blocks[i], device=query.device)
            table_index_0 = block_tables[i][local_block_index // blocks_per_page]
            table_index_1 = local_block_index % blocks_per_page
            local_sparse_max_table = self.sparse_max_table[table_index_0, table_index_1]
            local_sparse_min_table = self.sparse_min_table[table_index_0, table_index_1]
            local_attn_score = torch.maximum(local_query.float() * local_sparse_max_table.float(), local_query.float() * local_sparse_min_table.float()).sum(dim=-1)
            local_attn_score[-self.local_num_blocks:] = torch.finfo(torch.float32).max
            local_topk_indices = torch.topk(local_attn_score, k=num_selected_blocks[i], dim=0).indices.transpose(0, 1)
            topk_indices[i, :, :num_selected_blocks[i]] = torch.sort(local_topk_indices, dim=1, descending=False).values
        return topk_indices, num_selected_blocks
    
    def get_kv_cache_indices_fast(self, query, cache_seqlens, block_tables):
        bsz, num_heads, head_dim = query.shape
        num_blocks, num_selected_blocks = get_num_blocks(cache_seqlens, self.block_size, self.sparse_ratio, self.min_block_num)
        query = query.view(bsz, self.num_heads, num_heads // self.num_heads, head_dim).mean(dim=2) * (head_dim ** -0.5)
        block_attn_score = block_attention(query, self.sparse_max_table, self.sparse_min_table, num_blocks, self.local_num_blocks, block_tables, self.max_num_blocks)
        # topk_indices = get_topk_indices_fast(block_attn_score, num_blocks, num_selected_blocks, self.max_num_blocks, self.max_num_selected_blocks)
        topk_indices = get_topk_indices(block_attn_score, num_blocks, num_selected_blocks, self.max_num_blocks, self.max_num_selected_blocks)
        return topk_indices, num_selected_blocks

def main():
    torch.cuda.manual_seed(0)
    bsz, n_head, key_dim = 8, 4, 256
    n_kv_seq = 131072
    page_size = 256
    pages = bsz * n_kv_seq // page_size
    gqa_size = 6
    block_size = 16
    dtype = torch.bfloat16
    xq = torch.randn((bsz, n_head * gqa_size, key_dim), device='cuda', dtype=dtype)
    cache_seqlens = torch.randint(100, n_kv_seq, (bsz,), device='cuda', dtype=torch.int32)
    xk = torch.randn((cache_seqlens.sum().item(), n_head, key_dim), device='cuda', dtype=dtype)
    num_pages = torch.ceil(cache_seqlens / page_size).int()
    num_blocks = torch.ceil(cache_seqlens / block_size).int()
    cu_num_pages = torch.cat([torch.zeros(1, device='cuda', dtype=torch.int32), torch.cumsum(num_pages, dim=0)])

    kv_manager = KVManager(n_head, block_size, 0.1, 1, 16)
    kv_manager.max_num_blocks = n_kv_seq // block_size
    kv_manager.max_num_selected_blocks = math.ceil(kv_manager.max_num_blocks * kv_manager.sparse_ratio)
    kv_manager.sparse_max_table = torch.randn((pages, page_size // block_size, n_head, key_dim), device='cuda', dtype=dtype)
    kv_manager.sparse_min_table = torch.randn((pages, page_size // block_size, n_head, key_dim), device='cuda', dtype=dtype)
    naive_sparse_max_table = kv_manager.sparse_max_table.clone()
    naive_sparse_min_table = kv_manager.sparse_min_table.clone()
    # test init block
    cu_seqlens = torch.cat([torch.zeros(1, device='cuda', dtype=torch.int32), torch.cumsum(cache_seqlens, dim=0)])
    slot_mapping = []
    page_mapping = torch.randperm(num_pages.sum().item(), device='cuda', dtype=torch.int32)
    block_tables = torch.full((bsz, num_pages.max().item()), -1, device='cuda', dtype=torch.int32)
    for i in range(bsz):
        local_page_mapping = page_mapping[cu_num_pages[i]:cu_num_pages[i + 1]]
        local_slot_mapping = local_page_mapping[:, None] * page_size + torch.arange(page_size, device='cuda', dtype=torch.int32)
        block_tables[i, :num_pages[i]] = local_page_mapping
        slot_mapping.append(local_slot_mapping.flatten()[:cache_seqlens[i]])
    slot_mapping = torch.cat(slot_mapping, dim=0)
    print(cu_seqlens, num_blocks)

    # test init centeroids
    kv_manager.init_centeroids(xk, cu_seqlens, slot_mapping)
    for i in range(bsz):
        local_block_size = (cache_seqlens[i] + block_size - 1) // block_size
        for j in range(local_block_size):
            local_slot_mapping = slot_mapping[cu_seqlens[i] + j * block_size] // block_size
            begin_idx = cu_seqlens[i] + j * block_size
            end_idx = min(cu_seqlens[i + 1], begin_idx + block_size)
            local_xk = xk[begin_idx:end_idx]
            naive_sparse_max_table.flatten(0, 1)[local_slot_mapping, :] = local_xk.max(dim=0).values
            naive_sparse_min_table.flatten(0, 1)[local_slot_mapping, :] = local_xk.min(dim=0).values
    print((naive_sparse_max_table - kv_manager.sparse_max_table).abs().max(), (naive_sparse_max_table - kv_manager.sparse_max_table).abs().mean())
    print((naive_sparse_min_table - kv_manager.sparse_min_table).abs().max(), (naive_sparse_min_table - kv_manager.sparse_min_table).abs().mean())

    # test get kv cache indices
    import time
    for _ in range(3):
        topk_indices_fast, _ = kv_manager.get_kv_cache_indices_fast(xq, cache_seqlens, block_tables)
    for _ in range(3):
        topk_indices, _ = kv_manager.get_kv_cache_indices(xq, cache_seqlens, block_tables)
    print((topk_indices != topk_indices_fast).masked_fill(topk_indices == -1, False).sum(), topk_indices.numel())

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(100):
        topk_indices_fast, _ = kv_manager.get_kv_cache_indices_fast(xq, cache_seqlens, block_tables)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"get_kv_cache_indices_fast time: {end_time - start_time}")
    
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(100):
        topk_indices, _ = kv_manager.get_kv_cache_indices(xq, cache_seqlens, block_tables)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"get_kv_cache_indices time: {end_time - start_time}")
        
if __name__ == "__main__":
    main()