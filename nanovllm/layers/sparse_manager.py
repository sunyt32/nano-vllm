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

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=2)
        for num_warps in [1, 2, 4, 8]
    ],
    key=['head_dim', 'num_splits'],
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
        attn_score_ptr = attn_score + batch_idx * stride_ab + head_idx * stride_ah + block_block_idx * stride_ad
        tl.store(attn_score_ptr, score, mask=block_mask)

def block_attention(q, k_min, k_max, num_blocks, local_num_blocks, block_tables, block_attention_workspace=None):
    min_val = torch.finfo(torch.float32).min
    batch = q.shape[0]
    _, indices_per_page, n_kv_heads, head_dim = k_min.shape
    if block_attention_workspace is None or not block_attention_workspace.numel():
        max_num_blocks = num_blocks.max().to(torch.int64)
        attn_score = torch.full((batch, n_kv_heads, max_num_blocks), fill_value=min_val, device=q.device, dtype=torch.float32)
    else:
        attn_score = block_attention_workspace
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
    i_h = tl.program_id(0)
    i_m = tl.program_id(1)

    i_n, i_t = tl.load(chunk_indices + i_m * 2).to(tl.int32), tl.load(chunk_indices + i_m * 2 + 1).to(tl.int32)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    seqlen = eos - bos
    k_idx = i_t * block_size + tl.arange(0, block_size)
    mask = k_idx < seqlen
    p_k = k + (bos + k_idx[:, None]) * D * H + i_h * D + tl.arange(0, D)
    s_k = tl.load(slot_mapping + bos + i_t * block_size) // block_size
    b_k = tl.load(p_k, mask=mask[:, None])
    b_k_min = tl.where(mask[:, None], b_k, 1e6)
    b_k_max = tl.where(mask[:, None], b_k, -1e6)
    tl.store(k_min + H * D * s_k + i_h * D + tl.arange(0, D), tl.min(b_k_min, axis=0))
    tl.store(k_max + H * D * s_k + i_h * D + tl.arange(0, D), tl.max(b_k_max, axis=0))

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
def get_topk_indices_kernel(
    block_attn_score,
    topk_indices,
    num_blocks,
    num_selected_blocks,
    N_POW2: tl.constexpr,
    stride_bb, stride_bh, stride_bl,
    stride_tb, stride_th, stride_tl,
):
    off_b = tl.program_id(0).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)

    N = tl.load(num_blocks + off_b)
    N_SEL = tl.load(num_selected_blocks + off_b)
    ids = tl.arange(0, N_POW2)
    sort_ids = tl.arange(0, N_POW2)
    attn_score_ptr = block_attn_score + off_b * stride_bb + off_h * stride_bh + ids * stride_bl
    x = tl.load(attn_score_ptr, mask=ids < N, other=-1e38)

    n_dims: tl.constexpr = int(math.log2(N_POW2))
    for i in tl.static_range(1, n_dims + 1):
        x, sort_ids = _bitonic_merge(x, sort_ids, i, 2 if i < n_dims else 1, n_dims)

    sort_ids = tl.where(ids < N_SEL, sort_ids, N_POW2)
    sort_ids = tl.sort(sort_ids, dim=0, descending=False)
    topk_indices_ptr = topk_indices + off_b * stride_tb + off_h * stride_th + ids * stride_tl
    tl.store(topk_indices_ptr, sort_ids, mask=ids < N_SEL)

def get_topk_indices(
    block_attn_score,
    num_blocks,
    num_selected_blocks,
    max_num_blocks,
    sparse_indices_workspace=None
):
    B, H = num_blocks.shape[0], block_attn_score.shape[1]
    if sparse_indices_workspace is None or not sparse_indices_workspace.numel():
        topk_indices = torch.full((B, H, num_selected_blocks.max()), -1, device=block_attn_score.device, dtype=torch.int32)
    else:
        topk_indices = sparse_indices_workspace
    N_POW2 = triton.next_power_of_2(max_num_blocks)
    grid = (B, H)
    with torch.cuda.device(block_attn_score.device.index): 
        get_topk_indices_kernel[grid](block_attn_score, topk_indices, num_blocks, num_selected_blocks, N_POW2, *block_attn_score.stride(), *topk_indices.stride())
    return topk_indices

@torch.compile(fullgraph=True)
def get_num_blocks(cache_seqlens, block_size, sparse_ratio, min_block_num):
    num_blocks = (cache_seqlens + (block_size - 1)) // block_size
    num_selected_blocks = torch.ceil(num_blocks * sparse_ratio)
    num_selected_blocks = torch.maximum(num_selected_blocks, num_blocks.clamp(max=min_block_num)).long()
    return num_blocks, num_selected_blocks

class KVManager:
    def __init__(self, num_heads, block_size, sparse_ratio, local_num_blocks, min_block_num):
        self.num_heads = num_heads
        self.block_size = block_size
        self.sparse_ratio = sparse_ratio
        self.local_num_blocks = local_num_blocks
        self.min_block_num = min_block_num
        self.max_num_blocks = None
        self.sparse_max_table = self.sparse_min_table = self.block_attention_workspace = self.sparse_indices_workspace = torch.tensor([])
        assert self.local_num_blocks <= self.min_block_num, "local_num_blocks should be less than or equal to min_block_num"

    def init_centeroids(self, key, cu_seqlens, slot_mapping):
        assert self.sparse_max_table.numel() > 0 and self.sparse_min_table.numel() > 0, "sparse_max_table and sparse_min_table should not be empty"
        block_init(key, self.sparse_max_table, self.sparse_min_table, cu_seqlens, slot_mapping, self.block_size)
        
    @torch.compile(fullgraph=True)
    def update_centeroids(self, key, slot_mapping):
        new_blocks = slot_mapping % self.block_size == 0
        block_ids = slot_mapping // self.block_size
        sparse_max_table = self.sparse_max_table.flatten(0, 1)
        sparse_min_table = self.sparse_min_table.flatten(0, 1)
        sparse_max_table[block_ids] = torch.where(new_blocks[:, None, None], key, torch.maximum(sparse_max_table[block_ids], key))
        sparse_min_table[block_ids] = torch.where(new_blocks[:, None, None], key, torch.minimum(sparse_min_table[block_ids], key))

    def get_kv_cache_indices(self, query, cache_seqlens, block_tables):
        bsz, num_heads, head_dim = query.shape
        num_blocks, num_selected_blocks = get_num_blocks(cache_seqlens, self.block_size, self.sparse_ratio, self.min_block_num)
        blocks_per_page = self.sparse_max_table.shape[1]
        query = query.view(bsz, self.num_heads, num_heads // self.num_heads, head_dim).mean(dim=2) * (head_dim ** -0.5)
        topk_indices = torch.full((bsz, self.num_heads, num_selected_blocks.max().item()), device=query.device, dtype=torch.int32, fill_value=-1)
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
        block_attn_score = block_attention(query, self.sparse_max_table, self.sparse_min_table, num_blocks, self.local_num_blocks, block_tables, self.block_attention_workspace)
        topk_indices = get_topk_indices(block_attn_score, num_blocks, num_selected_blocks, self.max_num_blocks, self.sparse_indices_workspace)
        return topk_indices, num_selected_blocks

def main():
    torch.cuda.manual_seed(0)
    bsz, n_head, key_dim = 4, 4, 256
    n_kv_seq = 8192
    page_size = 256
    pages = bsz * n_kv_seq // page_size
    gqa_size = 6
    block_size = 32
    dtype = torch.float16
    xq = torch.randn((bsz, n_head * gqa_size, key_dim), device='cuda', dtype=dtype)
    cache_seqlens = torch.randint(100, n_kv_seq, (bsz,), device='cuda', dtype=torch.int32)
    xk = torch.randn((cache_seqlens.sum().item(), n_head, key_dim), device='cuda', dtype=dtype)
    num_pages = torch.ceil(cache_seqlens / page_size).int()
    num_blocks = torch.ceil(cache_seqlens / block_size).int()
    cu_num_pages = torch.cat([torch.zeros(1, device='cuda', dtype=torch.int32), torch.cumsum(num_pages, dim=0)])

    kv_manager = KVManager(n_head, block_size, 0.2, 1, 16)
    kv_manager.max_num_blocks = n_kv_seq // block_size
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
    for _ in range(10):
        topk_indices_fast, _ = kv_manager.get_kv_cache_indices_fast(xq, cache_seqlens, block_tables)
    start_time = time.time()
    for _ in range(100):
        topk_indices_fast, _ = kv_manager.get_kv_cache_indices_fast(xq, cache_seqlens, block_tables)
    end_time = time.time()
    print(f"get_kv_cache_indices_fast time: {end_time - start_time}")
    for _ in range(10):
        topk_indices, _ = kv_manager.get_kv_cache_indices(xq, cache_seqlens, block_tables)
    start_time = time.time()
    for _ in range(100):
        topk_indices, _ = kv_manager.get_kv_cache_indices(xq, cache_seqlens, block_tables)
    end_time = time.time()
    print(f"get_kv_cache_indices time: {end_time - start_time}")
    print((topk_indices != topk_indices_fast).sum(), topk_indices.numel())
        
if __name__ == "__main__":
    main()