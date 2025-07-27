import pickle
import math
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        model_args = config.model_args
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(model_args, config)
        load_model(self.model, config.model, config.checkpoint)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and not self.rank
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        model_args = config.model_args
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = model_args.kv_head // self.world_size
        num_cross_kv_heads = model_args.cross_kv_head // self.world_size if model_args.yoco_cross_layers > 0 else 0
        kv_cache_layers = model_args.n_layers - model_args.yoco_cross_layers
        cross_kv_layer_times = 5 if model_args.yoco_window_size > 0 else 1
        block_bytes = 2 * (kv_cache_layers+cross_kv_layer_times) * self.block_size * num_kv_heads * model_args.head_dim * self.config.dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0, "There is no enough GPU memory to allocate KV cache"
        self.kv_cache = torch.zeros(2, kv_cache_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, model_args.head_dim)
        self.cross_kv_cache = torch.zeros(2, cross_kv_layer_times*config.num_kvcache_blocks, self.block_size, num_cross_kv_heads, model_args.cross_head_dim) if num_cross_kv_heads > 0 else None
        self.window_size = model_args.yoco_window_size
        print(f"Global kv cache shape: {self.kv_cache.shape}")
        layer_id = 0
        for module in self.model.model.layers[:kv_cache_layers].modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
        if self.cross_kv_cache is not None:
            self.model.model.shared_attention.k_cache = self.cross_kv_cache[0]
            self.model.model.shared_attention.v_cache = self.cross_kv_cache[1]
        if config.sparse_decoding:
            assert self.block_size % config.sparse_block_size == 0
            sparse_indices_per_block = self.block_size // config.sparse_block_size
            max_num_blocks = config.max_model_len // config.sparse_block_size
            max_num_selected_blocks = math.ceil(max_num_blocks * config.sparse_block_ratio)
            self.sparse_max_tables = torch.zeros(kv_cache_layers, config.num_kvcache_blocks, sparse_indices_per_block, num_kv_heads, model_args.head_dim)
            self.sparse_min_tables = torch.zeros(kv_cache_layers, config.num_kvcache_blocks, sparse_indices_per_block, num_kv_heads, model_args.head_dim)
            self.cross_sparse_max_tables = torch.zeros(config.num_kvcache_blocks, sparse_indices_per_block, num_cross_kv_heads, model_args.cross_head_dim) if num_cross_kv_heads > 0 else None
            self.cross_sparse_min_tables = torch.zeros(config.num_kvcache_blocks, sparse_indices_per_block, num_cross_kv_heads, model_args.cross_head_dim) if num_cross_kv_heads > 0 else None
            layer_id = 0
            for module in self.model.model.layers[kv_cache_layers:].modules():
                if hasattr(module, "kv_manager"):
                    module.kv_manager.sparse_max_table = self.sparse_max_tables[layer_id]
                    module.kv_manager.sparse_min_table = self.sparse_min_tables[layer_id]
                    module.kv_manager.max_num_blocks = max_num_blocks
                    module.kv_manager.max_num_selected_blocks = max_num_selected_blocks
                    layer_id += 1
                    if layer_id == kv_cache_layers:
                        break
            if self.cross_sparse_max_tables is not None:
                self.model.model.shared_attention.kv_manager.sparse_max_table = self.cross_sparse_max_tables
                self.model.model.shared_attention.kv_manager.sparse_min_table = self.cross_sparse_min_tables
                self.model.model.shared_attention.kv_manager.max_num_blocks = max_num_blocks
                self.model.model.shared_attention.kv_manager.max_num_selected_blocks = max_num_selected_blocks
        else:
            self.sparse_max_table = self.sparse_min_table = None
            self.cross_sparse_max_table = self.cross_sparse_min_table = None

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.sliding_block_table) for seq in seqs)
        sliding_block_tables = [seq.sliding_block_table + [-1] * (max_len - len(seq.sliding_block_table)) for seq in seqs]
        sliding_block_tables = torch.tensor(sliding_block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return sliding_block_tables, block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        sliding_slot_mapping = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:]) 
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            context_lens.append(seqlen)
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.sliding_block_table:
                continue
            sliding_start_idx = max(seq.num_cached_blocks, seq.num_blocks - seq.num_sliding_blocks)
            tmp_slot_mapping = []
            for i in range(sliding_start_idx, seq.num_blocks):
                start = seq.sliding_block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                tmp_slot_mapping.extend(list(range(start, end)))
            sliding_slot_mapping.extend(tmp_slot_mapping[-self.window_size:])
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        sliding_block_tables, block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        sliding_slot_mapping = torch.tensor(sliding_slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, context_lens, slot_mapping, block_tables, sliding_slot_mapping,  sliding_block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        sliding_slot_mapping = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq))
            context_lens.append(len(seq))
            sliding_slot_mapping.append(seq.sliding_block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        sliding_slot_mapping = torch.tensor(sliding_slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        sliding_block_tables, block_tables = self.prepare_block_tables(seqs)
        set_context(False, sliding_slot_mapping=sliding_slot_mapping, context_lens=context_lens, slot_mapping=slot_mapping, block_tables=block_tables, sliding_block_tables=sliding_block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["sliding_slot_mapping"][:bs] = context.sliding_slot_mapping
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["sliding_block_tables"][:bs, :context.sliding_block_tables.size(1)] = context.sliding_block_tables
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        model_args = config.model_args
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        sliding_slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        sliding_block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, model_args.d_model)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, context_lens=context_lens[:bs], slot_mapping=slot_mapping[:bs], block_tables=block_tables[:bs], sliding_slot_mapping=sliding_slot_mapping[:bs], sliding_block_tables=sliding_block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            sliding_slot_mapping=sliding_slot_mapping,
            context_lens=context_lens,
            sliding_block_tables=sliding_block_tables,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            outputs=outputs,
        )
