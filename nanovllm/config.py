import os
import argparse
import torch
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    dtype: torch.dtype = torch.float16
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    sparse_decoding: bool = False
    sparse_block_size: int = 16
    sparse_block_ratio: float = 0.1
    sparse_min_num_blocks: int = 16
    sparse_local_num_blocks: int = 1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser(description='NanoVLLM Configuration')
        parser.add_argument('--model', type=str, required=True, help='Path to the model')
        parser.add_argument('--max_num_batched_tokens', type=int, default=16384, help='Maximum number of batched tokens')
        parser.add_argument('--max_num_seqs', type=int, default=512, help='Maximum number of sequences')
        parser.add_argument('--max_model_len', type=int, default=4096, help='Maximum model length')
        parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help='GPU memory utilization')
        parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Tensor parallel size')
        parser.add_argument('--enforce_eager', action='store_true', help='Enforce eager mode')
        parser.add_argument('--eos', type=int, default=-1, help='End of sequence token')
        parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'float32'], help='Data type')
        parser.add_argument('--kvcache_block_size', type=int, default=256, help='KV cache block size')
        parser.add_argument('--sparse_decoding', action='store_true', help='Enable sparse decoding')
                
        args = parser.parse_args()
        
        # 转换 dtype
        dtype_map = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16
        }
        
        return cls(
            model=args.model,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            enforce_eager=args.enforce_eager,
            eos=args.eos,
            dtype=dtype_map[args.dtype],
            kvcache_block_size=args.kvcache_block_size,
            sparse_decoding=args.sparse_decoding
        )
