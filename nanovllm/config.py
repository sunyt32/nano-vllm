import os
import json
import argparse
import torch
from dataclasses import dataclass

@dataclass
class ModelArgs:
    d_model: int = 4096
    d_ffn: int = 14336
    head: int = 32
    kv_head: int = 8
    head_dim: int = None
    n_layers: int = 32
    vocab_size: int = 128256
    max_seq_len: int = 4096
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    qk_norm: bool = False
    attention_bias: bool = False
    weight_tying: bool = False
    alibi: bool = False
    # YOCO config
    yoco_cross_layers: int = 0
    yoco_window_size: int = 1024
    cross_kv_head: int = None
    cross_head_dim: int = None
    from_scratch: bool = False

@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    model_args: ModelArgs | None = None
    checkpoint: str | None = None
    eos: int = -1
    dtype: torch.dtype = torch.float16
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    num_cross_kvcache_blocks: int = -1
    sparse_decoding: bool = False
    sparse_block_size: int = 16
    sparse_block_ratio: float = 0.1
    sparse_min_num_blocks: int = 16
    sparse_local_num_blocks: int = 1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        if self.checkpoint is not None:
            with open(os.path.join(self.checkpoint, "metadata.json")) as f:
                metadata = json.load(f)
            modelargs = ModelArgs()
            for k, v in metadata["modelargs"].items():
                setattr(modelargs, k, v)
        else:
            config = json.load(open(os.path.join(self.model, "config.json")))
            params = {
                "d_model": config["hidden_size"],
                "d_ffn": config["intermediate_size"],
                "head": config["num_attention_heads"],
                "kv_head": config["num_key_value_heads"],
                "head_dim": config["head_dim"],
                "n_layers": config["num_hidden_layers"],
                "rope_theta": config["rope_theta"],
                "qk_norm": config["model_type"] == "qwen3",
                "attention_bias": config.get("attention_bias", True),
                "norm_eps": config["rms_norm_eps"],
                "vocab_size": config["vocab_size"],
                "max_seq_len": config["max_position_embeddings"],
                "weight_tying": config["tie_word_embeddings"]
            }
            modelargs = ModelArgs(**params)
        self.model_args = modelargs
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
        parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16'], help='Data type')
        parser.add_argument('--kvcache_block_size', type=int, default=256, help='KV cache block size')
        parser.add_argument('--sparse_decoding', action='store_true', help='Enable sparse decoding')
        parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
                
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
            sparse_decoding=args.sparse_decoding,
            checkpoint=args.checkpoint
        )
