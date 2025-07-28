import torch
from torch import nn
import torch.distributed as dist

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, ColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear, KVParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.config import Config, ModelArgs
from nanovllm.utils.context import get_context

class Qwen3Attention(nn.Module):

    def __init__(
        self,
        layer_id: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        is_self_layer: bool,
        window_size: int,
        alibi: bool,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        sparse_decoding: bool = False,
        sparse_block_size: int = 16,
        sparse_block_ratio: float = 0.1,
        sparse_min_num_blocks: int = 16,
        sparse_local_num_blocks: int = 1,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.layer_id = layer_id
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.is_self_layer = is_self_layer
        self.window_size = window_size
        self.alibi = alibi
        if self.is_self_layer:
            self.qkv_proj = QKVParallelLinear(
                hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=qkv_bias,
            )
        else:
            self.q_proj = ColumnParallelLinear(
                hidden_size,
                self.total_num_heads * self.head_dim,
                bias=qkv_bias,
            )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        ) if self.is_self_layer and not alibi else None
        self.attn = Attention(
            self.layer_id,
            self.num_heads,
            self.head_dim,
            self.is_self_layer,
            self.window_size,
            self.alibi,
            self.scaling,
            self.num_kv_heads,
            sparse_decoding=sparse_decoding,
            sparse_block_size=sparse_block_size,
            sparse_block_ratio=sparse_block_ratio,
            sparse_min_num_blocks=sparse_min_num_blocks,
            sparse_local_num_blocks=sparse_local_num_blocks,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps) if self.is_self_layer else None
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps) if self.is_self_layer else None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.is_self_layer:
            qkv = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q_by_head = q.view(-1, self.num_heads, self.head_dim)
            q_by_head = self.q_norm(q_by_head)
            q = q_by_head.view(q.shape)
            k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
            k_by_head = self.k_norm(k_by_head)
            k = k_by_head.view(k.shape)
            if self.rotary_emb:
                q, k = self.rotary_emb(positions, q, k)
            o = self.attn(q, k, v)
        else:
            q = self.q_proj(hidden_states)
            o = self.attn(q, None, None)
        output = self.o_proj(o)
        return output


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        layer_id: int,
        model_args: ModelArgs,
        vllm_config: Config,
    ) -> None:
        super().__init__()
        self.is_self_layer = layer_id < model_args.n_layers - model_args.yoco_cross_layers
        if model_args.yoco_cross_layers > 0:
            self.window_size = model_args.yoco_window_size if self.is_self_layer else -1    
        else:
            self.window_size = -1
        self.self_attn = Qwen3Attention(
            layer_id=layer_id,
            hidden_size=model_args.d_model,
            num_heads=model_args.head,
            num_kv_heads=model_args.kv_head if self.is_self_layer else model_args.cross_kv_head,
            is_self_layer=self.is_self_layer,
            window_size=self.window_size,
            alibi=model_args.alibi and self.is_self_layer,
            max_position=vllm_config.max_model_len,
            rms_norm_eps=model_args.norm_eps,
            qkv_bias=model_args.attention_bias,
            head_dim=model_args.head_dim if self.is_self_layer else model_args.cross_head_dim,
            rope_theta=model_args.rope_theta,
            sparse_decoding=vllm_config.sparse_decoding,
            sparse_block_size=vllm_config.sparse_block_size,
            sparse_block_ratio=vllm_config.sparse_block_ratio,
            sparse_min_num_blocks=vllm_config.sparse_min_num_blocks,
            sparse_local_num_blocks=vllm_config.sparse_local_num_blocks,
        )
        self.mlp = Qwen3MLP(
            hidden_size=model_args.d_model,
            intermediate_size=model_args.d_ffn,
        )
        self.input_layernorm = RMSNorm(model_args.d_model, eps=model_args.norm_eps)
        self.post_attention_layernorm = RMSNorm(model_args.d_model, eps=model_args.norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        model_args: ModelArgs,
        vllm_config: Config,
    ) -> None:
        super().__init__()
        self.total_layers = model_args.n_layers
        self.yoco_cross_layers = model_args.yoco_cross_layers
        self.embed_tokens = VocabParallelEmbedding(model_args.vocab_size, model_args.d_model)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(layer_id, model_args, vllm_config) for layer_id in range(self.total_layers)])
        self.norm = RMSNorm(model_args.d_model, eps=model_args.norm_eps)
        if model_args.yoco_cross_layers > 0:
            self.kv_norm = RMSNorm(model_args.d_model, eps=model_args.norm_eps)
            self.kv_proj = KVParallelLinear(model_args.d_model, model_args.cross_head_dim, model_args.cross_kv_head, bias=model_args.attention_bias)
            self.shared_attention = self.layers[self.total_layers - self.yoco_cross_layers].self_attn.attn
            for layer in self.layers[self.total_layers - self.yoco_cross_layers:]:
                layer.self_attn.attn = self.shared_attention    
        

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers[:self.total_layers - self.yoco_cross_layers]:
            hidden_states, residual = layer(positions, hidden_states, residual)

        if self.yoco_cross_layers > 0:
            h_norm, _ = self.kv_norm(hidden_states, residual)
            kv = self.kv_proj(h_norm)
            key, value = kv.chunk(2, dim=-1)
            self.shared_attention.fill_kvcache(key.contiguous(), value.contiguous())
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            positions, hidden_states, residual = positions[last_indices].contiguous(), hidden_states[last_indices].contiguous(), residual[last_indices].contiguous()
        for layer in self.layers[self.total_layers - self.yoco_cross_layers:]:
            hidden_states, residual = layer(positions, hidden_states, residual)

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    def __init__(
        self,
        model_args: ModelArgs,
        vllm_config: Config,
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(model_args, vllm_config)
        self.lm_head = ParallelLMHead(model_args.vocab_size, model_args.d_model)
        if model_args.weight_tying:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
        self.packed_modules_mapping = {}
        for i in range(model_args.n_layers):
            if i < model_args.n_layers - model_args.yoco_cross_layers:
                self.packed_modules_mapping[f"model.layers.{i}.self_attn.q_proj"] = (f"model.layers.{i}.self_attn.qkv_proj", "q")
                self.packed_modules_mapping[f"model.layers.{i}.self_attn.k_proj"] = (f"model.layers.{i}.self_attn.qkv_proj", "k")
                self.packed_modules_mapping[f"model.layers.{i}.self_attn.v_proj"] = (f"model.layers.{i}.self_attn.qkv_proj", "v")
            self.packed_modules_mapping[f"model.layers.{i}.mlp.gate_proj"] = (f"model.layers.{i}.mlp.gate_up_proj", 0)
            self.packed_modules_mapping[f"model.layers.{i}.mlp.up_proj"] = (f"model.layers.{i}.mlp.gate_up_proj", 1)
        self.packed_modules_mapping["model.k_proj"] = ("model.kv_proj", "k")
        self.packed_modules_mapping["model.v_proj"] = ("model.kv_proj", "v")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits
