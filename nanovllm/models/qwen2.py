import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen2Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead

class Qwen2Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta


        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=self.rope_theta,
            rope_scaling=rope_scaling
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output = self.o_proj(attn_output)
        
        return output


class Qwen2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str
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
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x
    

class Qwen2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen2Config
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None)
        )

        self.mlp = Qwen2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None
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

class Qwen2Model(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    def __init__(
        self,
        config: Qwen2Config
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen2DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        print(f"init Qwen2Model here at {self.__class__.__name__} with config: {config}")

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None
    ) -> torch.Tensor:
        if hidden_states is None:
            hidden_states = self.embed_tokens(input_ids)
        # 以下都是enforce_eager=True时输出的信息
        # print(f"input_ids.shape = {input_ids.shape}") # [2] 2=batchsize  
        # print(f"positions.shape = {positions.shape}") # [2]
        # print(f"hidden_states.shape = {hidden_states.shape}") # [2, 896] 896是hidden size
        # print(f"input_ids = {input_ids}") # 2个数字，分别表示各自seq的最新token id
        # print(f"positions = {positions}") # 2个数字，分别表示各自seq的位置信息，从小数字一直涨
        # '''
        # prefilling：多个seq拼接起来一起prefill
        # input_ids.shape = torch.Size([70])
        # positions.shape = torch.Size([70])
        # hidden_states.shape = torch.Size([70, 896])
        # input_ids = tensor([151644,   8948,    198,   2610,    525,   1207,  16948,     11,   3465,
        #         553,  54364,  14817,     13,   1446,    525,    264,  10950,  17847,
        #             13, 151645,    198, 151644,    872,    198,    396,  47845,   6133,
        #         151645,    198, 151644,  77091,    198, 151644,   8948,    198,   2610,
        #         525,   1207,  16948,     11,   3465,    553,  54364,  14817,     13,
        #         1446,    525,    264,  10950,  17847,     13, 151645,    198, 151644,
        #         872,    198,   1607,    678,  10250,   5109,   2878,    220,     16,
        #             15,     15, 151645,    198, 151644,  77091,    198],
        #     device='cuda:2')
        # positions = tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
        #         18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,  0,  1,  2,  3,
        #         4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        #         22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]
                
        # decode：接着最新的input_ids和positions继续decode，一个token一个token的decode，输入是多个seq的最新一个token
        # input_ids.shape = torch.Size([2])
        # positions.shape = torch.Size([2])
        # hidden_states.shape = torch.Size([2, 896])
        # input_ids = tensor([ 9707, 39814], device='cuda:2')
        # positions = tensor([33, 39], device='cuda:2')
        # input_ids.shape = torch.Size([2])
        # positions.shape = torch.Size([2])
        # hidden_states.shape = torch.Size([2, 896])
        # input_ids = tensor([0, 0], device='cuda:2')
        # positions = tensor([34, 40], device='cuda:2')
        # '''
        # with open("tmp.txt", "a") as f:
        #     f.write(f"input_ids.shape = {input_ids.shape}\n")
        #     f.write(f"positions.shape = {positions.shape}\n")
        #     f.write(f"hidden_states.shape = {hidden_states.shape}\n")
        #     f.write(f"input_ids = {input_ids}\n")
        #     f.write(f"positions = {positions}\n")
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
    
# class Qwen2ModelHF(Qwen2Model):
#     def __init__(self, config: Qwen2Config) -> None:
#         super().__init__(config)
        
#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         positions: torch.Tensor
#     ) -> torch.Tensor:
#         hidden_states = super().forward(input_ids, positions)
#         return hidden_states

class Qwen2ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self, 
        config: Qwen2Config
    ) -> None:
        super().__init__()
        self.model = Qwen2Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

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