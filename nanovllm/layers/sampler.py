import torch
from torch import nn


class Sampler(nn.Module):
    '''
    温度 + Gumbel-Max 快速采样器
    '''
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        
        logits = logits.to(torch.float)
        greedy_tokens = logits.argmax(dim=-1) # 贪心解，温度 = 0 时直接用
        logits.div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        epsilon = 1e-10  
        # Gumbel-Max 近似: torch.empty_like(probs).exponential_(1) 生成了同形状的 Exp(1) 噪声, 除以这个噪声 + epsilon （防止除 0）,最后取 argmax 得到采样的类别。
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1) + epsilon).argmax(dim=-1)  
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)
