from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    context_lens: torch.Tensor | None = None
    slot_mapping: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    # This pair is used for sliding window attention
    sliding_slot_mapping: torch.Tensor | None = None
    sliding_block_tables: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, sliding_slot_mapping=None, context_lens=None, sliding_block_tables=None, slot_mapping=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, context_lens, sliding_slot_mapping, sliding_block_tables, slot_mapping, block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
