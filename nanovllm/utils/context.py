from dataclasses import dataclass
import torch


@dataclass
class Context:
    """记录全局状态的一个类

    参数：
    prefill 阶段：                                                              Decode 阶段：
    - cu_seqlens_q:  Q 的累积长度，告诉 FlashAttention 每条 seq 的 Q 边界           None
    - cu_seqlens_k:  K 的累积长度，有 Prefix Cache 时比 Q 长                       None
    - max_seqlen_q:  这批 seq 中最长的 Q 长度                                     None
    - max_seqlen_k:  这批 seq 中最长的 K 长度                                     None
    - slot_mapping:  每个新 token 的 K/V 要写到 cache 的哪个 block 中              每个新 token 的 K/V 要写到 cache 的哪个 block 中
    - context_lens:  None                                                      每条 seq 当前的总长度，FlashAttention 据此决定回看多远
    - block_tables:  只有命中 Prefix Cache 时才需要，用于 FlashAttention 查历史 KV  每条 seq 的"页表"，FlashAttention 用它找到所有历史 KV
    """
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    """获取全局的一个实例，这样这个实例就不用层层传递了"""
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0,
                slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
