import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank() # 当前进程的 rank_id
        self.tp_size = dist.get_world_size() # 一共有多少进程
        # 偷懒了，为了简洁不用考虑不均匀，当然词表 151936 对 1、2、4、8 都是可整除的
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx) # 这里的运算法则是 Broadcasting（广播）
            x = mask * (x - self.vocab_start_idx) # 确保 x 的索引是合法的位置，PyTorch 对越界索引的行为是未定义的
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y # 将 shape 右补 1
            dist.all_reduce(y) # 同步等待所有 rank 完成计算
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            # prefill 阶段只计算每一个 seq 的最后一个 token 的 logits
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous() # 压缩成最后想要的几行
        logits = F.linear(x, self.weight) # X @ W^T
        if self.tp_size > 1:
            # 由于只有主进程负责最后的采样，因此只要用 gather 就行
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0) # gather(tensor, gather_list, dst)，dst 是收集到的目标进程
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None # -1 表示沿最后一个维度拼
        return logits
