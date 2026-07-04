import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # 将 temperatures 先进行广播，然后用 logits 除以温度
        # div_：以 _ 结尾表示原地操作，节省显存
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1) # 将 logits 结果转成概率分布
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1) # Gumbel-Max trick 替代 multinomial 在 GPU 上运行更快
        return sample_tokens
