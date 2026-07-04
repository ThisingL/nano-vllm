from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int, # 模型能处理的最大序列长度
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        # inv_freq[i] = 1 / base^(2i/d) (base 默认 1000000)
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float) # 所有的 token 位置编号
        # 外积算出所有的旋转角： freqs[i, j] = t[i] * inv_freq[j]， shape:[max_position_embeddings, rotary_dim/2]
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() # 算出 cosθ
        sin = freqs.sin() # 算出 sinθ
        # 把 cosθ 和 sinθ 拼起来，再增加一维给 num_heads，shape：shape:[max_position_embeddings, 1， rotary_dim]
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        # register_buffer：把 cache 注册为模型的一部分，会随模型一起移动到 GPU，但不是可训练参数（不参与梯度计算）
        # persistent=False：不保存到 checkpoint 文件里，因为这是可以从配置重新计算的中间量，不需要存储
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
