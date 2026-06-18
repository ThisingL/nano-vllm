import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384 # 一次前向计算最多处理多少个 token（所有并发请求加起来）
    max_num_seqs: int = 512 # 同时并发处理的最大请求数量
    max_model_len: int = 4096 # 单个请求（输入+输出）最长能有多少 token
    gpu_memory_utilization: float = 0.9 # 使用 GPU 显存的比例上限
    tensor_parallel_size: int = 1
    enforce_eager: bool = False  # 是否禁用 CUDA Graph 优化
    hf_config: AutoConfig | None = None # HuggingFace 模型的配置对象
    eos: int = -1 # 结束符的 token ID
    kvcache_block_size: int = 256 # KV cache 每个 block 存多少个 token
    num_kvcache_blocks: int = -1 # KV cache block 的数量

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
