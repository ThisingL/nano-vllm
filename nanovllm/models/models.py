from nanovllm.models.llama import LlamaForCausalLM
from nanovllm.models.qwen2 import Qwen2ForCausalLM
from nanovllm.models.qwen3 import Qwen3ForCausalLM

model_dict = {
    "llama": LlamaForCausalLM,
    "qwen2": Qwen2ForCausalLM,
    "qwen3": Qwen3ForCausalLM,
}