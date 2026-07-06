import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
import argparse

def main(args: str):
    path = os.path.expanduser(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(
        path, enforce_eager=args.enforce_eager, tensor_parallel_size=args.tensor_parallel_size
    )

    sampling_params = SamplingParams(
        temperature=args.temperature, max_tokens=args.max_tokens
    )
    history = []  # 多轮对话历史

    print("已加载模型，开始对话（输入 'quit' 或 'exit' 退出，输入 'clear' 清空历史）")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出对话")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("退出对话")
            break
        if user_input.lower() == "clear":
            history = []
            print("已清空对话历史")
            continue

        history.append({"role": "user", "content": user_input})

        prompt = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
        )

        outputs = llm.generate([prompt], sampling_params)
        reply = outputs[0]["text"]

        history.append({"role": "assistant", "content": reply})
        print(f"\nAssistant: {reply}")


if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description="nano vllm")
    argparse.add_argument(
        "--model-path", type=str, default="/home/lixin370/nano-vllm/models/Qwen3-0.6B/",
    )
    argparse.add_argument("--tensor-parallel-size", "--tp", type=int, default=1)
    argparse.add_argument(
        "--enforce-eager", type = bool, default=True, help="禁用 CUDA Graph 优化"
    )
    argparse.add_argument("--temperature", type=float, default=0.6)
    argparse.add_argument("--max-tokens", type=int, default=512)
    args = argparse.parse_args()
    main(args)
