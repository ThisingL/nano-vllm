import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
import argparse

def main(model_path: str):
    path = os.path.expanduser(model_path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=512)
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
    args = argparse.parse_args()
    main(args.model_path)
    