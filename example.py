import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("/data/yaoyaochang/code/speech/data/Qwen/Qwen3-0.6B")
    path = os.path.expanduser("/data/yaoyaochang/code/speech/data/Qwen/Qwen2-0.5B")
    path = os.path.expanduser("/data/yaoyaochang/code/speech/data/Qwen/Qwen2.5-0.5B")
    path = os.path.expanduser("/data/yaoyaochang/code/speech/data/Qwen/Qwen2.5-0.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=False, tensor_parallel_size=1, cuda_start_idx=2)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
