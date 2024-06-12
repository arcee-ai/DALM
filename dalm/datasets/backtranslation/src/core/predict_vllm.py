
import os
import sys

sys.path.append(os.getcwd())

import argparse

from vllm import LLM, SamplingParams

from src.data import InferenceDataset
from src.utils.io import dump_jsonlines, load_jsonlines


def main(args):
    print("LLM")
    llm = LLM(
        args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
    )
    print("Sampling Params")
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    print("Loading data")

    raw_data = load_jsonlines(args.data_filepath)
    data = InferenceDataset(
        raw_data,
        content_name=args.prompt_column_name,
        reverse=args.reverse,
    )
    prompts = data.get_all()

    print("Generating")

    # 00:25 / 100 prompts on one GPU
    results = llm.generate(prompts, use_tqdm=True, sampling_params=sampling_params)

    # 07:24 / 100 prompts on one GPU
    # results = []
    # for prompt in tqdm(prompts):
    #     result = llm.generate(prompt, use_tqdm=False, sampling_params=sampling_params)
    #     results.append(result)

    dump_jsonl = []
    for raw, result in zip(raw_data, results):
        dump_jsonl.append(
            {
                "raw": raw,
                "full_prompt": result.prompt,
                "prompt": raw[args.prompt_column_name],
                "response": result.outputs[0].text,
            }
        )
    dump_jsonlines(dump_jsonl, args.save_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--data_filepath", type=str)
    parser.add_argument("--save_filepath", type=str)
    parser.add_argument("--prompt_column_name", type=str, default="instruction")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    main(args)
