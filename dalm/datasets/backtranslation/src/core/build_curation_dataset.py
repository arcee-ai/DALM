import argparse

from tqdm import tqdm

from src.utils.io import dump_jsonlines, load_jsonlines


def main(args):
    prompt_template = open(args.curation_prompt_filepath, "r").read().strip()
    data = load_jsonlines(args.data_filepath)
    results = []
    for ins in tqdm(data, desc="Building curation dataset"):
        generated_instruction = ins[args.generated_instruction_column_name]
        response = ins[args.response_column_name]
        prompt = prompt_template.format(
            generated_instruction=generated_instruction,
            response=response,
        )
        results.append(
            {
                "prompt": prompt,
                "generated_instruction": generated_instruction,
                "response": response,
            }
        )
    dump_jsonlines(results, args.save_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_filepath", type=str)
    parser.add_argument("--save_filepath", type=str)
    parser.add_argument(
        "--curation_prompt_filepath", type=str, default="data/prompts/self_curation.txt"
    )
    parser.add_argument(
        "--generated_instruction_column_name", type=str, default="response"
    )
    parser.add_argument("--response_column_name", type=str, default="prompt")
    args = parser.parse_args()

    main(args)
