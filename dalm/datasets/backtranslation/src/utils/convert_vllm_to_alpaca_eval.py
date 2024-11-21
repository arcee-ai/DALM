import argparse

from src.utils.io import dump_json, load_jsonlines


def main(args):
    vllm_data = load_jsonlines(args.input_filepath)
    alpaca_eval_data = []
    for ins in vllm_data:
        alpaca_eval_data.append(
            {
                "dataset": ins["raw"]["dataset"],
                "instruction": ins["raw"]["instruction"],
                "output": ins["response"],
                "generator": args.generator,
            }
        )
    dump_json(alpaca_eval_data, args.output_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_filepath", "-i", type=str, required=True)
    parser.add_argument("--output_filepath", "-o", type=str, required=True)
    parser.add_argument("--generator", "-g", default="Humback", type=str)
    args = parser.parse_args()
    main(args)
