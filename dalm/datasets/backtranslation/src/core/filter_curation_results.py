import argparse
import re
import statistics as sts
from collections import Counter

from tqdm import tqdm

from src.utils.io import dump_jsonlines, load_jsonlines


def main(args):
    data = load_jsonlines(args.data_filepath)
    regex = re.compile(r"[Ss]core:\s*(\d+)$")
    tgt_scores = list(map(int, args.scores.split(",")))

    scores = []
    qualified_results = []
    all_results = []
    instruction_lens = []
    response_lens = []
    for ins in tqdm(data, desc="Filtering curation results"):
        raw = ins["raw"]
        curation_response = ins[args.curation_response_column_name]
        score_matched = regex.search(curation_response)
        if score_matched:
            score = int(score_matched.group(1))
        else:
            score = None

        scores.append(str(score))

        all_results.append(
            {
                "instruction": raw["generated_instruction"],
                "response": raw["response"],
                "score": score,
            }
        )
        if isinstance(score, int) and score is not None and score in tgt_scores:
            if (
                args.min_instruction_len
                <= len(raw["generated_instruction"])
                <= args.max_instruction_len
            ):
                qualified_results.append(
                    {
                        "instruction": raw["generated_instruction"],
                        "response": raw["response"],
                        "score": score,
                    }
                )
                instruction_lens.append(len(raw["generated_instruction"]))
                response_lens.append(len(raw["response"]))

    dump_jsonlines(all_results, args.middle_save_filepath)
    dump_jsonlines(qualified_results, args.save_filepath)

    print(f"Scores: {Counter(scores).most_common()}")
    print(
        f"Number of qualified results (scores={args.scores}): {len(qualified_results)}/{len(all_results)}"
    )
    print(
        f"instruction len: {sts.mean(instruction_lens):.0f} ± {sts.stdev(instruction_lens):.0f}"
    )
    print(
        f"response len: {sts.mean(response_lens):.0f} ± {sts.stdev(response_lens):.0f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_filepath", type=str)
    parser.add_argument("--middle_save_filepath", type=str)
    parser.add_argument("--save_filepath", type=str)
    parser.add_argument("--curation_response_column_name", type=str, default="response")
    parser.add_argument(
        "--scores", type=str, default="5", help="scores separated in `,`. e.g. `3,4,5`."
    )
    parser.add_argument("--min_instruction_len", type=int, default=10)
    parser.add_argument("--max_instruction_len", type=int, default=800)
    args = parser.parse_args()

    main(args)
