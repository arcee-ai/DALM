import statistics as sts

import os
import sys

sys.path.append(os.getcwd())

from src.utils.io import dump_jsonlines, load_jsonlines


def load_tree_data(
    tree_filepath,
    instruction_quality: float = 0.6,
    response_quality: float = 0.6,
    instruction_word_num: int = 5,
    response_word_num: int = 5,
    lang: str = "en",
    response_rank: int = 0,
):
    trees = load_jsonlines(tree_filepath)
    pairs = []

    def _traverse(ins: dict):
        for reply in ins["replies"]:
            if (
                ins.get("lang") == lang
                and reply.get("lang") == lang
                and reply.get("rank") == response_rank
            ):
                inst_qlt = ins["labels"].get("quality", {"value": 0.0})["value"]
                resp_qlt = reply["labels"].get("quality", {"value": 0.0})["value"]
                if inst_qlt > instruction_quality and resp_qlt > response_quality:
                    if (
                        len(ins["text"].split()) > instruction_word_num
                        and len(reply["text"].split()) > response_word_num
                    ):
                        pairs.append(
                            {
                                "instruction": ins["text"],
                                "instruction_quality": inst_qlt,
                                "response": reply["text"],
                                "response_quality": resp_qlt,
                            }
                        )
        for reply in ins["replies"]:
            _traverse(reply)

    for tree in trees:
        prompt = tree["prompt"]
        _traverse(prompt)

    return pairs


if __name__ == "__main__":
    dump_num = 3200
    pairs = load_tree_data("data/seed/2023-04-12_oasst_ready.trees.jsonl")
    print(f"#data: {len(pairs)}, #dump: {dump_num}")
    pairs.sort(
        key=lambda ins: ins["instruction_quality"] + ins["response_quality"],
        reverse=True,
    )
    dump_data = pairs[:dump_num]
    instruction_lens = []
    response_lens = []
    for ins in dump_data:
        instruction_lens.append(len(ins["instruction"]))
        response_lens.append(len(ins["response"]))
    print(
        f"Instruction len: {sts.mean(instruction_lens):.0f}±{sts.stdev(instruction_lens):.0f}, "
        f"Response len: {sts.mean(response_lens):.0f}±{sts.stdev(response_lens):.0f}"
    )
    dump_jsonlines(dump_data, "data/seed/seed.jsonl")
