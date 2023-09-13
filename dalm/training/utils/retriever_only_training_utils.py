from typing import Dict

import torch
from datasets.formatting.formatting import LazyBatch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


def preprocess_dataset(
    examples: LazyBatch, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
) -> Dict[str, torch.Tensor]:
    queries = examples["question"]
    result_ = tokenizer(queries, padding="max_length", max_length=512, truncation=True)
    result_ = {f"query_{k}": v for k, v in result_.items()}

    passage = examples["Abstract"]
    result_passage = tokenizer(passage, padding="max_length", max_length=512, truncation=True)
    for k, v in result_passage.items():
        result_[f"passage_{k}"] = v

    return result_
