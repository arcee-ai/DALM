from typing import Dict

import torch
from datasets.formatting.formatting import LazyBatch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


def preprocess_dataset(
    examples: LazyBatch,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    query_col_name: str,
    passage_col_name: str,
) -> Dict[str, torch.Tensor]:
    query_list = examples[query_col_name]
    queries = [f"#query# {query}" for query in query_list]
    result_ = tokenizer(queries, padding="max_length", max_length=128, truncation=True)
    result_ = {f"query_{k}": v for k, v in result_.items()}

    passage_list = examples[passage_col_name]
    passages = [f"#passage# {passage}" for passage in passage_list]
    result_passage = tokenizer(passages, padding="max_length", max_length=128, truncation=True)
    for k, v in result_passage.items():
        result_[f"passage_{k}"] = v

    return result_
