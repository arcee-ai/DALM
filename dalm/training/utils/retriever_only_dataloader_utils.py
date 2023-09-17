from typing import Dict

import torch
from datasets.formatting.formatting import LazyBatch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


def preprocess_dataset(
    examples: LazyBatch,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    query_col_name: str,
    passage_col_name: str,
    query_max_len: int,
    passage_max_len: int,
) -> Dict[str, torch.Tensor]:
    query_list = examples[query_col_name]
    queries = [f"#query# {query}" for query in query_list]
    result_ = tokenizer(queries, padding="max_length", max_length=query_max_len, truncation=True)
    result_ = {f"query_{k}": v for k, v in result_.items()}

    passage_list = examples[passage_col_name]
    passages = [f"#passage# {passage}" for passage in passage_list]
    result_passage = tokenizer(passages, padding="max_length", max_length=passage_max_len, truncation=True)
    for k, v in result_passage.items():
        result_[f"passage_{k}"] = v

    return result_
