from typing import Any, Dict

from datasets.formatting.formatting import LazyBatch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from accelerate.logging import get_logger

logger = get_logger(__name__)

def preprocess_dataset(
    examples: LazyBatch,
    retriever_tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    generator_tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    query_column_name: str,
    passage_column_name: str,
    answer_column_name: str,
    query_max_len: int,
    passage_max_len: int,
    generator_max_len: int,
) -> Dict[str, Any]:
    
    logger.info("preprocess_dataset called with batch size: %s", len(examples))

    querie_list = examples[query_column_name]
    passage_list = examples[passage_column_name]
    answers = examples[answer_column_name]

    queries = [f"#query# {query}" for query in querie_list]
    passages = [f"#passage# {passage}" for passage in passage_list]

    # Tokenization for the retriever
    logger.info("Tokenization for the retriever")
    retriever_query_tokens = retriever_tokenizer(
        queries, padding="max_length", max_length=query_max_len, truncation=True
    )
    retriever_passage_tokens = retriever_tokenizer(
        passages, padding="max_length", max_length=passage_max_len, truncation=True
    )

    # Tokenize for causal model
    # Here, we need to combine the query, passage, and the answer as the input, and the answer as the output
    casual_input_text = [
        f"#query# {query} #passage# {passage} #answer# {answer}"
        for passage, query, answer in zip(passages, queries, answers, strict=True)
    ]
    causal_input_tokens = generator_tokenizer(
        casual_input_text, padding="max_length", max_length=generator_max_len, truncation=True
    )

    query_passage_text = [
        f"#query# {query} #passage# {passage} #answer#" for passage, query in zip(passages, queries, strict=True)
    ]

    query_passage_lengths = []

    logger.info("generator_tokenizer()")
    query_passage_tokens = generator_tokenizer(query_passage_text, padding=False)
    logger.info("/generator_tokenizer()")

    for single_query_passage in query_passage_tokens["input_ids"]:
        query_passage_lengths.append(len(single_query_passage))

    pre_batch = {}

    # for the retriever in-batch negats
    for k, v in retriever_query_tokens.items():
        pre_batch[f"retriever_query_{k}"] = v
    for k, v in retriever_passage_tokens.items():
        pre_batch[f"retriever_passage_{k}"] = v

    # for the generator
    for k, v in causal_input_tokens.items():
        pre_batch[f"generator_input_{k}"] = v

    pre_batch["query_passage_input_len"] = query_passage_lengths

    return pre_batch
