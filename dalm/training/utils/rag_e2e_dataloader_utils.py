from typing import Any, Dict, Optional

from datasets.formatting.formatting import LazyBatch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

DEFAULT_SYS_PROMPT = "You are a meticulous, honest and helpful assistant"

def llama2_prompt(query: str, context:str, answer:Optional[str]=None) -> str:
        prompt = f"<s>[INST] <<SYS>>\n{DEFAULT_SYS_PROMPT}\n<</SYS>>\n\n Answer this: {query} Using context: {context} [/INST]"
        if answer:
             prompt += f" {answer} </s>"
        return prompt

def preprocess_dataset(
    examples: LazyBatch,
    retriever_tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    generator_tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    dataset_query_col_name: str,
    dataset_passage_col_name: str,
    dataset_answer_col_name: str,
    query_max_len: int,
    passage_max_len: int,
    generator_max_len: int,
) -> Dict[str, Any]:
    querie_list = examples[dataset_query_col_name]
    passage_list = examples[dataset_passage_col_name]
    answers = examples[dataset_answer_col_name]

    queries = [f"{query}" for query in querie_list]
    passages = [f"{passage}" for passage in passage_list]

    # Tokenization for the retriever
    retriever_query_tokens = retriever_tokenizer(
        queries, padding="max_length", max_length=query_max_len, truncation=True
    )
    retriever_passage_tokens = retriever_tokenizer(
        passages, padding="max_length", max_length=passage_max_len, truncation=True
    )

    # Tokenize for causal model
    # Here, we need to combine the query, passage, and the answer as the input, and the answer as the output
    casual_input_text = [
        llama2_prompt(query, passage, answer)
        for passage, query, answer in zip(passages, queries, answers, strict=True)
    ]
    causal_input_tokens = generator_tokenizer(
        casual_input_text, padding="max_length", max_length=generator_max_len, truncation=True
    )

    query_passage_text = [
        llama2_prompt(query,passage) for passage, query in zip(passages, queries, strict=True)
    ]

    query_passage_lengths = []

    query_passage_tokens = generator_tokenizer(query_passage_text, padding=False)

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
