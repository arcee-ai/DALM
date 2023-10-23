import argparse
import os
import sys

# ruff:noqa
from argparse import Namespace
from typing import Final, Literal

import torch

from datasets import Dataset
from torch.utils.data import DataLoader

from dalm.eval.utils import (
    construct_search_index,
    mixed_collate_fn,
    preprocess_dataset,
    get_passage_embeddings,
    evaluate_retriever_on_batch,
    print_eval_results,
)
from dalm.models.retriever_only_base_model import AutoModelForSentenceEmbedding
from dalm.utils import load_dataset

import logging

logger = logging.getLogger(__name__)

def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Testing a PEFT model for Sematic Search task")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="dataset path in the local dir. Can be huggingface dataset directory or a csv file.",
        required=True,
    )
    parser.add_argument("--query_column_name", type=str, default="query", help="name of the query col")
    parser.add_argument(
        "--passage_column_name",
        type=str,
        default="passage",
        help="name of the passage col",
    )
    parser.add_argument("--embed_dim", type=int, default=1024, help="dimension of the model embedding")
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--retriever_name_or_path",
        type=str,
        help="Path to pretrained retriever model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--retriever_peft_model_path",
        type=str,
        help="Path to the finetunned retriever peft layers",
        required=False,
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the test dataloader.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device. cpu or cuda.",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        help="torch.dtype to use for tensors. float16 or bfloat16.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top K retrieval",
    )
    parser.add_argument(
        "--is_autoregressive",
        action="store_true",
        help="Whether the model is autoregressive or not",
    )
    args = parser.parse_args()

    return args


def evaluate_retriever(
    dataset_or_path: Dataset | str,
    retriever_name_or_path: str,
    retriever_peft_model_path: str,
    passage_column_name: str,
    query_column_name: str,
    embed_dim: int,
    max_length: int,
    test_batch_size: int = 8,
    device: str = "cuda",
    torch_dtype: Literal["float16", "bfloat16"] = "float16",
    top_k: int = 10,
    is_autoregressive: bool = False,
) -> None:
    """Runs rag evaluation. See `dalm eval-retriever --help for details on params"""
    test_dataset = load_dataset(dataset_or_path)
    selected_torch_dtype: Final[torch.dtype] = torch.float16 if torch_dtype == "float16" else torch.bfloat16
    retriever_model = AutoModelForSentenceEmbedding(
        retriever_name_or_path,
        get_peft=False,
        use_bnb=False,
        is_autoregressive=is_autoregressive,
    )
    retriever_tokenizer = retriever_model.tokenizer

    processed_datasets = preprocess_dataset(
        test_dataset, retriever_tokenizer, query_column_name, passage_column_name, max_length
    )
    # peft config and wrapping
    if retriever_peft_model_path is not None:
        retriever_model.attach_pre_trained_peft_layers(retriever_peft_model_path, device)
    unique_passage_dataset, passage_embeddings_array = get_passage_embeddings(
        processed_datasets,
        passage_column_name,
        retriever_model.forward,
        device,
        embed_dim,
        selected_torch_dtype,
        test_batch_size,
    )

    id_to_passage = {i: p[passage_column_name] for i, p in enumerate(unique_passage_dataset)}

    logger.info("Construct passage index")
    passage_search_index = construct_search_index(embed_dim, len(passage_embeddings_array), passage_embeddings_array)

    # Initialize counters
    batch_precision = []
    batch_recall = []
    total_hit = 0

    logger.info("Evaluation start")
    processed_datasets_dataloader = DataLoader(
        processed_datasets, batch_size=test_batch_size, shuffle=True, collate_fn=mixed_collate_fn
    )

    for batch in processed_datasets_dataloader:
        _batch_precision, _batch_recall, _total_hit, _ = evaluate_retriever_on_batch(
            batch,
            passage_column_name,
            retriever_model.forward,
            passage_search_index,
            selected_torch_dtype,
            device,
            top_k,
            id_to_passage,
        )
        batch_precision.extend(_batch_precision)
        batch_recall.extend(_batch_recall)
        total_hit += _total_hit

    print_eval_results(len(processed_datasets), batch_precision, batch_recall, total_hit)


def main() -> None:
    args = parse_args()
    evaluate_retriever(
        dataset_or_path=args.dataset_path,
        retriever_name_or_path=args.retriever_name_or_path,
        retriever_peft_model_path=args.retriever_peft_model_path,
        passage_column_name=args.passage_column_name,
        query_column_name=args.query_column_name,
        embed_dim=args.embed_dim,
        max_length=args.max_length,
        test_batch_size=args.test_batch_size,
        device=args.device,
        torch_dtype=args.torch_dtype,
        top_k=args.top_k,
        is_autoregressive=args.is_autoregressive,
    )


if __name__ == "__main__":
    main()
