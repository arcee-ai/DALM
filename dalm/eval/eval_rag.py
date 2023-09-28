import argparse
from argparse import Namespace
from typing import Final, List, Literal

import torch
from accelerate.logging import get_logger
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)

from dalm.eval.utils import (
    construct_search_index,
    evaluate_retriever_on_batch,
    get_passage_embeddings,
    mixed_collate_fn,
    preprocess_dataset,
    print_eval_results,
)
from dalm.models.rag_e2e_base_model import AutoModelForRagE2E
from dalm.utils import load_dataset

logger = get_logger(__name__)


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
    parser.add_argument("--answer_column_name", type=str, default="answer", help="name of the query col")
    parser.add_argument("--embed_dim", type=int, default=1024, help="dimension of the model embedding")
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
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
        "--generator_name_or_path",
        type=str,
        help="Path to pretrained generator model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--retriever_peft_model_path",
        type=str,
        help="Path to the finetuned retriever peft layers",
        required=False,
    )
    parser.add_argument(
        "--generator_peft_model_path",
        type=str,
        help="Path to the finetuned generator peft layers",
        required=False,
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the test dataloader.",
    )
    parser.add_argument(
        "--query_batch_size",
        type=int,
        default=16,
        help="Batch size for generator input",
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
        "--evaluate_generator",
        action="store_true",
        help="Enable generator evaluation.",
    )
    args = parser.parse_args()

    return args


def run_generator_on_prompts(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompts: List[str], max_length: int = 256
) -> List[str]:
    """Runs the generator model over the prompts (query + passage)"""
    # TODO: Is the max_length correct here? not sure
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    with torch.autocast("cuda"), torch.no_grad():
        outputs = model.generate(**inputs.to("cuda"), max_length=max_length, early_stopping=True)
    return tokenizer.batch_decode(outputs.cpu(), skip_special_tokens=True)


def eval_generator_on_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    queries: List[str],
    passages: List[str],
    query_batch_size: int,
    queries_for_gen_eval: list[str],
    max_length: int,
) -> tuple[list[str], list[str]]:
    generated_answers_for_eval = []
    for _query, search_result_passage in zip(queries, passages, strict=True):
        # This query comes without the answer
        query = f"#query# {_query} #passage# {search_result_passage} #answer# "
        queries_for_gen_eval.append(query)
        # Generate answers in batch
        if len(queries_for_gen_eval) >= query_batch_size:
            batch_answers = run_generator_on_prompts(model, tokenizer, queries_for_gen_eval, max_length=max_length)
            generated_answers_for_eval.extend(batch_answers)
            queries_for_gen_eval.clear()

    return queries_for_gen_eval, generated_answers_for_eval


def evaluate_rag(
    dataset_or_path: str | Dataset,
    retriever_name_or_path: str,
    generator_name_or_path: str,
    retriever_peft_model_path: str,
    generator_peft_model_path: str,
    passage_column_name: str,
    query_column_name: str,
    answer_column_name: str,
    embed_dim: int,
    max_length: int,
    test_batch_size: int = 8,
    query_batch_size: int = 16,
    device: str = "cuda",
    torch_dtype: Literal["float16", "bfloat16"] = "float16",
    top_k: int = 10,
    evaluate_generator: bool = True,
) -> None:
    """Runs rag evaluation. See `dalm eval-rag --help for details on params"""
    test_dataset = load_dataset(dataset_or_path)
    selected_torch_dtype: Final[torch.dtype] = torch.float16 if torch_dtype == "float16" else torch.bfloat16
    # rag retriever and the generator (don't load new peft layers no need)
    rag_model = AutoModelForRagE2E(retriever_name_or_path, generator_name_or_path)

    processed_datasets = preprocess_dataset(
        test_dataset, rag_model.retriever_tokenizer, query_column_name, passage_column_name, max_length
    )
    # peft config and wrapping
    rag_model.attach_pre_trained_peft_layers(retriever_peft_model_path, generator_peft_model_path, device)
    unique_passage_dataset, passage_embeddings_array = get_passage_embeddings(
        processed_datasets,
        passage_column_name,
        rag_model.retrieval_forward,
        device,
        embed_dim,
        selected_torch_dtype,
        test_batch_size,
    )

    id_to_passage = {i: p[passage_column_name] for i, p in enumerate(unique_passage_dataset)}

    print("Construct passage index")
    passage_search_index = construct_search_index(embed_dim, len(passage_embeddings_array), passage_embeddings_array)

    # Initialize counters
    batch_precision = []
    batch_recall = []
    total_hit = 0
    # For the generator. hint: use chatGPT to see what is Exact match when evaluating question answer models
    total_em_hit = 0

    print("Evaluation start")
    # For evaluating the generator
    queries_for_gen_eval: list[str] = []
    generated_answers_for_eval = []
    model = rag_model.generator_model
    tokenizer = rag_model.generator_tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    processed_datasets_dataloader = DataLoader(
        processed_datasets, batch_size=test_batch_size, shuffle=True, collate_fn=mixed_collate_fn
    )

    # to:do : torch_dtype make a variables float16 or bfloat16
    for batch in processed_datasets_dataloader:
        queries = batch[query_column_name]
        _batch_precision, _batch_recall, _total_hit, top_passages = evaluate_retriever_on_batch(
            batch,
            passage_column_name,
            rag_model.retrieval_forward,
            passage_search_index,
            selected_torch_dtype,
            device,
            top_k,
            id_to_passage,
        )
        batch_precision.extend(_batch_precision)
        batch_recall.extend(_batch_recall)
        total_hit += _total_hit
        if not evaluate_generator:
            continue

        # Evaluate the generator
        queries_for_gen_eval, batch_answers = eval_generator_on_batch(
            model, tokenizer, queries, top_passages, query_batch_size, queries_for_gen_eval, max_length
        )
        generated_answers_for_eval.extend(batch_answers)

    if not evaluate_generator:
        print_eval_results(len(processed_datasets), batch_precision, batch_recall, total_hit)
        return

    # TODO: imperative style code, refactor in future but works for now
    # If there are any leftover batches to query
    if len(queries_for_gen_eval) > 0:
        batch_answers = run_generator_on_prompts(model, tokenizer, queries_for_gen_eval, max_length=max_length)
        generated_answers_for_eval.extend(batch_answers)
        queries_for_gen_eval.clear()

    answers = processed_datasets[answer_column_name]

    for generated_answer, answer in zip(generated_answers_for_eval, answers, strict=True):
        generated_answer_strings = generated_answer.split("#answer#")
        if len(generated_answer_strings) < 2:
            continue

        generated_answer_string = generated_answer_strings[1].strip()
        if generated_answer_string == answer:
            total_em_hit += 1

    print_eval_results(len(processed_datasets), batch_precision, batch_recall, total_hit)
    print("Generator evaluation:")
    print("Exact match:", total_em_hit / len(processed_datasets))


def main() -> None:
    args = parse_args()
    evaluate_rag(
        dataset_or_path=args.dataset_path,
        retriever_name_or_path=args.retriever_name_or_path,
        generator_name_or_path=args.generator_name_or_path,
        retriever_peft_model_path=args.retriever_peft_model_path,
        generator_peft_model_path=args.generator_peft_model_path,
        passage_column_name=args.passage_column_name,
        query_column_name=args.query_column_name,
        answer_column_name=args.answer_column_name,
        embed_dim=args.embed_dim,
        max_length=args.max_length,
        test_batch_size=args.test_batch_size,
        query_batch_size=args.query_batch_size,
        device=args.device,
        torch_dtype=args.torch_dtype,
        top_k=args.top_k,
        evaluate_generator=args.evaluate_generator,
    )


if __name__ == "__main__":
    main()
