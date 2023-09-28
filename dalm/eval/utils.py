from typing import Any, Callable, Dict, List, Tuple, cast

import hnswlib
import numpy as np
import torch
from datasets import Dataset
from datasets.formatting.formatting import LazyBatch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer, default_data_collator


def construct_search_index(dim: int, num_elements: int, data: np.ndarray) -> hnswlib.Index:
    # Declaring index
    search_index = hnswlib.Index(space="ip", dim=dim)  # possible options are l2, cosine or ip

    # A lower value of ef_construction will result in faster index construction
    # but might lead to a lower quality index,
    # meaning that searching within the index might be less accurate or slower.
    # A higher value will result in better index quality but might increase the
    # index construction time and memory usage.

    # the number of bi-directional links created for every new element during
    # construction. Reasonable range for M is 2-100. Higher M work better on
    # datasets with high intrinsic dimensionality and/or high recall,
    # while low M work better for datasets with low intrinsic dimensionality and/or
    # low recalls

    # Initializing index - the maximum number of elements should be known beforehand
    search_index.init_index(max_elements=num_elements, ef_construction=200, M=100)

    # Element insertion (can be called several times):
    ids = np.arange(num_elements)
    search_index.add_items(data, ids)

    return search_index


def get_nearest_neighbours(
    k: int,
    search_index: hnswlib.Index,
    query_embeddings: np.ndarray,
    ids_to_cat_dict: Dict[int, Any],
    threshold: float = 0.7,
) -> List[List[Tuple[str, float]]]:
    # Controlling the recall by setting ef:
    search_index.set_ef(100)  # ef should always be > k

    # Query dataset, k - number of the closest elements (returns 2 numpy arrays)
    labels, distances = search_index.knn_query(query_embeddings, k=k)

    results = []

    for i in range(len(labels)):
        results.append(
            [
                (ids_to_cat_dict[label], (1 - distance))
                for label, distance in zip(labels[i], distances[i], strict=True)
                if (1 - distance) >= threshold
            ]
        )

    return results


def calculate_precision_recall(retrieved_items: List, correct_items: List) -> Tuple[float, float]:
    # Convert lists to sets for efficient intersection and counting
    retrieved_set = set(retrieved_items)
    correct_set = set(correct_items)

    # Calculate the number of correctly retrieved items
    correctly_retrieved = len(retrieved_set.intersection(correct_set))

    # Calculate precision and recall
    precision = correctly_retrieved / len(retrieved_set)
    recall = correctly_retrieved / len(correct_set)

    return precision, recall


def preprocess_function(
    examples: LazyBatch,
    retriever_tokenizer: PreTrainedTokenizer,
    query_column_name: str = "query",
    passage_column_name: str = "passage",
    max_length: int = 128,
) -> Dict[str, torch.Tensor]:
    queries = examples[query_column_name]
    passages = examples[passage_column_name]

    # Tokenization for the retriever
    retriever_query_tokens = retriever_tokenizer(queries, padding="max_length", max_length=max_length, truncation=True)
    retriever_passage_tokens = retriever_tokenizer(
        passages, padding="max_length", max_length=max_length, truncation=True
    )

    pre_batch = {}

    # for the retriever in-batch negatives
    for k, v in retriever_query_tokens.items():
        pre_batch[f"retriever_query_{k}"] = v
    for k, v in retriever_passage_tokens.items():
        pre_batch[f"retriever_passage_{k}"] = v

    return pre_batch


def preprocess_dataset(
    dataset: Dataset, tokenizer: PreTrainedTokenizer, query_column_name: str, passage_column_name: str, max_length: int
) -> Dataset:
    """Runs the tokenizer on the dataset, returning the tokenizes columns"""
    processed_dataset = dataset.map(
        lambda example: preprocess_function(
            example,
            tokenizer,
            query_column_name=query_column_name,
            passage_column_name=passage_column_name,
            max_length=max_length,
        ),
        batched=True,
        # remove_columns=test_dataset.column_names,
        desc="Running tokenizer on dataset",
        num_proc=4,
    )
    return processed_dataset


def filter_unique_passages(dataset: Dataset, passage_column_name: str) -> Dataset:
    """Filters the dataset by the unique passages"""
    unique_passages = set(dataset[passage_column_name])

    def _is_passage_unique(example: Dict[str, Any]) -> bool:
        is_in_unique_list = example[passage_column_name] in unique_passages
        unique_passages.discard(example[passage_column_name])
        return is_in_unique_list

    unique_passage_dataset = dataset.filter(_is_passage_unique)
    return unique_passage_dataset


def mixed_collate_fn(batch: List[Dict[str, torch.Tensor | str]]) -> Dict[str, torch.Tensor | List[str]]:
    """
    This is able to account for string values which the default PyTorch collate_fn would silently ignore
    """
    new_batch: Dict[str, torch.Tensor | List[str]] = {}

    keys = batch[0].keys()
    for key in keys:
        if isinstance(batch[0][key], str) or batch[0][key] is None:
            # We cast because if the first element is a string, all elements in the batch are strings
            new_batch[key] = cast(List[str], [sample[key] for sample in batch])
        else:
            # Otherwise all elements in the batch are tensors
            new_batch[key] = torch.stack([torch.tensor(sample[key]) for sample in batch])

    return new_batch


def get_retriever_embeddings(
    forward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str,
    retriever_input_ids: torch.Tensor,
    retriever_attention_masks: torch.Tensor,
) -> np.ndarray:
    """ "Runs the provided forward function on the inputs and attention masks"""
    return (
        forward_fn(
            retriever_input_ids.to(device),
            retriever_attention_masks.to(device),
        )
        .detach()
        .float()
        .cpu()
        .numpy()
    )


def get_passage_embeddings(
    passage_dataset: Dataset,
    passage_column_name: str,
    forward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str,
    embed_dim: int,
    torch_dtype: torch.dtype,
    batch_size: int,
) -> tuple[Dataset, np.ndarray]:
    unique_passage_dataset = filter_unique_passages(passage_dataset, passage_column_name)

    unique_passage_dataloader = DataLoader(
        unique_passage_dataset,
        shuffle=False,
        collate_fn=default_data_collator,
        batch_size=batch_size,
        pin_memory=True,
    )

    num_passages = len(unique_passage_dataset)
    print(f"Starting to generate passage embeddings (Number of passages: {num_passages})")
    passage_embeddings_array = np.zeros((num_passages, embed_dim))
    for step, batch in enumerate(tqdm(unique_passage_dataloader)):
        with torch.no_grad():
            with torch.amp.autocast(dtype=torch_dtype, device_type=device):
                passage_embs = get_retriever_embeddings(
                    forward_fn,
                    device,
                    batch["retriever_passage_input_ids"],
                    batch["retriever_passage_attention_mask"],
                )

        start_index = step * batch_size
        end_index = start_index + batch_size if (start_index + batch_size) < num_passages else num_passages
        passage_embeddings_array[start_index:end_index] = passage_embs
        del passage_embs, batch
    return unique_passage_dataset, passage_embeddings_array


def evaluate_retriever_on_batch(
    batch: Dataset,
    passage_column_name: str,
    forward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    search_index: hnswlib.Index,
    torch_dtype: torch.dtype,
    device: str,
    top_k: int,
    id_to_passage: dict[int, str],
) -> tuple[list[float], list[float], int, list[str]]:
    """Evaluates the retriever on the given batch of data

    Returns the list[precision], list[recall], total hit, and list[top_passage_per_query] for this batch
    """
    batch_precision = []
    batch_recall = []
    total_hit = 0
    top_passages = []
    with torch.no_grad(), torch.amp.autocast(dtype=torch_dtype, device_type=device):
        # use the batch size for the first dim
        # do not hard-code it
        retriever_query_input_ids = batch["retriever_query_input_ids"]
        retriever_query_attention_mask = batch["retriever_query_attention_mask"]

        query_embeddings = get_retriever_embeddings(
            forward_fn,
            device,
            retriever_query_input_ids,
            retriever_query_attention_mask,
        )

    search_results = get_nearest_neighbours(
        top_k,
        search_index,
        query_embeddings,
        id_to_passage,
        threshold=0.0,
    )
    correct_passages = batch[passage_column_name]

    for i, result in enumerate(search_results):
        retrieved_passages = [passage for passage, score in result]
        # Closest match passage for this query's search results
        top_passages.append(retrieved_passages[0])
        correct_passage = [correct_passages[i]]
        precision, recall = calculate_precision_recall(retrieved_passages, correct_passage)
        batch_precision.append(precision)
        batch_recall.append(recall)
        hit = any(passage in retrieved_passages for passage in correct_passage)
        total_hit += hit
    return batch_precision, batch_recall, total_hit, top_passages


def print_eval_results(
    total_examples: int,
    precisions: list[float],
    recalls: list[float],
    total_hit: int,
) -> None:
    precision = sum(precisions) / total_examples
    recall = sum(recalls) / total_examples
    hit_rate = total_hit / float(total_examples)

    print("Retriever results:")
    print("Recall:", recall)
    print("Precision:", precision)
    print("Hit Rate:", hit_rate)
    print("*************")
