from typing import Any, Dict, List, Tuple

import hnswlib
import numpy as np
import torch
from datasets.formatting.formatting import LazyBatch
from transformers import PreTrainedTokenizer


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
) -> List[Tuple[str, float]]:
    # Controlling the recall by setting ef:
    search_index.set_ef(100)  # ef should always be > k

    # Query dataset, k - number of the closest elements (returns 2 numpy arrays)
    labels, distances = search_index.knn_query(query_embeddings, k=k)

    return [
        (ids_to_cat_dict[label], (1 - distance))
        for label, distance in zip(labels[0], distances[0], strict=True)
        if (1 - distance) >= threshold
    ]


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
    query_col_name: str = "query",
    passage_col_name: str = "passage",
) -> Dict[str, torch.Tensor]:
    queries = examples[query_col_name]
    passages = examples[passage_col_name]

    # Tokenization for the retriever
    retriever_query_tokens = retriever_tokenizer(queries, padding="max_length", max_length=128, truncation=True)
    retriever_passage_tokens = retriever_tokenizer(passages, padding="max_length", max_length=128, truncation=True)

    pre_batch = {}

    # for the retriever in-batch negatives
    for k, v in retriever_query_tokens.items():
        pre_batch[f"retriever_query_{k}"] = v
    for k, v in retriever_passage_tokens.items():
        pre_batch[f"retriever_passage_{k}"] = v

    return pre_batch
