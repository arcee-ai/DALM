import numpy as np
import hnswlib
import torch

def preprocess_passage_function(examples, tokenizer, max_len):
    passages = examples["passage"]
    result = tokenizer(passages, padding="max_length", max_length=max_len, truncation=True)
    return result

def construct_search_index(dim, num_elements, data):
    # Declaring index
    search_index = hnswlib.Index(space="ip", dim=dim)  # possible options are l2, cosine or ip

    # Initializing index - the maximum number of elements should be known beforehand
    search_index.init_index(max_elements=num_elements, ef_construction=200, M=100)

    # Element insertion (can be called several times):
    ids = np.arange(num_elements)
    search_index.add_items(data, ids)

    return search_index

def get_query_embeddings(query, model, tokenizer, max_len,  device):
    inputs = tokenizer(query, padding="max_length", max_length=max_len, truncation=True, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        query_embs = model(**{k: v.to(device) for k, v in inputs.items()}).detach().cpu()
    return query_embs[0]


def get_nearest_neighbours(k, search_index, query_embeddings, ids_to_passage_dict, threshold=0.7):
    # Controlling the recall by setting ef:
    search_index.set_ef(100)  # ef should always be > k

    # Query dataset, k - number of the closest elements (returns 2 numpy arrays)
    labels, distances = search_index.knn_query(query_embeddings, k=k)

    return [
        (ids_to_passage_dict[label], (1 - distance))
        for label, distance in zip(labels[0], distances[0])
        if (1 - distance) >= threshold
    ]
