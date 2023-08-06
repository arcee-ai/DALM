import argparse
from tqdm import tqdm
import numpy as np

import datasets

import evaluate
import torch
from torch.utils.data import DataLoader

import transformers
from transformers import AutoTokenizer, default_data_collator

from accelerate.logging import get_logger
from peft import PeftModel


from contrastive_train.base_model import AutoModelForSentenceEmbedding
from test_utils import preprocess_passage_function, construct_search_index, get_query_embeddings, get_nearest_neighbours

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Testing a PEFT model for Sematic Search task")
    parser.add_argument("--dataset_path", type=str, default=None, help="dataset path in the local dir")
    parser.add_argument("--passage_column_name", type=str, default="passage", help="name of the passage col")
    parser.add_argument("--query_column_name", type=str, default="query", help="name of the query col")
    parser.add_argument("--embed_dim", type=int, default=1024, help="dimension of the model embedding")
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--peft_model_path",
        type=str,
        help="Path to the finetunned peft layers",
        required=True,
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
        help="Device",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top K retrieval",
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    
    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # work with the passage dataset
    dataset = datasets.load_dataset("csv", data_files={"test": f"{args.dataset_path}/test.csv"})
    dataset = dataset["test"].select(args.passage_column_name)
    index_column = [i for i in range(len(dataset))]
    passage_dataset_for_indexing = dataset.add_column("index", index_column)
    
    # tokenization
    tokenized_passages = dataset.map(
    preprocess_passage_function,
    batched=True,
    fn_kwargs={"tokenizer": tokenizer, "max_len": args.max_len},
    remove_columns=dataset.column_names,
    desc="Running tokenizer on the passage dataset",
    )
    
    dataloader = DataLoader(
        tokenized_passages,
        shuffle=False,
        collate_fn=default_data_collator,
        batch_size=args.test_batch_size,
        pin_memory=True,
    )
    
    # for the evaluation
    # not needed in the production
    ids_to_passage_dict = {i: p for i, p in zip(dataset["index"], 
                                                 passage_dataset_for_indexing[args.passage_column_name])}
    
    # base model
    model = AutoModelForSentenceEmbedding(args.model_name_or_path, tokenizer)

    # peft config and wrapping
    model = PeftModel.from_pretrained(model, args.peft_model_id)
    
    model.to(args.device)
    model.eval()
    # This method merges the LoRa layers into the base model. 
    # This is needed if someone wants to use the base model as a standalone model.
    model = model.merge_and_unload()
    
    num_passages = len(dataset)

    passage_embeddings_array = np.zeros((num_passages, args.embed_dim))
    for step, batch in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                passage_embs = model(**{k: v.to(args.device) for k, v in batch.items()}).detach().float().cpu()
        start_index = step * args.test_batch_size
        end_index = start_index + args.test_batch_size if (start_index + args.test_batch_size) < num_passages else num_passages
        passage_embeddings_array[start_index:end_index] = passage_embs
        del passage_embs, batch
    
    passage_search_index = construct_search_index(args.embed_dim, num_passages, passage_embeddings_array)
    
    query = "NLP and ML books"
    k = args.top_k
    query_embeddings = get_query_embeddings(query, model, tokenizer, args.max_len, args.device)
    search_results = get_nearest_neighbours(k, passage_search_index, query_embeddings, ids_to_passage_dict, threshold=0.7)

    print(f"{query=}")
    for product, cosine_sim_score in search_results:
        print(f"cosine_sim_score={round(cosine_sim_score,2)} {product=}")
            
    
    
    
    
    

