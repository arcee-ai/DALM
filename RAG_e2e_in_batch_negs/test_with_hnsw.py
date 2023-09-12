import argparse
from tqdm import tqdm
import numpy as np

import datasets

import torch
from torch.utils.data import DataLoader

import transformers
from transformers import default_data_collator

from accelerate.logging import get_logger
from peft import PeftModel

from base_model import AutoModelForRagE2E
from test_utils import (
    preprocess_function,
    construct_search_index,
    # get_query_embeddings,
    get_nearest_neighbours,
    calculate_precision_recall,
)

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Testing a PEFT model for Sematic Search task"
    )
    parser.add_argument(
        "--dataset_path", type=str, default=None, help="dataset path in the local dir"
    )
    parser.add_argument(
        "--query_column_name", type=str, default="query", help="name of the query col"
    )
    parser.add_argument(
        "--passage_column_name",
        type=str,
        default="passage",
        help="name of the passage col",
    )
    parser.add_argument(
        "--answer_column_name", type=str, default="answer", help="name of the query col"
    )
    parser.add_argument(
        "--embed_dim", type=int, default=384, help="dimension of the model embedding"
    )
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
        "--retriever_model_name_or_path",
        type=str,
        help="Path to pretrained retriever model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--generator_model_name_or_path",
        type=str,
        help="Path to pretrained generator model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--retriever_peft_model_path",
        type=str,
        help="Path to the finetunned retriever peft layers",
        required=True,
    )
    parser.add_argument(
        "--generator_peft_model_path",
        type=str,
        help="Path to the finetunned generator peft layers",
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

    # rag retriver and the generator (don't load new peft layers no need)
    rag_model = AutoModelForRagE2E(
        args.retriever_model_name_or_path,
        args.generator_model_name_or_path,
        get_peft=False,
    )

    # load the test dataset
    test_dataset = datasets.load_dataset(
        "csv", data_files={"test": f"{args.dataset_path}/triplets.csv"}
    )["test"]
    
    # test_dataset = datasets.load_from_disk("/home/datasets/question_answer_pairs")
    
    generator_tokenizer = rag_model.generator_tokenizer
    generator_tokenizer.pad_token = generator_tokenizer.eos_token
    
    retriever_tokenizer = rag_model.retriever_tokenizer
    

    processed_datasets = test_dataset.map(
        lambda example: preprocess_function(
            example,
            retriever_tokenizer,
            retriever_tokenizer,
            query_col_name=args.query_column_name,
            passage_col_name=args.passage_column_name,
            answer_col_name=args.answer_column_name,
        ),
        batched=True,
        # remove_columns=test_dataset.column_names,
        desc="Running tokenizer on dataset",
        num_proc=4,
    )
    
 
    unique_passages = set(processed_datasets[args.passage_column_name])

    def is_passage_unique(example):
        is_in_unique_list = example[args.passage_column_name] in unique_passages
        unique_passages.discard(example[args.passage_column_name])
        return is_in_unique_list

    unique_passage_dataset = processed_datasets.filter(is_passage_unique)
    
    passage_to_id_dict = {i: p[args.passage_column_name] for i, p in enumerate(unique_passage_dataset)}
    
    unique_passage_dataloader = DataLoader(
        unique_passage_dataset,
        shuffle=False,
        collate_fn=default_data_collator,
        batch_size=args.test_batch_size,
        pin_memory=True,
    )
    
    # peft config and wrapping
    retriever_with_peft_layers = rag_model.retriever_model
    # (
    #     PeftModel.from_pretrained(rag_model.retriever_model, args.peft_retriever_path)
    #     .to(args.device)
    #     .eval()
    #     .merge_and_unload()
    # )

    generator_with_peft_layers = rag_model.generator_model
    # (
    #     PeftModel.from_pretrained(rag_model.generator_model, args.peft_generator_path)
    #     .to(args.device)
    #     .eval()
    #     .merge_and_unload()
    # )

    num_passages = len(unique_passage_dataset)
    
    passage_embeddings_array = np.zeros((num_passages, args.embed_dim))
    for step, batch in enumerate(tqdm(unique_passage_dataloader)):
        with torch.no_grad():
            with torch.amp.autocast(dtype=torch.float16, device_type="cuda"):
                passage_embs = rag_model.forward(
                    "retrieval", 
                    retriever_with_peft_layers, 
                    batch["retriever_passage_input_ids"].to("cuda"), 
                    batch["retriever_passage_attention_mask"].to("cuda")
                ).detach().float().cpu().numpy()
                
        start_index = step * args.test_batch_size
        end_index = start_index + args.test_batch_size if (start_index + args.test_batch_size) < num_passages else num_passages
        passage_embeddings_array[start_index:end_index] = passage_embs
        del passage_embs, batch

    passage_search_index = construct_search_index(args.embed_dim, num_passages, passage_embeddings_array)
    
    # Initialize counters
    batch_precision = []
    batch_recall = []
    total_hit = 0
    
    
    print("Retriever evaluation start")
    
    # here we are interacting through the dataset, not a dataloader
    # so we need to convert them to a tensor
    # to do : convert this to batches by examples from the dataset to make it effcient
    # to:do : remove hard-coded args like device cpu  or cuda
    # to:do : torch_dtype make a varaibles float16 or bfloat16
    # to:do :  remove bilterplate codes and create two functios to get_query_embeds and get_passage_embedins 
    for step, test_example in enumerate(processed_datasets):
        with torch.no_grad():
            with torch.amp.autocast(dtype=torch.float16, device_type="cuda"):
                # use the batch size for the first dim
                # do not hard-code it
                retriever_query_input_ids = torch.tensor(test_example["retriever_query_input_ids"]).view(1, -1)
                retriever_query__attention_mask = torch.tensor(test_example["retriever_query_attention_mask"]).view(1, -1)
                
                query_embeddings = rag_model.forward(
                    "retrieval",
                    retriever_with_peft_layers,
                    retriever_query_input_ids.to("cuda"),
                    retriever_query__attention_mask.to("cuda"),
                ).detach().float().cpu().numpy()
           
   
        search_results = get_nearest_neighbours(args.top_k, passage_search_index, query_embeddings, passage_to_id_dict, threshold=0.0)
        
        retrieved_passages = [item[0] for item in search_results]
        
        correct_passages = [test_example[args.passage_column_name]]
        
   
        precision, recall = calculate_precision_recall(retrieved_passages, correct_passages)

        batch_precision.append(precision)
        batch_recall.append(recall)

        hit = any(passage in retrieved_passages for passage in correct_passages)
        total_hit += hit

    total_examples = len(processed_datasets)
    recall = sum(batch_recall) / total_examples
    precision = sum(batch_precision) / total_examples
    hit_rate = total_hit / float(total_examples)
    
    
    print("Retriever results:")

    print("Recall:", recall)
    print("Precision:", precision)
    print("Hit Rate:", hit_rate)
    
    
    print("*************")
    print(("Generator evaluation:"))
    
    # for the geneator
    total_em_hit = 0 # hinit use chatGPT to see what is Exact match when evalauting question answer models
    for test_example in processed_datasets:
        
        # select query:
        with torch.no_grad():
            with torch.amp.autocast(dtype=torch.float16, device_type="cuda"):
                # use the batch size for the first dim
                # do not hard-code it
                retriever_query_input_ids = torch.tensor(test_example["retriever_query_input_ids"]).view(1, -1)
                retriever_query__attention_mask = torch.tensor(test_example["retriever_query_attention_mask"]).view(1, -1)
                
                query_embeddings = rag_model.forward(
                    "retrieval",
                    retriever_with_peft_layers,
                    retriever_query_input_ids.to("cuda"),
                    retriever_query__attention_mask.to("cuda"),
                ).detach().float().cpu().numpy()
           
   
        search_result_passage = get_nearest_neighbours(args.top_k, passage_search_index, query_embeddings, passage_to_id_dict, threshold=0.0)[0][0]
        
        # this query comes without the answer
        query = f"#query# {test_example[args.query_column_name]} #passage# {search_result_passage} #answer# "
        
        answer = test_example[args.answer_column_name]
        
      
        pipeline = transformers.pipeline(
            "text-generation",
            model=generator_with_peft_layers,
            tokenizer= rag_model.generator_tokenizer,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )
        

        sequences = pipeline(
            query,
            max_length=200,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id= rag_model.generator_tokenizer.eos_token_id,
        )
        
        
    
        generated_answer_string = sequences[0]["generated_text"].split("#answer#")[1]
        
        if generated_answer_string == answer:
            total_em_hit =  total_em_hit +1
            
        
    print("Exact mactch:", total_em_hit/len(processed_datasets))
    



if __name__ == "__main__":
    main()
