import argparse
import logging
import os.path
import warnings
from functools import partial
from pathlib import Path

import datasets
import torch
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from accelerate import Accelerator  # Import the Accelerator
from accelerate.utils import gather_object

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TEST_SIZE = 0.2
QA_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate question answer pairs from the dataset of passages")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="dataset path in the local dir. Can be huggingface dataset directory or a csv file.",
        required=True,
    )
    parser.add_argument(
        "--title_column_name",
        type=str,
        default="Title",
        help="This title is used to identify passages from the same text",
    )
    parser.add_argument(
        "--passage_column_name",
        type=str,
        default="Abstract",
        help="name of the passage column",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Batch size (per device) for generating question answer pairs.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=1000,
        help="Number of examples to process",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory. Without '/' at the end",
        required=True,
    )
    parser.add_argument(
        "--as_csv",
        action="store_true",
        help="Save the files as CSV. If False, will save them as a dataset directory via [`~Dataset.save_to_disk`]",
    )
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=512,
        help="Maximum number of input tokens for the model.",
    )
    args = parser.parse_args()
    return args


def generate_question_answer_pairs(
    documents: dict, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, passage_column_name: str, max_input_tokens: int, accelerator : Accelerator
) -> dict:

    """Generate question answer pairs"""
    texts = documents[passage_column_name]
    example_passage = (
        "Dense retrieval models are essential for embedding-based information "
        "retrieval systems, as they map queries and documents into a shared "
        "embedding space where their relevance can be computed. By using in-batch "
        "negative contrastive learning, these models can be trained more efficiently, "
        "as each batch contains not only positive examples but also negative samples "
        "from unrelated queries or documents. This approach helps optimize the model's "
        "ability to retrieve the most relevant information in real-world applications, "
        "such as question-answering systems, where precision is critical."
    )

    example_question = (
        "What role does in-batch negative contrastive learning play in training dense "
        "retrieval models, particularly in optimizing the retrieval of relevant information "
        "across different applications?"
   )

    prompt_template = (
        "Read the following passage and generate a single, relevant question based "
        "on its content. The question should be less than 100 words and more than 10 "
        "words. Do not generate anything other than the question itself. Avoid any tokens, "
        "explanations, or formatting. Do not use words like 'Question:', 'Answer:', 'Example:', or 'Passage:'. "
        "Ensure there are no line breaks in the output. The output should be the question only, nothing more.\n\n"
        "Example:\nPassage: {example_passage}\n{example_question}\n\nNow, do the same for the next "
        "passage:\n{passage}\n"
    )

    system_message = {
        "role": "system",
        "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    }

    batch_messages = [
        [system_message, {"role": "user", "content": prompt_template.format(
            example_passage=example_passage,
            example_question=example_question,
            passage=passage)}]
        for passage in texts
    ]
    
    batch_texts = tokenizer.apply_chat_template(
        batch_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    print("accelerator.device : ", accelerator.device)
    model_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_input_tokens).to(model.device)
    
    # Move model inputs to the correct device
    # Use accelerator.unwrap_model to get the original model
    unwrapped_model = accelerator.unwrap_model(model)

    # Generate outputs
    generated_ids = unwrapped_model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    results = []
    for response in responses:
        question = response.strip()
        results.append({"Question": question, "Answer": ""})
    
    return {
        "Question": [r["Question"] for r in results],
        "Answer": [r["Answer"] for r in results]
    }


def filter_malformed_questions(record: dict) -> bool:
    question = record["Question"]
    return (
        question is not None and
        question != "" and
        question != "-" and
        len(question.split()) >= 5 and
        not question.startswith("<") and
        "instruction" not in question.lower() and
        "question" not in question.lower() and
        "answer" not in question.lower() and
        "Answer:" not in question and
        "Question:" not in question
    )


def splitting_dataset(
    shuffled_dataset: datasets.Dataset, title_column_name: str, test_size: float = TEST_SIZE
) -> datasets.DatasetDict:
    unique_titles = set(shuffled_dataset[title_column_name])

    train_titles, test_titles = train_test_split(list(unique_titles), test_size=test_size, random_state=42)

    train_dataset = shuffled_dataset.filter(lambda example: example[title_column_name] in train_titles)
    test_dataset = shuffled_dataset.filter(lambda example: example[title_column_name] in test_titles)

    return datasets.DatasetDict(
        {
            "train": train_dataset,
            "test": test_dataset,
        }
    )


def generate_qa_from_dataset(
    dataset: Dataset, passage_column_name: str, title_column_name: str, sample_size: int, batch_size: int, max_input_tokens: int, load_in_8bit: bool = True
) -> DatasetDict:
    
    accelerator = Accelerator()  # Initialize the Accelerator

    if accelerator.is_main_process:
        logger.info(f"Generating question answer pairs with batch size: {batch_size}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(QA_MODEL)
    model = AutoModelForCausalLM.from_pretrained(QA_MODEL, torch_dtype="auto")
    model = accelerator.prepare(model)  # Prepare the model with accelerator
    
    # Shuffle data
    dataset = dataset.shuffle(seed=42)
    
    # Sample dataset
    num_samples = min(sample_size, len(dataset))
    small_dataset = dataset.select(range(num_samples))
    
    # Train-test split
    small_dataset_splits = splitting_dataset(small_dataset, title_column_name)
    logger.info(
        f"Train dataset size: {len(small_dataset_splits['train'])}, "
        f"Test dataset size: {len(small_dataset_splits['test'])}"
    )
    
    processed_data_splits = {}
    for split in ["train", "test"]:
        split_dataset = small_dataset_splits[split]
        
        # Shard the dataset among processes
        split_dataset = split_dataset.shard(num_shards=accelerator.num_processes, index=accelerator.process_index)
        
        # Process data in batches
        results = {
            "Question": [],
            "Answer": []
        }
        print("The length of split_dataset : ", len(split_dataset))
        for i in range(0, len(split_dataset), batch_size):
            batch = split_dataset[i:i+batch_size]
            print("Split dataset : ", split_dataset)
            batch_results = generate_question_answer_pairs(
                batch, model, tokenizer, passage_column_name, max_input_tokens, accelerator
            )
            results["Question"].extend(batch_results["Question"])
            results["Answer"].extend(batch_results["Answer"])

            #print("Results : ", batch_results["Question"])
        
        # Prepare results for gathering
        results_list = [results]
        gathered_results = gather_object(results_list)
        
        if accelerator.is_main_process:
            all_questions = []
            all_answers = []
            for res in gathered_results:
                all_questions.extend(res["Question"])
                all_answers.extend(res["Answer"])
            
            # Reconstruct the dataset
            processed_dataset = Dataset.from_dict({
                "Question": all_questions,
                "Answer": all_answers
            })
            
            processed_data_splits[split] = processed_dataset

    # Only the main process proceeds to filter and return data
    if accelerator.is_main_process:
        # Filter malformed questions
        filtered_data = DatasetDict()
        for split in ["train", "test"]:
            split_dataset = processed_data_splits[split]
            split_dataset = split_dataset.filter(filter_malformed_questions)
            filtered_data[split] = split_dataset
        
        logger.info(
            f"Malformed question answer pairs: "
            f"(train: {len(processed_data_splits['train']) - len(filtered_data['train'])} "
            f"test: {len(processed_data_splits['test']) - len(filtered_data['test'])})"
        )

        print("All questions from test split after filtering:")
        for i, example in enumerate(filtered_data['test']):
            print(f"{i + 1}: {example['Question']}")
        
        return filtered_data
    else:
        return None


def _load_dataset_from_path(dataset_path: str) -> Dataset:
    if dataset_path.endswith(".csv"):
        dataset = Dataset.from_csv(dataset_path)
    elif not os.path.splitext(dataset_path)[-1]:
        if os.path.isdir(dataset_path):
            dataset = datasets.load_from_disk(dataset_path)
        else:
            dataset = datasets.load_dataset(dataset_path)
            if isinstance(dataset, DatasetDict):
                if "train" in dataset:
                    key = "train"
                elif "training" in dataset:
                    key = "training"
                else:
                    key = next(iter(dataset))
                warnings.warn(f"Found multiple keys in dataset. Generating qa for split {key}", stacklevel=0)
                dataset = dataset[key]
    else:
        raise ValueError(
            "dataset-path must be one of csv, dataset directory "
            "(ie saved using [`~Dataset.save_to_disk`] or a dataset on the huggingface hub"
        )
    return dataset


def generate_qa_from_disk(
    dataset_path: str,
    passage_column_name: str,
    title_column_name: str,
    sample_size: int,
    batch_size: int,
    output_dir: str,
    as_csv: bool,
    max_input_tokens: int,
) -> None:
    dataset = _load_dataset_from_path(dataset_path)
    qa_gen_data = generate_qa_from_dataset(dataset, passage_column_name, title_column_name, sample_size, batch_size, max_input_tokens)
    
    # Only the main process proceeds to save the data
    if qa_gen_data is not None:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        for split_name, split_ds in qa_gen_data.items():
            full_path = f"{output_path}/question_answer_pairs_{split_name}"
            if as_csv:
                full_path = f"{full_path}.csv"
                split_ds.to_csv(full_path)
            else:
                split_ds.save_to_disk(full_path)
            logger.info(f"Saving split {split_name} to {full_path}")


def main() -> None:
    args = parse_args()
    generate_qa_from_disk(
        args.dataset_path,
        args.passage_column_name,
        args.title_column_name,
        args.sample_size,
        args.batch_size,
        args.output_dir,
        args.as_csv,
        args.max_input_tokens,
    )


if __name__ == "__main__":
    main()

"""
accelerate launch question_answer_generation_final.py \
    --dataset_path=knowledge_dataset.csv \
    --batch_size=8 \
    --sample_size=50 \
    --output_dir=out \
    --max_input_tokens=512
"""