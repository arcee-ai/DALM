import argparse
import os.path
import warnings
from functools import partial
from pathlib import Path

import datasets
import torch
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

import logging

logger = logging.getLogger(__name__)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TEST_SIZE = 0.2
QA_MODEL = "potsawee/t5-large-generation-squad-QuestionAnswer"


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
    args = parser.parse_args()
    return args


def generate_question_answer_pairs(
    documents: dict, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, passage_column_name: str
) -> dict:
    """Generate question answer pairs"""
    texts = documents[passage_column_name]

    inputs = tokenizer(texts, return_tensors="pt", padding=True, max_length=150, truncation=True).to(DEVICE)
    outputs = model.generate(**inputs, max_new_tokens=50)
    question_answers = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    question_answers = [
        question_answer.replace(tokenizer.pad_token, "").replace(tokenizer.eos_token, "")
        for question_answer in question_answers
    ]

    question_answer_pairs = [
        question_answer.split(tokenizer.sep_token) if question_answer.count(tokenizer.sep_token) == 1 else ["-", "-"]
        for question_answer in question_answers
    ]

    questions = [question_answer_pair[0].strip() for question_answer_pair in question_answer_pairs]
    answers = [question_answer_pair[1].strip() for question_answer_pair in question_answer_pairs]

    return {"Question": questions, "Answer": answers}


def filter_malformed_questions(record: dict) -> bool:
    return record["Question"] != "-" and record["Answer"] != "-"


def split_dataset(
    shuffled_dataset: datasets.Dataset, title_column_name: str, test_size: float = TEST_SIZE
) -> datasets.DatasetDict:
    unique_titles = set(shuffled_dataset[title_column_name])

    train_titles, test_titles = train_test_split(list(unique_titles), test_size=test_size, random_state=42)

    train_dataset = shuffled_dataset.filter(lambda example: example[title_column_name] in train_titles, num_proc=128)
    test_dataset = shuffled_dataset.filter(lambda example: example[title_column_name] in test_titles, num_proc=128)

    return datasets.DatasetDict(
        {
            "train": train_dataset,
            "test": test_dataset,
        }
    )


def generate_qa_from_dataset(
    dataset: Dataset, passage_column_name: str, title_column_name: str, sample_size: int, batch_size: int
) -> DatasetDict:
    logger.info(f"Generating question answer pairs with batch size: {batch_size}")
    tokenizer = AutoTokenizer.from_pretrained(QA_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(QA_MODEL, device_map="auto", load_in_8bit=True)
    # shuffle data
    dataset.shuffle(seed=42)
    # select a subset
    num_samples = min(sample_size, len(dataset))
    small_dataset = dataset.select(range(num_samples))
    # train-test split
    small_dataset_splits = split_dataset(small_dataset, title_column_name)
    logger.info(
        f"Train dataset size: {len(small_dataset_splits['train'])}, "
        f"Test dataset size: {len(small_dataset_splits['test'])}"
    )
    qa_gen_map = partial(
        generate_question_answer_pairs, model=model, tokenizer=tokenizer, passage_column_name=passage_column_name
    )
    processed_data = small_dataset_splits.map(qa_gen_map, batched=True, batch_size=batch_size)
    filtered_data = processed_data.filter(filter_malformed_questions)
    logger.info(
        f"Malformed question answer pairs: "
        f"(train: {len(processed_data['train']) - len(filtered_data['train'])} "
        f"test: {len(processed_data['test']) - len(filtered_data['test'])})"
    )
    return processed_data


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
) -> None:
    dataset = _load_dataset_from_path(dataset_path)
    qa_gen_data = generate_qa_from_dataset(dataset, passage_column_name, title_column_name, sample_size, batch_size)
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
    )


if __name__ == "__main__":
    main()

"""
python question_answer_generation.py \
    --dataset_path=knowledge_dataset.csv \
    --batch_size=1000 \
    --sample_size=1000000 \
    --output_dir=out
"""
