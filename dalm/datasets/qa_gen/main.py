import os

import datasets
import torch.cuda
import typer
from datasets import Dataset
from transformers import BartForConditionalGeneration, BartTokenizerFast

from dalm.datasets.qa_gen.utils import generate_question

DEFAULT_MODEL = "voidful/context-only-question-generator"
DEFAULT_BATCH_SIZE = 128


def generate_questions_from_passages(
    ds: Dataset,
    qa_model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    text_col: str = "text",
) -> Dataset:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BartForConditionalGeneration.from_pretrained(qa_model).to(device)
    tokenizer = BartTokenizerFast.from_pretrained(qa_model)

    assert text_col in ds.features, f"Dataset missing {text_col}. Pass correct value into `text_col`"

    print("GENERATING QA WITH BATCH SIZE", batch_size)
    dataset_with_questions = ds.map(
        lambda row: {"question": generate_question(row[text_col], model, tokenizer)},
        batched=True,
        batch_size=batch_size,
        # num_proc=8
    )
    return dataset_with_questions


def main(
    dataset_path: str,
    qa_gen_model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    output_dir: str = ".",
    target_col: str = "text",
) -> None:
    """Reads the data from local, generates qa, writes results to disk"""
    ds = datasets.load_from_disk(dataset_path)
    ds_with_questions = generate_questions_from_passages(ds, qa_gen_model_name, batch_size, target_col)
    question_dataset_path = os.path.join(output_dir, "dataset_with_question.csv")
    ds_with_questions.to_csv(question_dataset_path)


if __name__ == "__main__":
    typer.run(main)
