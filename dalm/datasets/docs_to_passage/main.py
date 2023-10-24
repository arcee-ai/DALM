import logging
import os
from functools import partial
from typing import Optional, Union

import datasets
import pandas as pd
import typer
from datasets import Dataset

from dalm.datasets.docs_to_passage.utils import (
    DEFAULT_MAX_WORDS,
    DEFAULT_MIN_WORDS,
    TEXT_COL,
    TITLE_COL,
    keep_sufficiently_long_passages,
    split_documents,
)

logger = logging.getLogger(__name__)

# We won't use more than 500k documents for training
MAX_NUM_DOCUMENTS = 500_000


def docs_to_passages(
    df: Union[Dataset, pd.DataFrame],
    max_words: int = DEFAULT_MAX_WORDS,
    title_col: str = TITLE_COL,
    text_col: str = TEXT_COL,
    max_docs: Optional[int] = None,
) -> Dataset:
    """Take the dataframe from supabase and split each document into chunks of 100

    :param df: The dataframe/raw data from supabase. Df must have columns
        'title' and 'text'
    :param max_words: The max words per passage. Default 100
    :param title_col: The column in df that corresponds to the title. Default 'title'
    :param text_col: The column in df that corresponds to the title. Default 'text'
    :param max_docs: The maximum number of documents to process. If not passed, all documents are processed
    """
    logger.info("Create the dataset by converting docs to passages")

    cols = df.columns if isinstance(df, pd.DataFrame) else df.column_names
    assert title_col in cols and text_col in cols, (
        f"{title_col} and {text_col} must be present in df. If your columns don't "
        f"match, pass in values to `title_col` and `text_col`"
    )
    if isinstance(df, pd.DataFrame):
        dataset = Dataset.from_pandas(df[[title_col, text_col]])
    else:
        dataset = df
    dataset = dataset.rename_columns({title_col: TITLE_COL, text_col: TEXT_COL})
    dataset = dataset.select_columns(column_names=[TITLE_COL, TEXT_COL])

    if max_docs is not None and len(dataset) > max_docs:
        logger.info("Split into training/test set")
        dataset = dataset.train_test_split(train_size=max_docs)["train"]

    # Then split the documents into passages of 100 words
    batch_size = 1000
    logger.info(f"Split into passages of {max_words} words with batch size of {batch_size}")
    split_docs = partial(split_documents, max_words=max_words)
    dataset = dataset.map(split_docs, batched=True, batch_size=batch_size)

    # Filter the dataset using the defined function
    logger.info(f"Filter the dataset to remove passages less than {DEFAULT_MIN_WORDS} words")
    filter_abstracts = partial(keep_sufficiently_long_passages)
    filtered_dataset = dataset.filter(filter_abstracts)
    return filtered_dataset


def main(
    dataset_path: str,
    title_col: str = TITLE_COL,
    text_col: str = TEXT_COL,
    output_dir: str = ".",
    max_words: int = DEFAULT_MAX_WORDS,
    max_docs: Optional[int] = None,
) -> None:
    """Converts documents into passages of length max_words"""
    ds = datasets.load_from_disk(dataset_path)
    passages = docs_to_passages(ds, max_words, title_col, text_col, max_docs)
    passages_path = os.path.join(output_dir, "passages.csv")
    passages.to_csv(passages_path)


if __name__ == "__main__":
    typer.run(main)
