import os

import datasets
from datasets import Dataset


def load_dataset(dataset_or_path: str | Dataset) -> Dataset:
    """Loads the dataset

    :param dataset_or_path: disk path to a csv, dataset folder, Dataset, or hf hub
    """
    return (
        dataset_or_path
        if isinstance(dataset_or_path, Dataset)
        else datasets.load_from_disk(dataset_or_path)
        if os.path.isdir(dataset_or_path)
        else datasets.load_dataset("csv", data_files=dataset_or_path)["train"]
    )
