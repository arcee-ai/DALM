import os

import datasets
import torch
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


def eos_mask(mask: torch.Tensor, padding: str = "left") -> torch.Tensor:
    """
    Returns a mask that only masks everything else but the last token of each sequence.
    """
    new_mask = torch.zeros_like(mask)

    if padding == "right":
        ones_counts = mask.sum(dim=1)
        indices = (torch.arange(mask.size(0)), ones_counts - 1)
        new_mask[indices] = 1
    else:
        new_mask[:, -1] = 1

    return new_mask
