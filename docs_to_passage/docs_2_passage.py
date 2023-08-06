import argparse
import logging
import os

import faiss
import torch
from datasets

from split_utils import split_documents


logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Converting documents in to passages")
    parser.add_argument("--dataset_path", type=str, default=None, help="dataset path in the local dir")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final dataset.")
    parser.add_argument("--max_words", type=int, default=100, help="max number of words per passafe")
    parser.add_argument(
        "--num_proc",
        type=int,
        default=4,
        help="Number of processes to used during the data processing ",
    )

    args = parser.parse_args()

    return args

def main():
    
    args = parse_args()
    ######################################
    logger.info("Step 1 - Create the dataset")
    ######################################

    # The dataset needed for for splliting must consist of the following:
    # - title (string): title of the document
    # - text (string): text of a passage of the document
    # Let's say you have documents in tab-separated csv files with columns "title" and "text"
    assert os.path.isfile(args.dataset_path), "Please provide a valid path to a csv file"

    # You can load a Dataset object this way
    dataset = datasets.load_dataset("csv", data_files={"test": f"{args.dataset_path}/test.csv"})
    # Then split the documents into passages of 100 words
    dataset = dataset.map(split_documents, batched=True, num_proc=args.num_proc)

    # And finally save your dataset
    passages_path = os.path.join(args.output_dir, "knowledge_dataset")
    dataset.save_to_disk(passages_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)
    main()