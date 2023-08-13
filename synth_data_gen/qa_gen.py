import pandas as pd
import argparse
import logging
import os

import datasets

from transformers import pipeline

from qa_gen_utils import generate_question

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Converting documents in to passages")
    parser.add_argument("--dataset_path", type=str, default=None, help="dataset path in the local dir")
    parser.add_argument("--qa_gen_model_name", type=str, default="voidful/context-only-question-generator", help="hf-hub model path")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final dataset.")
    parser.add_argument(
        "--num_proc",
        type=int,
        default=4,
        help="Number of processes to used during the data processing ",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Number of inference batches",
    )
    args = parser.parse_args()

    return args


def main():
    
    args = parse_args()

    # The dataset needed for for splliting must consist of the following:
    # - title (string): title of the document
    # - text (string): text of a passage of the document (100 words passages)t"

    # You can load a Dataset object this way
    dataset = datasets.load_from_disk(args.dataset_path)
  
    question_generator = pipeline(task="text2text-generation", model=args.qa_gen_model_name, device="cuda:0")

    # Add a question for each passage
    dataset_question = dataset.map(
        lambda x: {"question": generate_question(x['Abstract'],question_generator)},
        batched = True,
        batch_size=args.batch_size,
    )

    # And finally save your dataset
    question_dataset_path = os.path.join(args.output_dir, "dataset_with_question.csv")
    dataset_question["abstracts"].to_csv(question_dataset_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)
    main()


#  python synth_data_gen/qa_gen.py  --dataset_path "./dataset/knowledge_dataset" --output_dir "./dataset"