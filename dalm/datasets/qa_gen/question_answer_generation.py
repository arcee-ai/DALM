import argparse

import datasets
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

device = "cuda:0" if torch.cuda.is_available() else "cpu"
TEST_SIZE = 0.2

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
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "potsawee/t5-large-generation-squad-QuestionAnswer", device_map="auto", load_in_8bit=True
)


def generate_question_answer_pairs(documents: dict) -> dict:
    """Generate question answer pairs"""
    texts = documents[args.passage_column_name]

    inputs = tokenizer(texts, return_tensors="pt", padding=True, max_length=150, truncation=True).to(device)
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
    question = record["Question"]
    answer = record["Answer"]

    return question != "-" and answer != "-"


dataset = datasets.load_dataset("csv", data_files={"data": args.dataset_path})["data"]

# shuffle data
dataset.shuffle(seed=42)

# select a subset
small_dataset = dataset.select(range(args.sample_size))

# train-test split
small_dataset_splits = shuffled_dataset.train_test_split(test_size=TEST_SIZE)

print(
    f"Train dataset size: {len(small_dataset_splits['train'])}, Test dataset size: {len(small_dataset_splits['test'])}"
)

for split_name in small_dataset_splits:
    processed_split = small_dataset_splits[split_name].map(
        generate_question_answer_pairs, batched=True, batch_size=args.batch_size
    )
    filtered_split = processed_split.filter(filter_malformed_questions)

    print(f"Malformed question answer pairs: {len(processed_split) - len(filtered_split)}")

    filtered_split.save_to_disk(f"{args.output_dir}/question_answer_pairs_{split_name}")
    filtered_split.to_csv(f"{args.output_dir}/question_answer_pairs_{split_name}.csv")

"""
python question_answer_generation.py \
    --dataset_path=knowledge_dataset.csv \
    --batch_size=1000 \
    --sample_size=1000000 \
    --output_dir=out
"""
