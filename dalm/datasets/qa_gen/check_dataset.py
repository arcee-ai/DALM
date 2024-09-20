from datasets import load_from_disk


# Path to the dataset saved on disk
dataset_path = "/home/ec2-user/shamane/DALM/dalm/datasets/qa_gen/out/question_answer_pairs_train"

# Load the dataset
dataset = load_from_disk(dataset_path)

print(dataset)

# Iterate over and print all rows in the dataset
for idx, row in enumerate(dataset):
    print(f"Row {idx}: {row}")
