from llama_index.finetuning.embeddings.common import EmbeddingQAFinetuneDataset
import pandas as pd
from uuid import uuid4
from tqdm import tqdm
from datasets import load_dataset


DATA_NAME = "arcee-ai/synthetic-data-gen"

def map_df_to_llama_format(
) -> None:
	"""Generate examples given a set of nodes.

	The format of this is fairly confusing, especially since we already have the queries and answers.
	I'm just trying to map to the format created in
	https://github.com/jerryjliu/llama_index/blob/main/llama_index/finetuning/embeddings/common.py#L60
	"""
	ds = load_dataset(DATA_NAME)
	train_df = ds["train"].to_pandas()
	val_df = ds["test"].to_pandas()
	for fname, df in zip(["train.json", "val.json"], [train_df, val_df]):
		node_ids = [str(uuid4()) for _ in range(len(df))]
		node_dict = dict(zip(node_ids, df.Abstract.tolist()))

		queries = {}
		relevant_docs = {}
		for node_id, (_, row) in tqdm(zip(node_ids, df.iterrows()), total=len(df)):
			questions = [row["Question"]]
			questions = [question for question in questions if len(question) > 0]

			for question in questions:
				question_id = str(uuid4())
				queries[question_id] = question
				relevant_docs[question_id] = [node_id]

		# construct dataset
		dataset = EmbeddingQAFinetuneDataset(
			queries=queries, corpus=node_dict, relevant_docs=relevant_docs
		)
		dataset.save_json(fname)


if __name__ == "__main__":
	map_df_to_llama_format()
