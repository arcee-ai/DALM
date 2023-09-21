from llama_index.finetuning import EmbeddingAdapterFinetuneEngine, EmbeddingQAFinetuneDataset
from llama_index.embeddings import resolve_embed_model
import torch



def run_finetune(train_data: str) -> None:
	base_embed_model = resolve_embed_model("local:BAAI/bge-small-en")
	train_dataset = EmbeddingQAFinetuneDataset.from_json(train_data)

	finetune_engine = EmbeddingAdapterFinetuneEngine(
		train_dataset,
		base_embed_model,
		model_output_path="model_output_test",
		epochs=1,
		verbose=True,
		batch_size=153
		# optimizer_class=torch.optim.SGD,
		# optimizer_params={"lr": 0.01}
	)
	finetune_engine.finetune()


if __name__ == "__main__":
	print("Training on train.json")
	run_finetune("train.json")
