from llama_index.embeddings import LinearAdapterEmbeddingModel, resolve_embed_model
from llama_index.finetuning import EmbeddingQAFinetuneDataset

from eval_utils import evaluate, display_results

def run_eval(val_data: str) -> None:
	val_dataset = EmbeddingQAFinetuneDataset.from_json(val_data)
	embed_model_name = "local:BAAI/bge-small-en"
	base_embed_model = resolve_embed_model("local:BAAI/bge-small-en")
	embed_model = LinearAdapterEmbeddingModel(base_embed_model, "model_output_test")
	bge_val_results = evaluate(val_dataset, embed_model_name)
	print("Base Model Results:")
	display_results(["bge"], [bge_val_results])
	ft_val_results = evaluate(val_dataset, embed_model)
	print("Fine-Tuned Model Results")
	display_results(["ft"], [ft_val_results])
	print("All Results")
	display_results(
		["bge", "ft"], [bge_val_results, ft_val_results]
	)




if __name__ == "__main__":
	run_eval("val.json")