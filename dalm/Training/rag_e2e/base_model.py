import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


class AutoModelForRagE2E(torch.nn.Module):
    def __init__(self, retriever_name: str, generator_name: str, normalize: bool = True, get_peft: bool = True) -> None:
        super(AutoModelForRagE2E, self).__init__()

        # Retriver initialization
        self.retriever_model = AutoModel.from_pretrained(
            retriever_name,
            quantization_config=AutoModelForRagE2E.__get_bnb_config(),
        )
        self.retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_name)
        self.normalize = normalize

        # Generator initialization
        self.generator_model = AutoModelForCausalLM.from_pretrained(
            generator_name,
            quantization_config=AutoModelForRagE2E.__get_bnb_config(),
            trust_remote_code=True,
        )

        self.generator_tokenizer = AutoTokenizer.from_pretrained(
            generator_name,
        )

        if get_peft:
            self.retriever_model = get_peft_model(
                self.retriever_model,
                peft_config=AutoModelForRagE2E.__get_lora_config(
                    TaskType.FEATURE_EXTRACTION,
                    target_modules=["key", "query", "value"],
                ),
            )

            self.generator_model = get_peft_model(
                self.generator_model,
                peft_config=AutoModelForRagE2E.__get_lora_config(
                    TaskType.CAUSAL_LM,
                    target_modules=["q_proj", "v_proj"],
                ),
            )

    def forward(
        self, task: str, model: AutoModel, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if task == "retrieval":
            model_output = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = self.mean_pooling(model_output, attention_mask)
            if self.normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            return embeddings
        else:
            gen_outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            return gen_outputs.logits

    def mean_pooling(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @staticmethod
    def __get_bnb_config() -> BitsAndBytesConfig:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    @staticmethod
    def __get_lora_config(
        task_type: TaskType,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=None,
    ) -> LoraConfig:
        return LoraConfig(
            task_type=task_type,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            target_modules=target_modules,
        )
