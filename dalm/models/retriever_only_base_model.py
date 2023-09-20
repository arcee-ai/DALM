from typing import List, Optional, Union

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig


class AutoModelForSentenceEmbedding(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        normalize: bool = True,
        use_bnb: bool = True,
        get_peft: bool = True,
    ) -> None:
        super(AutoModelForSentenceEmbedding, self).__init__()

        self.model = AutoModel.from_pretrained(
            model_name,
            device_map={"": 0},
            quantization_config=AutoModelForSentenceEmbedding.__get_bnb_config() if use_bnb else None,
        )

        if get_peft:
            self.model = get_peft_model(
                self.model,
                peft_config=AutoModelForSentenceEmbedding.__get_lora_config(
                    target_modules=["key", "query", "value"],
                ),
            )

        self.normalize = normalize
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, **kwargs: torch.Tensor) -> torch.Tensor:
        model_output = self.model(**kwargs)
        embeddings = self.mean_pooling(model_output, kwargs["attention_mask"])
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def mean_pooling(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def __getattr__(self, name: str) -> Union[torch.Tensor, torch.nn.modules.module.Module]:
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def attach_pre_trained_peft_layers(self, peft_retriever_path: str, device: str) -> None:
        self.model = (
            PeftModel.from_pretrained(self.model, peft_retriever_path, load_in_4bit=True, device_map="auto")
            .to(device)
            .eval()
            .merge_and_unload()
        )

    @staticmethod
    def __get_bnb_config() -> BitsAndBytesConfig:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    @staticmethod
    def __get_lora_config(
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        bias: str = "none",
        target_modules: Optional[Union[List[str], str]] = None,
    ) -> LoraConfig:
        return LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            target_modules=target_modules,
        )
