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
        is_autoregressive: bool = False,
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
                peft_config=AutoModelForSentenceEmbedding.__get_lora_config(is_autoregressive=is_autoregressive),
            )

        self.normalize = normalize
        self.is_autoregressive = is_autoregressive
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.is_autoregressive:
            # we take the last hidden state of the model
            token_embeddings = self.model(
                input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True
            ).hidden_states[-1]
        else:
            # First element of model_output contains all token embeddings
            token_embeddings = self.model(input_ids, attention_mask)[0]
        embeddings = self.mean_pooling(token_embeddings, attention_mask)
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
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
        is_autoregressive: bool = False,
    ) -> LoraConfig:
        return LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION if not is_autoregressive else TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            target_modules=target_modules
            or (["key", "query", "value"] if not is_autoregressive else ["q_proj", "v_proj"]),
        )
