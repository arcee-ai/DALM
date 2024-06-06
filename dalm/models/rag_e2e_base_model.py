from enum import Enum
from typing import List, Optional, Union

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from dalm.utils import eos_mask


class Mode(Enum):
    GENERATOR = "generator"
    RETRIEVER = "retriever"
    BOTH = "both"


class AutoModelForRagE2E(torch.nn.Module):
    def __init__(
        self,
        retriever_name: str,
        generator_name: str,
        normalize: bool = True,
        get_peft: Optional[Mode] = None,
        use_bnb: Optional[Mode] = None,
        retriever_is_autoregressive: bool = False,
    ) -> None:
        super(AutoModelForRagE2E, self).__init__()

        # Retriever initialization
        self.retriever_model = AutoModel.from_pretrained(
            retriever_name,
            quantization_config=AutoModelForRagE2E.__get_bnb_config()
            if use_bnb in [Mode.RETRIEVER, Mode.BOTH]
            else None,
        )
        self.retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_name)
        if retriever_is_autoregressive:
            self.retriever_tokenizer.add_eos_token = True
            self.retriever_tokenizer.pad_token = self.retriever_tokenizer.eos_token

        self.normalize = normalize

        # Generator initialization
        self.generator_model = AutoModelForCausalLM.from_pretrained(
            generator_name,
            quantization_config=AutoModelForRagE2E.__get_bnb_config()
            if use_bnb in [Mode.GENERATOR, Mode.BOTH]
            else None,
            trust_remote_code=True,
        )

        self.generator_tokenizer = AutoTokenizer.from_pretrained(
            generator_name,
        )

        if get_peft is not None:
            if get_peft in [Mode.RETRIEVER, Mode.BOTH]:
                self.retriever_model = get_peft_model(
                    self.retriever_model,
                    peft_config=AutoModelForRagE2E.__get_lora_config(
                        TaskType.FEATURE_EXTRACTION,
                        target_modules=["key", "query", "value"]
                        if not retriever_is_autoregressive
                        else ["q_proj", "v_proj"],
                    ),
                )

            if get_peft in [Mode.GENERATOR, Mode.BOTH]:
                self.generator_model = get_peft_model(
                    self.generator_model,
                    peft_config=AutoModelForRagE2E.__get_lora_config(
                        TaskType.CAUSAL_LM,
                        target_modules=["q_proj", "v_proj"],
                    ),
                )
        self.retriever_is_autoregressive = retriever_is_autoregressive

    def retrieval_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.retriever_is_autoregressive:
            # we take the last hidden state of the model
            token_embeddings = self.retriever_model(
                input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True
            ).hidden_states[-1]

            attention_mask = eos_mask(attention_mask)
        else:
            # First element of model_output contains all token embeddings
            token_embeddings = self.retriever_model(input_ids, attention_mask)[0]

        embeddings = self.mean_pooling(token_embeddings, attention_mask)
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def forward(self, task: str, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if task == "retrieval":
            return self.retrieval_forward(input_ids, attention_mask)
        else:
            gen_outputs = self.generator_model(input_ids=input_ids, attention_mask=attention_mask)
            return gen_outputs.logits

    def mean_pooling(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def attach_pre_trained_peft_layers(
        self, peft_retriever_path: Optional[str], peft_generator_path: Optional[str], device: str
    ) -> None:
        if peft_retriever_path is not None:
            self.retriever_model = (
                PeftModel.from_pretrained(
                    self.retriever_model, peft_retriever_path, load_in_4bit=True, device_map="auto"
                )
                .to(device)
                .eval()
                .merge_and_unload()
            )

        if peft_generator_path is not None:
            self.generator_model = (
                PeftModel.from_pretrained(
                    self.generator_model, peft_generator_path, load_in_4bit=True, device_map="auto"
                )
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
        task_type: TaskType,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        bias: str = "none",
        target_modules: Optional[Union[List[str], str]] = None,
    ) -> LoraConfig:
        return LoraConfig(
            task_type=task_type,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            target_modules=target_modules,
        )
