
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from dalm.models.retriever_only_base_model import AutoModelForSentenceEmbedding


class AutoRegressiveModel(AutoModelForSentenceEmbedding):

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        model_output = self.model.sample(input_ids, attention_mask, output_hidden_states=True, return_dict_in_generate=True)
        embeddings = self.mean_pooling(model_output, attention_mask)
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings