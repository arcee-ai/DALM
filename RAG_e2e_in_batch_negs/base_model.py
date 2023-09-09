import torch
from transformers import AutoModel, AutoModelForCausalLM,  AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model


class AutoModelForRagE2E(torch.nn.Module):
    def __init__(self, retriever_name, generator_name, normalize=True, get_peft=True):
        super(AutoModelForRagE2E, self).__init__()
        
        # Retriver initialization
        self.retriever_model = AutoModel.from_pretrained(
            retriever_name, load_in_4bit=True, 
        )
        self.retriever_tokenizer = AutoTokenizer.from_pretrained(
            retriever_name
        )        
        self.normalize = normalize
        
        # Generator initialization
        self.generator_model = AutoModelForCausalLM.from_pretrained(
            generator_name, load_in_4bit=True, trust_remote_code=True
        )
        
        self.generator_tokenizer = AutoTokenizer.from_pretrained(
            generator_name
        ) 
        
        if get_peft:
            
            preft_config_retriever = dict(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                target_modules=["key", "query", "value"],
                task_type="FEATURE_EXTRACTION",
            )
            
            self.retriever_model = self.use_peft(self.retriever_model, 
                                                preft_config_retriever,
                                                # TaskType.FEATURE_EXTRACTION 
                                                )
            
            # trainable_params = sum(p.numel() for p in self.retriever_model .parameters() if p.requires_grad)
            
            preft_config_generator = dict(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.generator_model = self.use_peft(self.generator_model, 
                                                preft_config_generator,
                                                # TaskType.CAUSAL_LM 
                                                )
            
            # ptrainable_params = sum(p.numel() for p in self.generator_model .parameters() if p.requires_grad)
 
        

    def forward(self, task, model, input_ids, attention_mask):
        
        if task == "retrieval":
            model_output = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = self.mean_pooling(model_output, attention_mask)
            if self.normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            return embeddings
        else:
            gen_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            return gen_outputs.logits

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
        
    def use_peft(self, model, peft_dict, task_type):
        peft_dict['task_type'] = task_type
        return  get_peft_model(model,  LoraConfig(**peft_dict))


    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)
