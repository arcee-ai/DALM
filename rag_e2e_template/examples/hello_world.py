# 0. imports
import torch
from transformers import GPT2Tokenizer
import datasets

from trl import AutoModelForCausalLMWithRAG, RagE2EConfig, ArceeRagTrainer

dataset = datasets.load_dataset("csv", data_files="triplets.csv")

# 1. load a pretrained model
model = AutoModelForCausalLMWithRAG.from_pretrained("gpt2")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 2. initialize trainer
ppo_config = {"batch_size": 1}
config = RagE2EConfig(**ppo_config)
arcee_rag_trainer = ArceeRagTrainer(config, model, tokenizer)


# 3. encode a query
query_txt = ["I really like to drink "]

tokenized_query = tokenizer.batch_encode_plus(query_txt, padding=True, truncation=True)
query_tokens = tokenized_query['input_ids'][0]
query_am = tokenized_query['attention_mask'][0]

# this will control whether we are 
# marginalizing the next word prediction on the prompt or not
query_am_for_loss =  [ 0 for _ in range (len(query_tokens))]  # am for attention mask



# 4. encode passage
top_k_text = ["my morning tea."]

tokenized_top_k = tokenizer.batch_encode_plus(top_k_text, padding=True, truncation=True)
top_k_tokens = tokenized_top_k['input_ids'][0]
top_k_am = tokenized_top_k['attention_mask'][0]


# 5. combine the query with passages for  the casual llm
prompt_tensor = torch.tensor([query_tokens + top_k_tokens ]).to(model.pretrained_model.device)
prompt_am_tensor = torch.tensor([ query_am + top_k_am ]).to(model.pretrained_model.device)

# 5. cosine score tensor
# (this could be any reward such as human feedback or output from another model)
similarity_scores = torch.rand(5, device=model.pretrained_model.device)


# 6. train model with ppo
marginalize_casual_loss = arcee_rag_trainer.compute_marginalized_loss(
                            prompt_tensor,
                            prompt_am_tensor,
                            similarity_scores.unsqueeze(0),
                            len(query_tokens)
)

print(marginalize_casual_loss)

exit()

####

# normal training

# marginalize_casual_loss + retriever's in batch negative loss