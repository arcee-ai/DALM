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
query_txt = "This morning I went to the "

query_tokens = tokenizer.encode(query_txt)  #.to(model.pretrained_model.device)
query_am = [ 1 for _ in range (len(query_tokens))]  # am for attention mask

# this will control whether we are 
# marginalizing the next word prediction on the prompt or not
query_am_for_loss =  [ 0 for _ in range (len(query_tokens))]  # am for attention mask



# 4. encode passage
top_k_text = ["super market to buy some stuff", "ground to play foot ball", "caffe to meet my girl friend"]

top_k_tokens = tokenizer.batch_encode_plus(top_k_text, padding=True, truncation=True)['input_ids']
top_k_am = tokenizer.batch_encode_plus(top_k_text, padding=True, truncation=True)['attention_mask']

# 5. combine the query with passages for  the casual llm
prompt_tensor = torch.tensor([query_tokens + single_passage_token for 
                                       single_passage_token in top_k_tokens ]).to(model.pretrained_model.device)
prompt_am_tensor = torch.tensor([ query_am + single_passage_am for 
                                               single_passage_am in top_k_am ]).to(model.pretrained_model.device)
marginalize_am_tensor =  torch.tensor([ query_am_for_loss + single_passage_am for 
                                               single_passage_am in top_k_am ]).to(model.pretrained_model.device)

# 5. cosine score tensor
# (this could be any reward such as human feedback or output from another model)
similarity_scores = torch.rand(len(top_k_text), device=model.pretrained_model.device)

# 6. train model with ppo
marginalize_log_prob = arcee_rag_trainer.compute_marginalized_loss(
                            prompt_tensor.unsqueeze(0),
                            prompt_am_tensor.unsqueeze(0),
                            marginalize_am_tensor.unsqueeze(0),
                            similarity_scores.unsqueeze(0),
                            len(query_tokens)
)

####

# normal training