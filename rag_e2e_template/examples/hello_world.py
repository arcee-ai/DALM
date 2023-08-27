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
query_txt = ["##query## Who is the largest mammal on Earth ?"]

tokenized_query = tokenizer.batch_encode_plus(query_txt, padding=True, truncation=True)
query_tokens = tokenized_query['input_ids'][0]
query_am = tokenized_query['attention_mask'][0]

# this will control whether we are 
# marginalizing the next word prediction on the prompt or not
query_am_for_loss =  [ 0 for _ in range (len(query_tokens))]  # am for attention mask

# 4. encode passage
top_k_text = ["##passaga## The blue whale is the largest mammal to have ever existed."]

# 5. answer
answer = ["##answer## The answer is Blue whale"]

# 6. prompt tokenization
tokenized_prompt = tokenizer.batch_encode_plus([top_k_text+answer], padding=True, truncation=True)
prompt_tokens = tokenized_prompt['input_ids'][0]
prompt_am = tokenized_prompt['attention_mask'][0]


# 7. answer tokenization
tokenized_answer = tokenizer.batch_encode_plus(answer, padding=True, truncation=True)
answer_k_tokens = tokenized_answer['input_ids'][0]
answer_k_am = tokenized_answer['attention_mask'][0]


# 8. combine the prompt with the answer for  casual llm training
input_tensor = torch.tensor([prompt_tokens + answer_k_tokens ]).to(model.pretrained_model.device)
iput_am_tensor = torch.tensor([ prompt_am + answer_k_am ]).to(model.pretrained_model.device)

# 9. cosine score tensor
# (this could be any reward such as human feedback or output from another model)
similarity_scores = torch.rand(5, device=model.pretrained_model.device)


# 6. train model with ppo
marginalize_casual_loss = arcee_rag_trainer.compute_marginalized_loss(
                            input_tensor,
                            iput_am_tensor,
                            similarity_scores.unsqueeze(0),
                            len(prompt_tokens)
)


exit()

####

# normal training

# marginalize_casual_loss + retriever's in batch negative loss