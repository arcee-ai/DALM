# Domain Adapted Language Modeling Toolkit

This repository primarily contains code for fine-tuning a **fully differential** Retrieval Augmented Generation (RAG-end2end) architecture. For the first time in the literature, we modified the initial RAG-end2end model ((paper)[https://aclanthology.org/2023.tacl-1.1/], (HuggingFace implementation)[https://github.com/huggingface/transformers/tree/main/examples/research_projects/rag-end2end-retriever]) to work with decoder-only language models like Llma, Falcon, or GPT. We also incorporated the **in-batch negative concept** alongside the RAG's marginalization to make the entire process **efficient**.

- Inside the Training folder, you'll find two codes to train the RAG-end2end and Retriever with contrastive learning.

- All evaluations related to the Retriever and the Generator are located in the Evaluation folder.

- Additionally, we have data processing codes and synthetic data generation code inside the Dataset folder.

# Dependencies - Installation
Ensure you have the following dependencies installed before running the code:

- Transformers
- PEFT
- Accelerate
- HNSW

## Train Retriever Only

## Train Retriever and Generator Jointly

## Arcee Domain Pretrained Models - DPT (Coming Soon)

* Arcee-DPT-PubMed-7b
* Arcee-DPT-Patent-7b
* Arcee-DPT-SEC-7b
