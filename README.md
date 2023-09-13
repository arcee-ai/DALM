# Domain Adapted Language Modeling Toolkit

This repository primarily contains code for fine-tuning a **fully differential** Retrieval Augmented Generation (RAG-end2end) architecture. For the first time in the literature, we modified the initial RAG-end2end model ([TACL paper](https://aclanthology.org/2023.tacl-1.1/), [HuggingFace implementation](https://github.com/huggingface/transformers/tree/main/examples/research_projects/rag-end2end-retriever)) to work with decoder-only language models like Llma, Falcon, or GPT. We also incorporated the **in-batch negative concept** alongside the RAG's marginalization to make the entire process **efficient**.

- Inside the [Training](https://github.com/arcee-ai/DALM/tree/main/Training) folder, you'll find two codes to train the RAG-end2end and Retriever with contrastive learning.

- All evaluations related to the Retriever and the Generator are located in the [Evaluation](https://github.com/arcee-ai/DALM/tree/main/Evaluation) folder.

- Additionally, we have data processing codes and synthetic data generation code inside the [Datasets](https://github.com/arcee-ai/DALM/tree/main/Datasets) folder.

# Installation

`pip install indomain`

## Local Development
Create your virtual environment and install. We suggest pyenv
```shell
python -m venv .venv && source .venv/bin/activate
pip install invoke && pyenv rehash
inv install
```

# Contributing
We use invoke to manage our codebase, and run checks on PRs,

* format the code with `inv format`
* check linting with `inv lint`
* test with `inv test`
  * test coverage must be above 95% for the PR tests to pass

## Train Retriever Only

## Train Retriever and Generator Jointly

## Arcee Domain Pretrained Models - DPT (Coming Soon)

* Arcee-DPT-PubMed-7b
* Arcee-DPT-Patent-7b
* Arcee-DPT-SEC-7b
