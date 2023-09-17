# Domain Adapted Language Modeling Toolkit

## Manifesto

A great rift has emerged between general LLMs and the vector stores that are providing them with contextual information. The unification of these systems is an important step in grounding AI systems in efficient, factual domains, where they are utilized not only for their generality, but for their specificity and uniqueness. To this end, we are excited to open source the Arcee Domain Adapted Language Model (DALM) toolkit for developers to build on top of our Arcee open source Domain Pretrained (DPT) LLMs. We believe that our efforts will help usher in the next phase of language modeling, where organization's deeply tailor AI to operate according to their unique intellectual property and worldview. 

## Research Progress

This repository primarily contains code for fine-tuning a **fully differential** Retrieval Augmented Generation (RAG-end2end) architecture. 

![E2E](https://i.imgur.com/SDoY0oq.png)

For the first time in the literature, we modified the initial RAG-end2end model ([TACL paper](https://aclanthology.org/2023.tacl-1.1/), [HuggingFace implementation](https://github.com/huggingface/transformers/tree/main/examples/research_projects/rag-end2end-retriever)) to work with decoder-only language models like Llma, Falcon, or GPT. We also incorporated the **in-batch negative concept** alongside the RAG's marginalization to make the entire process **efficient**.

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
```
python contrastive_train/peft_lora_constrastive_learning.py  --train_dataset_csv_path "xxxx.csv" \
    --test_dataset_csv_path "yyyy.csv" \
    --model_name_or_path "BAAI/bge-small-en" --output_dir "./retriever_only_checkpoints" --use_peft  \
    --with_tracking --report_to all --per_device_train_batch_size 30
```

## Train Retriever and Generator Jointly
```
python dalm/training/rag_e2e/train_rage2e.py
  --dataset_path "/root/DALM/dataset/out/question_answer_pairs_train"
  --retriever_name_or_path "BAAI/bge-large-en"
  --generator_name_or_path "meta-llama/Llama-2-7b-hf"
  --output_dir "./rag_e2e_checkpoints"
  --with_tracking
  --report_to all
  --per_device_train_batch_size 32
```

## Domain Pretrained Models - DPT (Coming Soon)

![DALM](https://i.imgur.com/rqW405I.png)

* DPT-PubMed-7b
* DPT-Patent-7b
* DPT-SEC-7b
