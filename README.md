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

# Usage
To perform training and evaluation for both the retriever model and the new rag-e2e model, please adhere to the following steps.

## Installation

You can install this repo directly via `pip install indomain`

Alternatively, for development, you can clone and install the repo locally. From the root directory `DALM`:
```shell
git clone https://github.com/arcee-ai/DALM.git && cd DALM
python -m venv .venv && source .venv/bin/activate
pip install invoke
inv install
```
This will install the DALM repo and all necessary dependencies.

Make sure things are installed correctly by running `dalm version`

## Data setup
### tl;dr
You can run `dalm qa-gen <path-to-dataset>` to preprocess your dataset for training. See `dalm qa-gen --help` for more options
<br>If you do not have a dataset, you can start with ours
```shell
dalm qa-gen dalm/datasets/toy_data_train.csv
```
- The setup for training and evaluation can be effortlessly executed provided you possess a [CSV](https://github.com/arcee-ai/DALM/tree/main/dalm/datasets/toy_data_train.csv) file containing two/three columns: `Passage`, `Query` (and `Answer` if running e2e). You can utilize the script [question_answer_generation.py](https://github.com/arcee-ai/DALM/blob/main/dalm/datasets/qa_gen/question_answer_generation.py) to generate this CSV. 
- It's important to highlight that the retriever-only training method employs solely the passages and queries, whereas the rag-e2e training code utilizes all three columns.
- In our experiments, we utilize `BAAI/bge-large-en` as the default retriever and employ `meta-llama/Llama-2-7b-hf` as the default generator. The code is designed to be compatible with any embedding model or autoregressive model available in the Hugging Face model repository at https://huggingface.co/models.


## Training

You can leverage our scripts directly if you'd like, or you can use the `dalm` cli. The arguments for both are identical

### Train Retriever Only
```shell
dalm train-retriever-only "BAAI/bge-large-en" "./dalm/datasets/toy_data_train.csv" \
--output-dir "./dalm/training/rag_e2e/retriever_only_checkpoints" \
--use-peft \
--with-tracking \
--report-to all \
--per-device-train-batch-size 150
```
or
```shell
python dalm/training/retriever_only/train_retriever_only.py \
--train_dataset_csv_path "./dalm/datasets/toy_data_train.csv" \
--model_name_or_path "BAAI/bge-large-en" \
--output_dir "./dalm/training/rag_e2e/retriever_only_checkpoints" \
--use_peft \
--with_tracking \
--report_to all \
--per_device_train_batch_size 150
```

For all available arguments and options, see `dalm train-retriever-only --help`

### Train Retriever and Generator Jointly (RAG-e2e)
```shell
dalm train-rag-e2e \
"./dalm/datasets/toy_data_train.csv" \
"BAAI/bge-large-en" \
"meta-llama/Llama-2-7b-hf" \
--output-dir "./dalm/training/rag_e2e/rag_e2e_checkpoints" \
--with-tracking \
--report-to all \
--per-device-train-batch-size 24
```
or
```shell
python dalm/training/rag_e2e/train_rage2e.py \
  --dataset_path "./dalm/datasets/toy_data_train.csv" \
  --retriever_name_or_path "BAAI/bge-large-en" \
  --generator_name_or_path "meta-llama/Llama-2-7b-hf" \
  --output_dir "./dalm/training/rag_e2e/rag_e2e_checkpoints" \
  --with_tracking \
  --report_to all \
  --per_device_train_batch_size 24
```

For all available arguments and options, see `dalm train-rag-e2e --help`

## Evaluation

### Evaluate the top-k recall of the retriever that trained only with contrastive learning


### Evaluate the top-k recall of the  retriever and the exact match of the generator in the RAG-e2e models


## Domain Pretrained Models - DPT (Coming Soon)

![DALM](https://i.imgur.com/rqW405I.png)

* DPT-PubMed-7b
* DPT-Patent-7b
* DPT-SEC-7b
