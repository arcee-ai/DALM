# Domain Adapted Language Modeling Toolkit

## Manifesto

A great rift has emerged between general LLMs and the vector stores that are providing them with contextual information. The unification of these systems is an important step in grounding AI systems in efficient, factual domains, where they are utilized not only for their generality, but for their specificity and uniqueness. To this end, we are excited to open source the Arcee Domain Adapted Language Model (DALM) toolkit for developers to build on top of our Arcee open source Domain Pretrained (DPT) LLMs. We believe that our efforts will help as we begin next phase of language modeling, where organizations deeply tailor AI to operate according to their unique intellectual property and worldview. 

## Demo DALMs

Query example DALMs created by the Arcee Team.

[DALM-Patent](https://app.arcee.ai)            |  [DALM-PubMed](https://app.arcee.ai)             |  [DALM-SEC](https://app.arcee.ai)               | [DALM-Yours](https://app.arcee.ai)  
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
[![](https://i.imgur.com/Geh28Q8.jpg)](https://app.arcee.ai)  |  [![](https://i.imgur.com/IY73TcV.jpg)](https://app.arcee.ai)  |  [![](https://i.imgur.com/XgWn1VI.jpg)](https://app.arcee.ai)  |  [![](https://i.imgur.com/7KOgcEX.png)](https://app.arcee.ai)

## Research Contents

This repository primarily contains code for fine-tuning a **fully differential** Retrieval Augmented Generation (RAG-end2end) architecture. 

![E2E](https://i.imgur.com/SDoY0oq.png)

For the first time in the literature, we modified the initial RAG-end2end model ([TACL paper](https://aclanthology.org/2023.tacl-1.1/), [HuggingFace implementation](https://github.com/huggingface/transformers/tree/main/examples/research_projects/rag-end2end-retriever)) to work with decoder-only language models like Llama, Falcon, or GPT. We also incorporated the **in-batch negative concept** alongside the RAG's marginalization to make the entire process **efficient**.

- Inside the [training](https://github.com/arcee-ai/DALM/tree/main/dalm/training) folder, you'll find two codes to train the RAG-end2end and Retriever with contrastive learning.

- All evaluations related to the Retriever and the Generator are located in the [eval](https://github.com/arcee-ai/DALM/tree/main/dalm/eval) folder.

- Additionally, we have data processing codes and synthetic data generation code inside the [datasets](https://github.com/arcee-ai/DALM/tree/main/dalm/datasets) folder.

# Usage
To perform training and evaluation for both the retriever model and the new rag-e2e model, please adhere to the following steps.

## System Requirements

The system reqs depend on the retriever model, generator model, and batch size. But for reference (e2e rag), we used the following for our experiments (eval results below):
* retriever: `BAAI/bge-large-en`
* generator: `meta-llama/Llama-2-7b-hf`
* batch size: 18
* dataset size: 200k

This took 7 hours on a single A100 GPU (80GB).

## Installation

You can install this repo directly via `pip install indomain`

Alternatively, for development or research, you can clone and install the repo locally:
```shell
git clone https://github.com/arcee-ai/DALM.git && cd DALM
pip install --upgrade -e .
```
This will install the DALM repo and all necessary dependencies.

Make sure things are installed correctly by running `dalm version`.  On an non-intel Mac you may need to downgrade `transformers` library: `pip install transformers==4.30`.

## Data setup
### tl;dr
You can run `dalm qa-gen <path-to-dataset>` to preprocess your dataset for training. See `dalm qa-gen --help` for more options
<br>If you do not have a dataset, you can start with ours
```shell
# Note - our dataset already has queries and answers, so you don't actually need to run this.
# replace `toy_dataset_train.csv` with your dataset of titles and passages
dalm qa-gen dalm/datasets/toy_data_train.csv
```
- The setup for training and evaluation can be effortlessly executed provided you possess a [CSV](https://github.com/arcee-ai/DALM/tree/main/dalm/datasets/toy_data_train.csv) file containing two/three columns: `Passage`, `Query` (and `Answer` if running e2e). You can utilize the script [question_answer_generation.py](https://github.com/arcee-ai/DALM/blob/main/dalm/datasets/qa_gen/question_answer_generation.py) to generate this CSV. 
- It's important to highlight that the retriever-only training method employs solely the passages and queries, whereas the rag-e2e training code utilizes all three columns.
- In our experiments, we utilize `BAAI/bge-large-en` as the default retriever and employ `meta-llama/Llama-2-7b-hf` as the default generator. The code is designed to be compatible with any embedding model or autoregressive model available in the Hugging Face model repository at https://huggingface.co/models.

## Training

You can leverage our scripts directly if you'd like, or you can use the `dalm` cli. The arguments for both are identical

### Train Retriever Only

Train `BAAI/bge-large-en` retriever with contrastive learning.
```shell
python dalm/training/retriever_only/train_retriever_only.py \
--dataset_path "./dalm/datasets/toy_data_train.csv" \
--retriever_name_or_path "BAAI/bge-large-en" \
--output_dir "retriever_only_checkpoints" \
--use_peft \
--with_tracking \
--report_to all \
--per_device_train_batch_size 150
```
or
```shell
dalm train-retriever-only "BAAI/bge-large-en" "./dalm/datasets/toy_data_train.csv" \
--output-dir "retriever_only_checkpoints" \
--use-peft \
--with-tracking \
--report-to all \
--per-device-train-batch-size 150
```

For all available arguments and options, see `dalm train-retriever-only --help`

### Train Retriever and Generator Jointly (RAG-e2e)
Train `Llama-2-7b` generator jointly with the retriever model `BAAI/bge-large-en`.

```shell
python dalm/training/rag_e2e/train_rage2e.py \
  --dataset_path "./dalm/datasets/toy_data_train.csv" \
  --retriever_name_or_path "BAAI/bge-large-en" \
  --generator_name_or_path "meta-llama/Llama-2-7b-hf" \
  --output_dir "rag_e2e_checkpoints" \
  --with_tracking \
  --report_to all \
  --per_device_train_batch_size 20
```

or

```shell
dalm train-rag-e2e \
"./dalm/datasets/toy_data_train.csv" \
"BAAI/bge-large-en" \
"meta-llama/Llama-2-7b-hf" \
--output-dir "rag_e2e_checkpoints" \
--with-tracking \
--report-to all \
--per-device-train-batch-size 20
```

For all available arguments and options, see `dalm train-rag-e2e --help`

## Evaluation

#### Summary of how evaluation is done

The Retriever in general is trained to be good at finding the most relevant passages in a corpus given a query.

Given a ground-truth test dataset that is a 200,000-line CSV containing patent abstracts and more importantly this evaluation dataset was not present in the training dataset, the below listed steps were followed:

1. Use the trained retriever to encode all passages into an ad-hoc indexed vector store using the HNSW library.
2. Take each query and use the trained retriever to encode it into an embedding vector (QE)
3. For each encoded passage (PE) in the vector store, find the nearest neighbor similarity search score between QE and PE (**Note**: with HNSW, exhaustiveness is avoided)
4. Find the top-K (eg, top 5) best matches based on nearest neighbor similarity search scores
5. Compare the matches against the ground truth top-K best matches to calculate `recall` and `hit rate`.

#### Results

| Type of Retriever | Recall | Hit rate |
| --- | ----- | ----|
| Plain Retriever | 0.45984 | 0.45984 |
| Retriever with contrastive learning | 0.46037 | 0.46038 |
| Retriever End2End | 0.73634 | 0.73634 |

To run retriever only eval 
(make sure you have the checkpoints in the project root)

```bash
 python dalm/eval/eval_retriever_only.py  \
 --dataset_path qa_pairs_test.csv \
 --retriever_name_or_path "BAAI/bge-large-en" \
 --passage_column_name Abstract \
 --query_column_name Question \
 --retriever_peft_model_path retriever_only_checkpoints
```
or 
```bash
dalm eval-retriever qa_pairs_test.csv \
 --retriever-name-or-path "BAAI/bge-large-en" \
 --passage-column-name Abstract \
 --query-column-name Question \
 --retriever-peft-model-path retriever_only_checkpoints
```
See `dalm eval-retriever --help` for all available arguments

For the e2e eval

```bash
python dalm/eval/eval_rag.py  \
 --dataset_path qa_pairs_test_2.csv \
 --retriever_name_or_path "BAAI/bge-large-en" \
 --generator_name_or_path "meta-llama/Llama-2-7b-hf" \
 --passage_column_name Abstract \
 --query_column_name Question \
 --answer_column_name Answer \
 --evaluate_generator \
 --query_batch_size 5 \
 --retriever_peft_model_path rag_e2e_checkpoints/retriever \
 --generator_peft_model_path rag_e2e_checkpoints/generator
```
or
```bash
dalm eval-rag qa_pairs_test.csv \
 --retriever-name-or-path "BAAI/bge-large-en" \
 --generator-name-or-path "meta-llama/Llama-2-7b-hf" \
 --retriever-peft-model-path rag_e2e_checkpoints/retriever \
 --generator-peft-model-path rag_e2e_checkpoints/generator \
 --passage-column-name Abstract \
 --query-column-name Question \
 --answer-column-name Answer \
 --query-batch-size 5
```
See `dalm eval-rag --help` for all available arguments


## Contributing
See [CONTRIBUTING](https://github.com/arcee-ai/DALM/tree/main/CONTRIBUTING.md)
