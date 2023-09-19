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

For the first time in the literature, we modified the initial RAG-end2end model ([TACL paper](https://aclanthology.org/2023.tacl-1.1/), [HuggingFace implementation](https://github.com/huggingface/transformers/tree/main/examples/research_projects/rag-end2end-retriever)) to work with decoder-only language models like Llma, Falcon, or GPT. We also incorporated the **in-batch negative concept** alongside the RAG's marginalization to make the entire process **efficient**.

- Inside the [Training](https://github.com/arcee-ai/DALM/tree/main/Training) folder, you'll find two codes to train the RAG-end2end and Retriever with contrastive learning.

- All evaluations related to the Retriever and the Generator are located in the [Evaluation](https://github.com/arcee-ai/DALM/tree/main/Evaluation) folder.

- Additionally, we have data processing codes and synthetic data generation code inside the [Datasets](https://github.com/arcee-ai/DALM/tree/main/Datasets) folder.

## Code execution
To perform training and evaluation for both the retriever model and the new rag-e2e model, please adhere to the following steps. 

- The setup for training and evaluation can be effortlessly executed provided you possess a [CSV](https://github.com/arcee-ai/DALM/tree/main/dalm/datasets/toy_dataset_train.py) file containing three columns: passage, query, and answer. You can utilize the script [question_answer_generation.py](https://github.com/arcee-ai/DALM/blob/main/dalm/datasets/qa_gen/question_answer_generation.py) to generate this CSV. 
- It's important to highlight that the retriever-only training method employs solely the passages and queries, whereas the rag-e2e training code utilizes all three columns.
- In our experiments, we utilize BAAI/bge-large-en as the retriever and employ meta-llama/Llama-2-7b-hf as the generator. It's important to note that this code is designed to be compatible with any embedding model or autoregressive model available in the Hugging Face model repository at https://huggingface.co/models.

## Clone the repositary
- `git clone https://github.com/arcee-ai/DALM.git`
-  `cd DALM`

## Install the necesarry libraries
Create your desired virtual environment isntall all necasary librries.
- `pip install -r requirements.txt`

## Training

### Train Retriever Only

Train `BAAI/bge-large-en` retriever with contrastive learning.

```
python dalm/training/retriever_only/train_retriever_only.py 
--train_dataset_csv_path ./dalm/datasets/toy_data_train.csv" \
--model_name_or_path "BAAI/bge-large-en" \
--output_dir "./dalm/training/rag_e2e/retriever_only_checkpoints" \
--use_peft \
--with_tracking \
--report_to all \
--per_device_train_batch_size 150
```

### Train Retriever and Generator Jointly (RAG-e2e)

Train `Llama-2-7b` generator jointly with the retriever model `BAAI/bge-large-en`.

```
python dalm/training/rag_e2e/train_rage2e.py \
  --dataset_path "./dalm/datasets/toy_data_train.csv" \
  --retriever_name_or_path "BAAI/bge-large-en" \
  --generator_name_or_path "meta-llama/Llama-2-7b-hf" \
  --output_dir "./dalm/training/rag_e2e/rag_e2e_checkpoints" \
  --with_tracking \
  --report_to all \
  --per_device_train_batch_size 24
```
## Evaluation

### Evaluate the top-k recall of the retriver that trained only with constrstive learning


### Evaluate the top-k recall of the  retriever and the exact match of the generator in the RAG-e2e models