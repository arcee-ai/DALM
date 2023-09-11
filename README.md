# Domain Adapted Language Modeling Toolkit

This repository houses the code for building a dense retriever using Transformers, PEFT (Parameter-Efficient Fine-Tuning), Accelerate, and HNSW. The retriever is designed to preprocess data, generate synthetic data, train a dense retriever model, and test it using indexing.

# Dependencies - Installation
Ensure you have the following dependencies installed before running the code:

- Transformers
- PEFT
- Accelerate
- HNSW

# Contents

The repository is structured as follows:

- docs_to_passage/: Contains code for preprocessing the patent papers.
- synth_data_gen/: Includes code for generating synthetic data for training augmentation.
- contrastive_train/: Contains code to train the dense retriever model using Transformers and PEFT.
- synth_data_gen/: Includes code for testing the trained model with indexing using HNSW.

## Train Retriever Only

## Train Generator Only

## Train Retriever and Generator Jointly

## Arcee Domain Pretrained Models - DPT (Coming Soon)

* Arcee-DPT-PubMed-7b
* Arcee-DPT-Patent-7b
* Arcee-DPT-SEC-7b
