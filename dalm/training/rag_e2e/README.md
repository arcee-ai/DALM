# End2End Differentiable RAG Repository

This repository contains an implementation of a fully differentiable RAG that works seamlessly with dense retrievers and casual language models (LLMs) as generators. The key innovation of this model is the use of a casual language model to marginalize the next word prediction probability, taking into account both the query and passage selection probability. To ensure efficient engineering, we employ in-batch negatives to facilitate both the retriever and the generator.

# Prerequisites

Before you can execute the code, please make sure you have the following components:

- **train_rag_e2e.py**: This script contains the training code and associated arguments. We utilize Hugging Face's Accelerated Training API for efficiency.

- **base_model.py**: This file defines the base model class, which initializes both the retriever and the generator components of the RAG model.

- **train_utils.py**: Inside this file, you will find helper functions that are used to compute loss functions and other utility functions for training.

# Getting Started
- To execute the code and get started with the End2End Differentiable RAG, follow these steps:

- Clone this repository to your local machine.

- Make sure you have a mock dataset available. You can specify the dataset path when running the script.

- Execute the following script with your desired configuration:

To train the model using a smaller retriever, and gpt2, you can run


```shell
dalm train-rag-e2e \
"./dataset" \
"BAAI/bge-small-en" \
"gpt2"  \
--output-dir "./rag_e2e_checkpoints" \
--with-tracking \
--report-to tensorboard 
```
or, to execute directly,
```shell
python train_rag_e2e.py --dataset_path "./dataset" \
                       --retriever_name_or_path "BAAI/bge-small-en" \
                       --generator_name_or_path "gpt2" \
                       --output_dir "./rag_e2e_checkpoints" \
                       --with_tracking \
                       --report_to tensorboard
```

This script will start training the End2End Differentiable RAG model using the specified dataset and model configurations.

