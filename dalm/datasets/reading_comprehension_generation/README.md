## What this is 



## A note about reading comprehension



We have two ways of generating  reading comprehension data

1. Via regex based methods that combs the input data for match and aligns them into questions and answers
2. Via prompting a large language model to come up with questions and answers 

## How to get started

For the input, either a single csv file or a 

### LLM based

Assuming you have your dataset as a csv file with the column `text` containing the raw texts

(chunking based on context length of model is enabled by default) 

```bash

python dalm/datasets/reading_comprehension_generation/synthetic_based.py \
            --model HuggingFaceH4/zephyr-7b-alpha \
            --input input_dataset.csv  --output_directory synth_data --dataset_name llm_generated_dataset

```


### Regex based

For this please ensure you have a general sentencepiece model.

Please note there is the choice of passing in a domain sentence model in addition, but this is not required as
the script will train a domain speicifc sentencepiece model on the input corpus

```bash


```





