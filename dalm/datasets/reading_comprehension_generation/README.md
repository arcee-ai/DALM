## A note about reading comprehension

This aproach of adapting LLMs is based on this [paper](https://arxiv.org/abs/2309.09530) by Microsoft 
The way espoused by the paper is generating reading comprehension questions and answers based on the raw corpora
and training a llm on said generated dataset can enhance its domain adaptiveness

We have two ways of generating reading comprehension data

1. Via regex based methods that combs the input data for match and aligns them into questions and answers
2. Via prompting a large language model to come up with questions and answers 

To see the prompt behind LLM based reading-comprehension dataset generation please go [here](https://github.com/arcee-ai/DALM/blob/4d93d4a198cc64ce5d19ee98786b70f579dbef0c/dalm/datasets/reading_comprehension_generation/synthetic_based.py#L22)

## How to get started

For the input, either a single csv file or a directory of individual files each containing raw text will do.


### LLM based

Assuming you have your dataset as a csv file with the column `text` containing the raw texts

(chunking based on context length of model is enabled by default) 

```bash
python dalm/datasets/reading_comprehension_generation/synthetic_based.py \
            --model HuggingFaceH4/zephyr-7b-alpha \
            --context-length  4192
            --input input_dataset.csv  --output_directory synth_data --dataset_name llm_generated_dataset
```

the output directory serves as a temporary holding place of all generated data before it can be made a dataset.
The generation process is time consuming and expensive. On average, because the process uses an LLM (if using the recommended 13b llama2 model), it take about 20-30 minutes to produce 10 questions (numbers may vary depending on the content of your dataset and the unpredictability of the model). So every step is taken to ensure that if the process is interrupted, once back running will pick up where it left off. 

Chunking of data is enabled by default and requires the context length to be passed  which is why it passed in in the example

### Regex based

(Same, as above i.e assuming you have your dataset as a csv file with the column `text` containing the raw texts)

Please note there is the choice of passing in a domain sentence model in addition, but this is not required as
the script will train a domain specific sentencepiece model on the input corpus

```bash

python dalm/datasets/reading_comprehension_generation/regex_based.py  --input input.csv \
            --csv_column text --general_spm_path resources/general.spm  \
            --output_dataset_name regex_dataset
```