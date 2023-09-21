# Testing e2e retriever hit rate


Inspired by [llama-index](https://gpt-index.readthedocs.io/en/latest/examples/finetuning/embeddings/finetune_embedding_adapter.html)

# Setup

```shell
pip install indomain
pip install -r requirements.txt
python data_gen.py
mkdir qa-outputs
```

## Create the datasets
First, create the train dataset
```shell
dalm qa-gen train_data.csv --output-dir qa-outputs --passage-column-name text --title-column-name title --sample-size 1000000
```
This creates a train and test file (because we typically want to split), so merge those into 1
```shell
head -n 1 qa-outputs/question_answer_pairs_train.csv > question_answer_pairs.csv && tail -n+2 -q qa-outputs/*.csv >> question_answer_pairs.csv
rm qa-outputs/*.csv
mv question_answer_pairs.csv qa-outputs
```

Same for the validation data
```shell
dalm qa-gen val_data.csv --output-dir qa-outputs-test --passage-column-name text --title-column-name title --sample-size 100000
head -n 1 qa-outputs-test/question_answer_pairs_train.csv > question_answer_pairs_test.csv && tail -n+2 -q qa-outputs-test/*.csv >> question_answer_pairs_test.csv
rm -rf qa-outputs-test
mv question_answer_pairs_test.csv qa-outputs
```

Now we have 2 files for training and eval
```shell
(.venv) root@f4ec1ae23983:# ls -lash qa-outputs/
total 2.3M
1.4M -rw-r--r-- 1 root root 1.4M Sep 20 20:02 question_answer_pairs.csv
956K -rw-r--r-- 1 root root 953K Sep 20 20:14 question_answer_pairs_test.csv
```

## Rage2e training

Then we train e2e
```shell
dalm train-rag-e2e \
"qa-outputs/question_answer_pairs.csv" \
"BAAI/bge-small-en" \
"meta-llama/Llama-2-7b-hf" \
--dataset-passage-col-name text \
--output-dir "rag_e2e_checkpoints_bgsmall" \
--no-with-tracking \
--per-device-train-batch-size 12
```

And eval
```
python ../../dalm/eval/eval_retriever_only.py  --dataset_path qa-outputs/question_answer_pairs_test.csv --retriever_name_or_path "BAAI/bge-small-en" --passage_column_name text --query_column_name Question --retriever_peft_model_path rag_e2e_checkpoints_bgsmall/retriever --embed_dim 384

*************
Retriever results:
Recall: 0.8202054794520548
Hit Rate: 0.8202054794520548
*************
```



## Retriever only training

Train the retriever only
```
dalm train-retriever-only "BAAI/bge-small-en" "qa-outputs/question_answer_pairs.csv" \
--output-dir "retriever_only_checkpoints_bgsmall" \
--use-peft \
--dataset-passage-col-name text \
--per-device-train-batch-size 150
```

and eval
```
python ../../dalm/eval/eval_retriever_only.py  --dataset_path qa-outputs/question_answer_pairs_test.csv --retriever_name_or_path "BAAI/bge-small-en" --passage_column_name text --query_column_name Question --retriever_peft_model_path retriever_only_checkpoints_bgsmall/ --embed_dim 384

*************
Retriever results:
Recall: 0.8116438356164384
Precision: 0.08116438356164453
Hit Rate: 0.8116438356164384
*************
```