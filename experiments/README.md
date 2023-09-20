                               ```
pip install .
cd expeirments
pip install -r requirements.txt
python data_gen.py
mkdir qa-outputs
dalm qa-gen train_data.csv --output-dir qa-outputs --passage-column-name text --title-column-name title --sample-size 1000000
# had to concat the train and test csvs....

dalm train-rag-e2e \
"qa-outputs/question_answer_pairs.csv" \
"BAAI/bge-large-en" \
"meta-llama/Llama-2-7b-hf" \
--dataset-passage-col-name text \
--output-dir "rag_e2e_checkpoints" \
--no-with-tracking \
--per-device-train-batch-size 12

  
dalm qa-gen val_data.csv --output-dir qa-outputs --passage-column-name text --title-column-name title --sample-size 100000
# had to combine those 2 csv files...
python ../dalm/eval/eval_retriever_only.py  --dataset_path qa-outputs/question_answer_pairs_test.csv --retriever_model_name_or_path "BAAI/bge-large-en" --passage_column_name text --query_column_name Question --retriever_peft_model_path rag_e2e_checkpoints/retriever
###
Retriever results:
Recall: 0.8578767123287672
Precision: 0.08578767123287746
Hit Rate: 0.8578767123287672
*************
###


----- retriever only


dalm train-retriever-only "BAAI/bge-large-en" "qa-outputs/question_answer_pairs.csv" \
--output-dir "retriever_only_checkpoints" \
--use-peft \
--dataset-passage-col-name text \
--per-device-train-batch-size 150

```