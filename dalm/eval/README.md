# How to run evaluation

To run retriever only eval 
(make sure you have the checkpoints in the project root)

```bash
 python dalm/eval/eval_retriever_only.py  --dataset_path qa_paits_test.csv --retriever_model_name_or_path "BAAI/bge-large-en" --passage_column_name Abstract --query_column_name Question --retriever_peft_model_path retriever_only_checkpoints
```

For the e2e eval

```bash
python dalm/eval/eval_rag.py  --dataset_path qa_pairs_test_2.csv --retriever_model_name_or_path "BAAI/bge-large-en" --generator_model_name_or_path "meta-llama/Llama-2-7b-hf" --passage_column_name Abstract --query_column_name Question --answer_column_name Answer --evaluate_generator --query_batch_size 5 --retriever_peft_model_path retriever_only_checkpoints --generator_peft_model_path generator_only_checkpoints
```
