## Fine-tuning llama-index on the arcee synthetic data

First, we need to convert the data into the format for llama-index

```shell
python prepare_data_for_llama.py
```

Then train on the dataset
```shell
python train_dataset_llama.py
```

Finally eval
```shell
python evaluate_llama_model.py
```