# Fine-tuning llama-index on the arcee synthetic data

## Important note about our evaluation on the llama-index model
The llama-index evaluation process was _extremely_ slow. On our A100 80GB, evaluating all
200,000 rows would have taken 400+hours. So (as seen in the `evaluate_llama_model.py` script), we evaluate the first
2k rows (for 4+ hours of eval). 

We did **not** subsample 2k records and run eval on that. We embedded the entire corpus, to make the 
retrieval likelihood equivalent to the 200k dataset (ie, with only 2k embedding, the odds you'd pick
the correct record in the top-10 choices is much higher than if you have 200k embeddings). 

So, we embedded all 200k records (~1 hour), then evaled the first 2k records (~4 hours). If you'd like to 
run the eval on 200k records, we'd love that PR! But we feel this is representative of the performance on the full 
dataset. And as the results show, it is in line with our retriever-only results.

**Note: Since the model checkpoints are saved here, you can skip straight to the eval**
```shell
python evaluate_llama_model.py
```

## e2e
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

## Results!

llama-index results:
```markdown
  retrievers  hit_rate       mrr
0         ft    0.4915  0.353698
```

Now, compared to the results of our training process. First 3 are the same as in our README, they are our results. The final row is
the result of llama-index fine-tuning, and their own hit-rate eval

| Type of Retriever | Recall | Hit rate |
| --- | ----- | ----|
| Plain Retriever | 0.45984 | 0.45984 |
| Retriever with contrastive learning | 0.46037 | 0.46038 |
| Retriever End2End | 0.73634 | 0.73634 |
| Llama-index fine-tune retriever | N/A | 0.4915 |

It seems as though, while marginally better than our retriever-only training, it is 
significantly worse than our end-to-end training process!
