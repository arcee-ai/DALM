## Fine-tuning llama-index on the arcee synthetic data

First, we need to convert the data into the format for llama-index

```python
import pandas as pd

df = pd.read_csv("/root/DALM/dataset/out/question_answer_pairs_train.csv")
df["title_id"] = df.title.map(hash)
```