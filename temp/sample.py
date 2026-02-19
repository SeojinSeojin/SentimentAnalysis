import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("train.tsv", delimiter="\t")

sample = df.sample(20)
for _, row in sample.iterrows():
    print(f"Label: {row['label']}")
    print(f"Text: {row['sentence'][:200]}")
    print("---")