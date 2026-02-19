import pandas as pd
import torch
from torch.utils.data import Dataset


class SSTDataset(Dataset):
    """
    Stanford Sentiment Treebank V1.0
    Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank
    Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher Manning, Andrew Ng and Christopher Potts
    Conference on Empirical Methods in Natural Language Processing (EMNLP 2013)
    """

    def __init__(self, filename, maxlen, tokenizer):
        # Store the contents of the file in pandas dataframe.
        self.df = pd.read_csv(filename, delimiter="\t")
        # Initialize tokenizer for the desired transformer model.
        self.tokenizer = tokenizer
        # Maximum length of tokens list to keep all the sequences of fixed size.
        self.maxlen = maxlen

    def __len__(self):
        # Return length of dataframe.
        return len(self.df)

    def __getitem__(self, index):
        sentence = self.df.loc[index, "sentence"]
        label = int(self.df.loc[index, "label"]) + 1

        # RoBERTa 방식으로 토크나이징 (special token 자동 추가)
        encoding = self.tokenizer(
            sentence,
            max_length=self.maxlen,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        return input_ids, attention_mask, label
