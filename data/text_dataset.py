import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        token_ids = self.tokenizer.encode(text)[:self.max_length]
        padding_length = self.max_length - len(token_ids)

        # Input and target sequences
        input_ids = token_ids[:-1] + [self.tokenizer.vocabulary.get_index("[PAD]")] * padding_length
        target_ids = token_ids[1:] + [self.tokenizer.vocabulary.get_index("[PAD]")] * padding_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long)
        }
