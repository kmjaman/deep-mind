import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.append('/path/to/project/')

from model.base_model import BaseTransformer
from model.config import TransformerConfig
from utils.vocabulary import Vocabulary
from utils.tokenizer import Tokenizer
from data.text_dataset import TextDataset
from training.train import train_model

# Prepare data
texts = [
    "hello world this is a test",
    "this is another example",
    "how are you doing today",
    "let's build a transformer model"
]
vocab = Vocabulary()
for text in texts:
    for token in text.split():
        vocab.add_token(token)

tokenizer = Tokenizer(vocab)
dataset = TextDataset(texts, tokenizer, max_length=10)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Model and training components
config = TransformerConfig(vocab_size=len(vocab))
model = BaseTransformer(config)
loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.get_index("[PAD]"))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_model(model, dataloader, optimizer, loss_fn, device, epochs=5)
