import sys
sys.path.append('/path/to/project/')

from data.text_dataset import TextDataset
from data.preprocess_custom_data import load_custom_data
from utils.vocabulary import Vocabulary
from utils.tokenizer import Tokenizer
from model.base_model import BaseTransformer
from model.config import TransformerConfig
from training.train import train_model
from evaluation.evaluate import evaluate_model
from utils.model_utils import save_model, load_model
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

# Load custom data
file_path = "custom_data.txt"
custom_texts = load_custom_data(file_path)

# Prepare vocabulary and tokenizer
vocab = Vocabulary()
for text in custom_texts:
    for token in text.split():
        vocab.add_token(token)

tokenizer = Tokenizer(vocab)

# Create dataset and dataloader
max_length = 20
dataset = TextDataset(custom_texts, tokenizer, max_length=max_length)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Define model and components
config = TransformerConfig(vocab_size=len(vocab))
model = BaseTransformer(config)
loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.get_index("[PAD]"))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_model(model, dataloader, optimizer, loss_fn, device, epochs=10)

# Evaluate the fine-tuned model
evaluate_model(model, dataloader, loss_fn, device)

# Save the fine-tuned model
save_model(model, "fine_tuned_transformer.pth")
