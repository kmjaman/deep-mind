import torch

import sys
sys.path.append('/path/to/project/')

from data.preprocess_custom_data import load_custom_data
from inference.text_generation import generate_text
from utils.model_utils import load_model
from model.base_model import BaseTransformer
from model.config import TransformerConfig
from utils.tokenizer import Tokenizer
from utils.vocabulary import Vocabulary


# Load custom data
file_path = "custom_data.txt"
custom_texts = load_custom_data(file_path)

# Prepare vocabulary and tokenizer
vocab = Vocabulary()
for text in custom_texts:
    for token in text.split():
        vocab.add_token(token)

tokenizer = Tokenizer(vocab)


config = TransformerConfig(vocab_size=len(vocab))
# Load fine-tuned model
fine_tuned_model = BaseTransformer(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fine_tuned_model = load_model(fine_tuned_model, "fine_tuned_transformer.pth", device)


# Generate text
input_text = "A journey of a"
generated_text = generate_text(fine_tuned_model, tokenizer, input_text, max_length=20, device=device)

print(f"Input: {input_text}")
print(f"Generated: {generated_text}")
