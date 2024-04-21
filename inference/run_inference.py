import torch

import sys
sys.path.append('/path/to/project/')


from model.base_model import BaseTransformer
from model.config import TransformerConfig
from utils.vocabulary import Vocabulary
from utils.tokenizer import Tokenizer
from utils.model_utils import load_model
from inference.text_generation import generate_text

# Prepare vocabulary and tokenizer
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

# Load model
config = TransformerConfig(vocab_size=len(vocab))
model = BaseTransformer(config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(model, "transformer_model.pth", device)

# Generate text
input_text = "hello world"
generated_text = generate_text(model, tokenizer, input_text, max_length=20, device=device)

print(f"Input: {input_text}")
print(f"Generated: {generated_text}")
