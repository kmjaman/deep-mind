import torch

import sys
sys.path.append('/path/to/project/')

from model.base_model import BaseTransformer
from model.config import TransformerConfig
from utils.vocabulary import Vocabulary
from utils.tokenizer import Tokenizer

def test_full_model_with_tokenizer():
    # Set up vocabulary and tokenizer
    vocab = Vocabulary()
    for token in ["hello", "world", "this", "is", "a", "test"]:
        vocab.add_token(token)

    tokenizer = Tokenizer(vocab)

    # Prepare input
    text = "hello world! this is a test."
    token_ids = tokenizer.encode(text)
    token_ids = torch.tensor([token_ids])  # Batch of 1

    # Initialize model
    config = TransformerConfig(vocab_size=len(vocab))
    model = BaseTransformer(config)

    # Forward pass
    output = model(token_ids)

    assert output.shape == (1, len(token_ids[0]), config.d_model)
    print("Full model with tokenizer test passed!")

if __name__ == "__main__":
    test_full_model_with_tokenizer()
