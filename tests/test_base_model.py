import torch
import sys
sys.path.append('/path/to/project/')

from model.config import TransformerConfig
from model.base_model import BaseTransformer


def test_embeddings():
    config = TransformerConfig()
    model = BaseTransformer(config)

    dummy_input = torch.randint(0, config.vocab_size, (2, 10))  # Batch of 2, Seq Length 10
    output = model(dummy_input)

    assert output.shape == (2, 10, config.d_model)
    print("Embeddings test passed!")

if __name__ == "__main__":
    test_embeddings()
