import torch

import sys
sys.path.append('/path/to/project/')


from model.base_model import BaseTransformer
from model.config import TransformerConfig

def test_full_model():
    config = TransformerConfig()
    model = BaseTransformer(config)

    dummy_input = torch.randint(0, config.vocab_size, (2, 10))  # Batch size 2, Seq length 10
    output = model(dummy_input)

    assert output.shape == (2, 10, config.d_model)
    print("Full Transformer Model test passed!")

if __name__ == "__main__":
    test_full_model()
    