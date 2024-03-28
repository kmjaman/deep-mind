import torch

import sys
sys.path.append('/path/to/project/')

from model.transformer_block import TransformerBlock

def test_transformer_block():
    d_model = 512
    num_heads = 8
    ff_hidden_size = 2048
    seq_len = 10
    batch_size = 2

    transformer_block = TransformerBlock(d_model, num_heads, ff_hidden_size)
    dummy_input = torch.rand(batch_size, seq_len, d_model)

    output = transformer_block(dummy_input)

    assert output.shape == (batch_size, seq_len, d_model)
    print("Transformer Block test passed!")

if __name__ == "__main__":
    test_transformer_block()