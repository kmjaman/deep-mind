import torch

import sys
sys.path.append('/path/to/project/')

from model.attention import MultiHeadAttention

def test_multihead_attention():
    d_model = 512
    num_heads = 8
    seq_len = 10
    batch_size = 2

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    dummy_input = torch.rand(batch_size, seq_len, d_model)

    output, weights = mha(dummy_input, dummy_input, dummy_input)

    assert output.shape == (batch_size, seq_len, d_model)
    print("Multi-Head Attention test passed!")

if __name__ == "__main__":
    test_multihead_attention()