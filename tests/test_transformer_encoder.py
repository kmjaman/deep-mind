import torch

import sys
sys.path.append('/path/to/project/')

from model.transformer_encoder import TransformerEncoder

def test_transformer_encoder():
    num_layers = 6
    d_model = 512
    num_heads = 8
    ff_hidden_size = 2048
    seq_len = 10
    batch_size = 2

    encoder = TransformerEncoder(num_layers, d_model, num_heads, ff_hidden_size)
    dummy_input = torch.rand(batch_size, seq_len, d_model)

    output = encoder(dummy_input)

    assert output.shape == (batch_size, seq_len, d_model)
    print("Transformer Encoder test passed!")

if __name__ == "__main__":
    test_transformer_encoder()
