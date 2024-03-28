import torch.nn as nn
from model.attention import MultiHeadAttention
from model.feedforward import FeedForwardNetwork

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_hidden_size, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, ff_hidden_size, dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Self-attention layer
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)  # Add residual connection

        # Feedforward layer
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)  # Add residual connection

        return x