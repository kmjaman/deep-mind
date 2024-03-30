import torch
import torch.nn as nn

from model.transformer_encoder import TransformerEncoder

class BaseTransformer(nn.Module):
    def __init__(self, config):
        super(BaseTransformer, self).__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.d_model)
        self.encoder = TransformerEncoder(
            num_layers=config.num_layers,
            d_model=config.d_model,
            num_heads=config.num_heads,
            ff_hidden_size=config.ff_hidden_size,
            dropout=config.dropout
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids):
        token_embeds = self.token_embeddings(input_ids)
        position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embeddings(position_ids)

        x = self.dropout(token_embeds + position_embeds)
        return self.encoder(x)
