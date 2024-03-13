import torch.nn as nn
from model.config import TransformerConfig

class BaseTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(BaseTransformer, self).__init__()
        self.config = config

        # Embedding layers
        self.token_embeddings = nn.Embedding(
            config.vocab_size, config.d_model
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.d_model
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids):
        # Token + positional embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_ids = torch.arange(
            input_ids.size(1), device=input_ids.device
        ).unsqueeze(0)
        position_embeds = self.position_embeddings(position_ids)

        embeddings = token_embeds + position_embeds
        embeddings = self.dropout(embeddings)

        return embeddings
