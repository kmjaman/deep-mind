class TransformerConfig:
    def __init__(
        self,
        vocab_size=30522,
        max_position_embeddings=512,
        d_model=512,
        num_heads=8,
        num_layers=6,
        ff_hidden_size=2048,
        dropout=0.1,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_hidden_size = ff_hidden_size
        self.dropout = dropout
