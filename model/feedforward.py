import torch.nn as nn

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, ff_hidden_size, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(d_model, ff_hidden_size)
        self.fc2 = nn.Linear(ff_hidden_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)
    