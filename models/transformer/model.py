import torch.nn as nn
from .positional_encoding import PositionalEncoding

class TransformerPricePredictor(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding = nn.Linear(
            config.input_size,
            config.d_model
        )

        self.position = PositionalEncoding(
            config.d_model
        )

        encoder = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dropout=config.dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder,
            num_layers=config.num_layers
        )

        self.fc = nn.Linear(
            config.d_model,
            1
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.position(x)
        x = self.encoder(x)
        x = x[:, -1]
        return self.fc(x)