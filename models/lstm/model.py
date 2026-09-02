import torch
import torch.nn as nn

class LSTMPricePredictor(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size

        self.num_layers = config.num_layers

        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            batch_first=True,
        )

        self.dropout = nn.Dropout(config.dropout)

        self.output = nn.Linear(
            config.hidden_size,
            config.output_size
        )

    def forward(self, x):
        batch = x.size(0)

        h0 = torch.zeros(
            self.num_layers,
            batch,
            self.hidden_size,
            device=x.device
        )

        c0 = torch.zeros(
            self.num_layers,
            batch,
            self.hidden_size,
            device=x.device
        )

        out, _ = self.lstm(x, (h0, c0))

        out = out[:, -1, :]

        out = self.dropout(out)

        out = self.output(out)

        return out.squeeze(-1)