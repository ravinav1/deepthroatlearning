"""Read the supplied scheduling week forwards and backwards."""

from torch import nn


class BidirectionalLSTM(nn.Module):
    def __init__(self, channels, generators, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            channels, hidden_size, num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.output = nn.Linear(2 * hidden_size, generators)

    def forward(self, x):
        # Shared training code uses case -> feature -> hour.
        context, _ = self.lstm(x.transpose(1, 2))
        return self.output(context).transpose(1, 2)
