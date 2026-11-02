"""A small attention encoder over the complete supplied scheduling week."""

import torch
from torch import nn

from features import HOURS


class WeeklyTransformer(nn.Module):
    def __init__(self, channels, generators, width=64, heads=4, layers=4,
                 feedforward=128, dropout=0.1):
        super().__init__()
        self.input = nn.Linear(channels, width)
        self.positions = nn.Parameter(torch.empty(1, HOURS, width))
        nn.init.normal_(self.positions, std=0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                width, heads, feedforward, dropout=dropout,
                batch_first=True, norm_first=True,
            )
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, generators)

    def forward(self, x):
        x = self.input(x.transpose(1, 2)) + self.positions
        # All 168 input hours are supplied; attention can look in both directions.
        for block in self.blocks:
            x = block(x)
        return self.output(self.norm(x)).transpose(1, 2)
