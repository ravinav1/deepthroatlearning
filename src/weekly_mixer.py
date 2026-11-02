"""Residual time/channel MLPs inspired by TSMixer, adapted to weekly labels."""

from torch import nn

from features import HOURS


class MixerBlock(nn.Module):
    def __init__(self, width, time_hidden, channel_hidden, dropout):
        super().__init__()
        self.time_norm = nn.LayerNorm(width)
        self.time_mlp = nn.Sequential(
            nn.Linear(HOURS, time_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(time_hidden, HOURS), nn.Dropout(dropout),
        )
        self.channel_norm = nn.LayerNorm(width)
        self.channel_mlp = nn.Sequential(
            nn.Linear(width, channel_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(channel_hidden, width), nn.Dropout(dropout),
        )

    def forward(self, x):
        # Time weights are shared across latent channels; channel weights across hours.
        time_values = self.time_norm(x).transpose(1, 2)
        x = x + self.time_mlp(time_values).transpose(1, 2)
        return x + self.channel_mlp(self.channel_norm(x))


class WeeklyMixer(nn.Module):
    def __init__(self, channels, generators, width=64, layers=4,
                 time_hidden=168, channel_hidden=128, dropout=0.1):
        super().__init__()
        self.input = nn.Linear(channels, width)
        self.blocks = nn.ModuleList([
            MixerBlock(width, time_hidden, channel_hidden, dropout)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, generators)

    def forward(self, x):
        # All 168 input hours are supplied, so each hour can use the entire week.
        x = self.input(x.transpose(1, 2))
        for block in self.blocks:
            x = block(x)
        return self.output(self.norm(x)).transpose(1, 2)
