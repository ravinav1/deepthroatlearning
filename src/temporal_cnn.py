"""A small CNN that predicts all generators across a scheduling week."""

import numpy as np
import torch
from torch import nn

from features import HOURS, build_features


def sequence_features(cases, generators, inputs, topology, use_weekly_normalization=False,
                      use_economic_rescaling=False, **options):
    assert not (use_weekly_normalization and use_economic_rescaling), \
        "Choose either appended weekly normalization or economic rescaling."
    frame = build_features(cases, generators, inputs, topology, **options)
    # System features are shared by all generators; keep just one copy.
    split = frame.columns.get_loc("generator_id")
    shared = frame.columns[:split].tolist()
    local = frame.columns[split + 1:].tolist()
    values = frame.to_numpy().reshape(len(generators), len(cases), HOURS, -1)
    common = values[0, :, :, :split].transpose(0, 2, 1)
    per_generator = values[:, :, :, split + 1:].transpose(1, 0, 3, 2)
    per_generator = per_generator.reshape(len(cases), -1, HOURS)
    names = shared + [f"{generator}/{name}" for generator in generators for name in local]
    features = np.concatenate([common, per_generator], axis=1).copy()
    if use_economic_rescaling:
        # Express all economic inputs in the same case-local price units.
        # The log scale retains the absolute level without the raw linear extremes.
        price = features[:, names.index("price"), :].astype(np.float64)
        scale = np.maximum(np.abs(price).mean(axis=1, keepdims=True), 10.0)
        monetary = [i for i, name in enumerate(names) if "price" in name or "water_value" in name]
        features[:, monetary, :] = features[:, monetary, :] / scale[:, None, :]
        names = [f"{name}_week_scaled" if i in monetary else name for i, name in enumerate(names)]
        log_scale = np.broadcast_to(np.log(scale)[:, None, :], (len(cases), 1, HOURS)).astype(np.float32)
        features = np.concatenate([features, log_scale], axis=1)
        names.append("log_week_price_scale")
    elif use_weekly_normalization:
        # Each case's complete supplied price week defines its own economic scale.
        # Keep every original channel, including absolute prices and water values.
        price = features[:, names.index("price"), :].astype(np.float64)
        week_mean = price.mean(axis=1, keepdims=True)
        week_std = np.maximum(price.std(axis=1, keepdims=True), 1.0)
        scale = np.maximum(np.abs(price).mean(axis=1, keepdims=True), 10.0)
        normalized = [(price - week_mean) / week_std, price / scale,
                      np.broadcast_to(scale, price.shape)]
        extra_names = ["price_week_z", "price_relative", "weekly_price_scale"]
        for generator in generators:
            water_value = features[:, names.index(f"{generator}/terminal_water_value"), :]
            normalized.extend([water_value / scale, (price - water_value) / scale])
            extra_names.extend([f"{generator}/terminal_water_value_relative_to_week_price_scale",
                                f"{generator}/price_minus_water_value_relative_to_week_price_scale"])
        extra = np.stack(normalized, axis=1).astype(np.float32)
        features = np.concatenate([features, extra], axis=1)
        names += extra_names
    return features, names


class TemporalCNN(nn.Module):
    def __init__(self, channels, generators, width=64, dropout=0.1):
        super().__init__()
        self.input = nn.Conv1d(channels, width, 1)
        # Symmetric convolutions see earlier and later provided inputs.
        # Receptive field: 509 hours, covering the entire 168-hour week.
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv1d(width, width, 5, dilation=d, padding=2*d),
                          nn.ReLU(), nn.Dropout(dropout))
            for d in [1, 2, 4, 8, 16, 32, 64]
        ])
        self.output = nn.Conv1d(width, generators, 1)

    def forward(self, x):
        x = torch.relu(self.input(x))
        for block in self.blocks:
            x = x + block(x)
        return self.output(x)


def predict(model, features, device, batch_size):
    model.eval()
    with torch.no_grad():
        return np.concatenate([
            model(features[start:start + batch_size].to(device)).sigmoid().cpu().numpy()
            for start in range(0, len(features), batch_size)
        ])
