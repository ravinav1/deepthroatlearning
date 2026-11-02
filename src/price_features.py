"""Describe neighbouring prices without crossing a case's week boundary."""

import numpy as np


def neighbouring_prices(price):
    # price has one row per case and one column per hour.
    previous = np.concatenate([price[:, :1], price[:, :-1]], axis=1)
    following = np.concatenate([price[:, 1:], price[:, -1:]], axis=1)
    features = {
        "price_change_from_previous_hour": (price - previous).ravel(),
        "price_change_to_next_hour": (following - price).ravel(),
    }
    cumulative = np.concatenate([np.zeros((len(price), 1)), price.cumsum(axis=1, dtype="float64")], axis=1)
    hour = np.arange(price.shape[1])
    for window in [6, 12, 24]:
        left = np.maximum(hour - window, 0)
        right = np.minimum(hour + window + 1, price.shape[1])
        for direction, sums, counts in [
            ("previous", cumulative[:, hour] - cumulative[:, left], hour - left),
            ("next", cumulative[:, right] - cumulative[:, hour + 1], right - hour - 1),
        ]:
            # Exclude the current hour. If no neighbours exist, use its own price.
            mean = np.divide(sums, counts, out=price.astype("float64", copy=True), where=counts > 0)
            features[f"price_mean_{direction}_{window}h"] = mean.ravel()
            features[f"price_minus_{direction}_{window}h_mean"] = (price - mean).ravel()
    return features
