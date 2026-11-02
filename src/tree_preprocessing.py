"""Optional case-relative economic units for the tree models."""

import numpy as np

from features import HOURS


def rescale_tree_features(features):
    # Rows are generator -> case -> hour; each consecutive week supplies its scale.
    price = features["price"].to_numpy(dtype="float64").reshape(-1, HOURS)
    week_scale = np.maximum(np.abs(price).mean(axis=1), 10.0)
    row_scale = np.repeat(week_scale, HOURS)
    monetary = [name for name in features if "price" in name or "water_value" in name]
    result = features.copy()
    result[monetary] = (features[monetary].to_numpy() / row_scale[:, None]).astype("float32")
    result = result.rename(columns={name: f"{name}_week_scaled" for name in monetary})
    result["log_week_price_scale"] = np.log(row_scale).astype("float32")
    return result
