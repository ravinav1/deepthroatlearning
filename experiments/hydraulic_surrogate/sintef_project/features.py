"""Case-hour features using only the documented hydropower optimizer inputs.

Every case has 168 rows. Cases are ordered as the training CSV followed by the
prediction mapping; hour increases fastest. Initial storage is read at case
start and terminal water value at case start plus seven days. Neither future
storage nor any commitment labels enter the features.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd


HORIZON = 168


def _read_timeseries(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.index = pd.to_datetime(frame.pop("date"), utc=True)
    if frame.index.has_duplicates:
        raise ValueError(f"Duplicate timestamps in {path}")
    return frame.astype(np.float32)


def _lookup(frame: pd.DataFrame, dates: pd.DatetimeIndex, name: str) -> np.ndarray:
    positions = frame.index.get_indexer(dates)
    if (positions < 0).any():
        raise ValueError(f"Missing {name} timestamps: {dates[positions < 0][:5]}")
    values = frame.to_numpy()[positions]
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite values in {name}")
    return values


def _capacities(path: Path) -> dict[str, float]:
    """Read simple reservoir max_vol entries without requiring a YAML package."""
    result = {}
    current = None
    in_reservoir = False
    for line in path.read_text().splitlines():
        if line == "  reservoir:":
            in_reservoir = True
            continue
        if in_reservoir and re.match(r"^  \S", line):
            break
        if in_reservoir and re.match(r"^    \S", line):
            current = line.strip().removesuffix(":")
        if in_reservoir and line.startswith("      max_vol:"):
            result[current] = float(line.split(":", 1)[1])
    if len(result) != 17:
        raise ValueError("Expected 17 reservoir capacities in topology")
    return result


def build_features(root: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame, pd.DataFrame, dict]:
    """Return features, per-unit labels, case frames, and reproducibility metadata.

    ``root`` may be the repository root or its ``sintef_project`` directory.
    X contains all training cases then all test cases, 168 rows per case.
    Each label vector covers training cases only in the same case-hour order.
    The case frames contain Run No and UTC starttime. The metadata includes
    ordered units, submission label columns, and detailed feature definitions.
    """
    root = Path(root)
    if (root / "sintef_project").is_dir():
        root = root / "sintef_project"
    kernel = root / "data" / "kernel"
    extended = root / "data" / "extended"
    training = pd.read_csv(kernel / "Unit_commitment_decisions.csv")
    train = training[["Run No", "starttime"]].copy()
    mapping = pd.read_csv(root / "prediction_mapping.csv")
    train["starttime"] = pd.to_datetime(train["starttime"], utc=True)
    mapping["starttime"] = pd.to_datetime(mapping["starttime"], utc=True)
    cases = pd.concat([train, mapping], ignore_index=True)
    if cases["Run No"].duplicated().any() or cases["starttime"].duplicated().any():
        raise ValueError("Case IDs and start dates must be unique")
    label_columns = pd.read_csv(root / "sample_submission.csv", nrows=0).columns.tolist()[1:]
    if training.columns.tolist()[3:] != label_columns:
        raise ValueError("Training and submission labels have different order")
    units = [column.removeprefix("result_committed_").removesuffix("_t0")
             for column in label_columns if column.endswith("_t0")]
    labels = {}
    for unit in units:
        columns = [f"result_committed_{unit}_t{hour}" for hour in range(HORIZON)]
        y = training[columns].to_numpy(dtype=np.uint8)
        if not np.isin(training[columns].to_numpy(), [0, 1]).all():
            raise ValueError(f"Invalid binary labels for {unit}")
        labels[unit] = y.reshape(-1)
    del training

    starts = pd.DatetimeIndex(cases["starttime"])
    terminal_dates = starts + pd.Timedelta(days=7)
    n_cases = len(cases)
    hours = np.arange(HORIZON)
    timestamps = pd.DatetimeIndex(
        np.repeat(starts.as_unit("ns").asi8, HORIZON)
        + np.tile(hours.astype(np.int64) * 3_600_000_000_000, n_cases), tz="UTC"
    )
    price_frame = _read_timeseries(kernel / "Historical_day_ahead_price_2015_2025.csv")
    price = _lookup(price_frame, timestamps, "price").reshape(n_cases, HORIZON)
    volume_frame = _read_timeseries(kernel / "Historical_volume_2015_2024.csv")
    water_frame = _read_timeseries(kernel / "Synthetic_water_value_2015_2024.csv")
    volume = _lookup(volume_frame, starts, "initial volume")
    water = _lookup(water_frame, terminal_dates, "terminal water value")
    inflow_frame = _read_timeseries(kernel / "Historical_inflow_1958_2025.csv")
    inflow = _lookup(inflow_frame, timestamps, "inflow").reshape(n_cases, HORIZON, -1)
    capacities = _capacities(extended / "Tokke_Vinje_topology.yaml")
    features = {}

    def add(name: str, value: np.ndarray) -> None:
        array = np.asarray(value, dtype=np.float32)
        if array.shape == (n_cases,):
            array = np.repeat(array, HORIZON)
        elif array.shape == (HORIZON,):
            array = np.tile(array, n_cases)
        else:
            array = array.reshape(-1)
        if array.size != n_cases * HORIZON:
            raise ValueError(f"Wrong feature size: {name}: {array.shape}")
        features[name] = array

    add("horizon_hour", hours)
    add("horizon_day", hours // 24)
    add("hour_of_day", hours % 24)
    add("hour_sin", np.sin(2 * np.pi * hours / 24))
    add("hour_cos", np.cos(2 * np.pi * hours / 24))
    add("weekday", timestamps.dayofweek.to_numpy())
    add("month", timestamps.month.to_numpy())
    add("day_of_year", timestamps.dayofyear.to_numpy())
    add("season_sin", np.sin(2 * np.pi * timestamps.dayofyear.to_numpy() / 365.25))
    add("season_cos", np.cos(2 * np.pi * timestamps.dayofyear.to_numpy() / 365.25))
    add("price", price)
    for offset in (-24, -12, -6, -3, -1, 1, 3, 6, 12, 24):
        add(f"price_offset_{offset:+d}", price[:, np.clip(hours + offset, 0, HORIZON - 1)])
    price_mean = price.mean(axis=1)
    price_std = price.std(axis=1)
    add("price_week_mean", price_mean)
    add("price_week_std", price_std)
    add("price_week_min", price.min(axis=1))
    add("price_week_max", price.max(axis=1))
    for quantile in (0.1, 0.25, 0.5, 0.75, 0.9):
        add(f"price_week_q{int(100 * quantile)}", np.quantile(price, quantile, axis=1))
    add("price_minus_week_mean", price - price_mean[:, None])
    add("price_week_zscore", (price - price_mean[:, None]) / (price_std[:, None] + 1))
    add("price_week_rank", pd.DataFrame(price).rank(axis=1, pct=True).to_numpy())
    daily_price = price.reshape(n_cases, 7, 24)
    for stat in ("mean", "min", "max", "std"):
        add(f"price_day_{stat}", np.repeat(getattr(daily_price, stat)(axis=2), 24, axis=1))
    add("price_remaining_mean", np.cumsum(price[:, ::-1], axis=1)[:, ::-1] / (HORIZON - hours))
    add("price_remaining_max", np.maximum.accumulate(price[:, ::-1], axis=1)[:, ::-1])

    for idx, reservoir in enumerate(volume_frame.columns):
        initial = volume[:, idx]
        capacity = capacities[reservoir]
        water_idx = water_frame.columns.get_loc(reservoir)
        terminal = water[:, water_idx]
        add(f"volume_{reservoir}", initial)
        add(f"fill_{reservoir}", initial / capacity)
        add(f"water_value_{reservoir}", terminal)
        add(f"price_minus_water_{reservoir}", price - terminal[:, None])
        add(f"price_over_water_{reservoir}", price / (terminal[:, None] + 1))
        inflow_idx = inflow_frame.columns.get_loc(reservoir)
        inflow_volume = inflow[:, :, inflow_idx].sum(axis=1) * 0.0036
        add(f"headroom_after_inflow_{reservoir}", (capacity - initial - inflow_volume) / capacity)
    for idx, reservoir in enumerate(inflow_frame.columns):
        flow = inflow[:, :, idx]
        add(f"inflow_{reservoir}", flow)
        add(f"inflow_week_mean_{reservoir}", flow.mean(axis=1))
        add(f"inflow_elapsed_mm3_{reservoir}", np.cumsum(flow, axis=1) * 0.0036)
        add(f"inflow_remaining_mm3_{reservoir}", np.cumsum(flow[:, ::-1], axis=1)[:, ::-1] * 0.0036)
    for filename, prefix in (("Constraint_min_volume.csv", "min_volume"),
                             ("Constraint_min_flow.csv", "min_flow")):
        frame = _read_timeseries(extended / filename)
        values = _lookup(frame, timestamps, prefix).reshape(n_cases, HORIZON, -1)
        for idx, name in enumerate(frame.columns):
            add(f"{prefix}_{name}", values[:, :, idx])
            add(f"{prefix}_week_max_{name}", values[:, :, idx].max(axis=1))
    X = pd.DataFrame(features, copy=False)
    for name in X:
        if not np.isfinite(X[name].to_numpy()).all():
            raise ValueError(f"Nonfinite feature {name}")
    metadata = {
        "units": units,
        "label_columns": label_columns,
        "feature_names": list(X.columns),
        "n_train_cases": len(train),
        "n_test_cases": len(mapping),
        "horizon_hours": HORIZON,
        "row_order": "training CSV cases, then prediction mapping cases; hour increases fastest",
        "feature_spec": {
            "volume": "Initial storage at case start; fill = volume/max_vol from topology",
            "water_value": "Terminal water value at start+7 days; margins price-water; ratio price/(water+1)",
            "inflow": "Hourly flow, 168h mean, elapsed incl. current and remaining incl. current volumes; m3/s times 0.0036 -> million m3 per hour",
            "headroom": "(max_vol-initial_volume-sum(horizon_inflow)*0.0036)/max_vol; no generation/outflow approximation",
            "price": "Current, +/-1/3/6/12/24h clamped inside case, horizon stats/rank, daily stats, remaining mean/max; zscore divisor std+1",
            "constraints": "Current hourly and maximum across 168-hour horizon",
            "calendar": "UTC; annual sine/cosine period 365.25; daily sine/cosine period24",
            "excluded": "Commitment labels, calculation time, future historical reservoir volumes, calendar year, row ID",
        },
    }
    return X, labels, train, mapping, metadata
