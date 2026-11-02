"""Hydropower inputs, with one row per generator, case and hour."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from topology_features import topology_features
from price_features import neighbouring_prices
from operating_features import water_context, economic_context
from hydraulic_features import initial_hydraulics
from water_budget_features import natural_water_budgets
from temporal_features import temporal_ranks, favourable_periods


ROOT = Path(__file__).resolve().parents[1]
HOURS = 168


def load_data(use_extended=False):
    data_dir = ROOT / "data/raw/kernel"
    label_columns = pd.read_csv(ROOT / "data/raw/sample_submission.csv", nrows=0).columns[1:].tolist()
    generators = list(dict.fromkeys(
        column.removeprefix("result_committed_").rsplit("_t", 1)[0]
        for column in label_columns
    ))
    assert len(generators) == 14
    assert label_columns == [
        f"result_committed_{generator}_t{hour}"
        for generator in generators for hour in range(HOURS)
    ]

    # Solver runtime and case identifiers are not model features.
    cases = pd.read_csv(
        data_dir / "Unit_commitment_decisions.csv",
        usecols=["Run No", "starttime", *label_columns],
        dtype={column: "uint8" for column in label_columns},
    )
    cases["starttime"] = pd.to_datetime(cases["starttime"], utc=True)
    cases = cases.sort_values("starttime").reset_index(drop=True)

    files = {
        "price": "Historical_day_ahead_price_2015_2025.csv",
        "volume": "Historical_volume_2015_2024.csv",
        "water_value": "Synthetic_water_value_2015_2024.csv",
        "inflow": "Historical_inflow_1958_2025.csv",
    }
    inputs = {}
    for name, filename in files.items():
        frame = pd.read_csv(data_dir / filename)
        frame["date"] = pd.to_datetime(frame["date"], utc=True)
        inputs[name] = frame.set_index("date")

    if use_extended:
        for name in ["min_volume", "min_flow"]:
            frame = pd.read_csv(ROOT / f"data/raw/extended/Constraint_{name}.csv")
            frame["date"] = pd.to_datetime(frame["date"], utc=True)
            inputs[name] = frame.set_index("date")

    with (ROOT / "configs/Tokke_Vinje_topology.yaml").open() as file:
        topology = yaml.safe_load(file)
    return cases, label_columns, generators, inputs, topology


def feeding_reservoirs(topology, name, kind="plant"):
    """Follow incoming tunnels until reaching the plant's feeding reservoirs."""
    reservoirs = []
    for link in topology["connections"]:
        if (link["to"], link["to_type"]) != (name, kind):
            continue
        if link["from_type"] == "reservoir":
            reservoirs.append(link["from"])
        elif link["from_type"] == "tunnel":
            reservoirs.extend(feeding_reservoirs(topology, link["from"], "tunnel"))
    return sorted(set(reservoirs))


def build_features(cases, generators, inputs, topology, use_extended=False, use_system_context=False,
                   use_topology=False, use_price_context=False, use_water_context=False,
                   use_economic_context=False, use_river_context=False,
                   use_inflow_profiles=False, use_hydraulic_context=False, use_water_budget=False,
                   use_temporal_ranks=False, use_favourable_periods=False):
    # All rows for generator 0, then generator 1, etc.; cases and hours stay ordered.
    starts = cases["starttime"]
    horizon_hour = np.tile(np.arange(HOURS), len(cases))
    timestamps = pd.DatetimeIndex(np.repeat(starts.array, HOURS))
    timestamps = timestamps + pd.to_timedelta(horizon_hour, unit="h")
    price = inputs["price"].loc[timestamps].iloc[:, 0].to_numpy().reshape(-1, HOURS)

    common = pd.DataFrame({
        "horizon_hour": horizon_hour,
        "hour_of_day": timestamps.hour,
        "day_of_week": timestamps.dayofweek,
        "month": timestamps.month,
        "price": price.ravel(),
        "week_mean_price": np.repeat(price.mean(axis=1), HOURS),
        "week_max_price": np.repeat(price.max(axis=1), HOURS),
        "price_minus_week_mean": (price - price.mean(axis=1, keepdims=True)).ravel(),
        # Later hours' provided prices are valid inputs for this scheduling task.
        "remaining_max_price": np.maximum.accumulate(price[:, ::-1], axis=1)[:, ::-1].ravel(),
    })

    if use_price_context:
        for name, values in neighbouring_prices(price).items():
            common[name] = values

    if use_temporal_ranks:
        for name, values in temporal_ranks(price).items():
            common[name] = values.ravel()

    if use_river_context:
        assert use_extended, "River shortfall needs the extended minimum-flow data."
        river = "r_Vest_Vassdraget"
        natural = inputs["inflow"].loc[timestamps, river].to_numpy()
        required = inputs["min_flow"].loc[timestamps, river].to_numpy()
        common["river_natural_inflow_m3_s"] = natural
        # Other releases can cover this gap; it does not force Lio ON.
        common["river_min_flow_shortfall_m3_s"] = np.maximum(required - natural, 0)

    if use_system_context:
        # Give every generator the starting state and supplied week inputs of all reservoirs.
        for name in inputs["volume"].columns:
            volume = inputs["volume"].loc[starts, name].to_numpy()
            water_value = inputs["water_value"].loc[
                starts + pd.Timedelta(hours=HOURS), name
            ].to_numpy()
            inflow = inputs["inflow"].loc[timestamps, name].to_numpy().reshape(-1, HOURS)
            common[f"system_initial_volume_{name}"] = np.repeat(volume, HOURS)
            common[f"system_terminal_water_value_{name}"] = np.repeat(water_value, HOURS)
            common[f"system_week_inflow_million_m3_{name}"] = np.repeat(
                inflow.sum(axis=1) * 3600 / 1_000_000, HOURS
            )

    if use_extended:
        # System-wide requirements are inputs, not rules that force a generator ON/OFF.
        for kind in ["min_volume", "min_flow"]:
            constraints = inputs[kind].loc[timestamps]
            for name in constraints.columns:
                values = constraints[name].to_numpy().reshape(-1, HOURS)
                common[f"{kind}_{name}"] = values.ravel()
                common[f"week_max_{kind}_{name}"] = np.repeat(values.max(axis=1), HOURS)
                if kind == "min_volume":
                    initial = inputs["volume"].loc[starts, name].to_numpy()
                    # Starting volume relative to each hour's requirement, not predicted storage.
                    common[f"starting_volume_minus_min_volume_{name}"] = (
                        initial[:, None] - values
                    ).ravel()

    extra = {}
    if use_inflow_profiles:
        # Keep each reservoir's timing; these are natural inputs, not downstream arrivals.
        for name in inputs["volume"].columns:
            flow = inputs["inflow"].loc[timestamps, name].to_numpy().reshape(-1, HOURS)
            water = flow * 3600 / 1_000_000
            extra[f"profile_natural_inflow_{name}_m3_s"] = flow.ravel()
            extra[f"profile_cumulative_natural_inflow_{name}_million_m3"] = water.cumsum(axis=1).ravel()
            extra[f"profile_remaining_natural_inflow_{name}_million_m3"] = water[:, ::-1].cumsum(axis=1)[:, ::-1].ravel()
    if use_hydraulic_context:
        extra.update({name: np.repeat(values, HOURS)
                      for name, values in initial_hydraulics(starts, inputs, topology).items()})
    if use_water_budget:
        assert use_extended, "Water budgets need extended minimum-flow and volume data."
        extra.update(natural_water_budgets(starts, timestamps, inputs, topology))
    if extra:
        common = pd.concat([common, pd.DataFrame(extra)], axis=1)

    frames = []
    for generator_id, generator in enumerate(generators):
        plant = generator.rsplit("_G", 1)[0]
        reservoirs = feeding_reservoirs(topology, plant)
        capacity = sum(topology["model"]["reservoir"][name]["max_vol"] for name in reservoirs)
        volume = inputs["volume"].loc[starts, reservoirs].sum(axis=1, skipna=False).to_numpy()
        # For Songa's two reservoirs this is a simple average, not a hydraulic valuation.
        water_value = inputs["water_value"].loc[
            starts + pd.Timedelta(hours=HOURS), reservoirs
        ].mean(axis=1, skipna=False).to_numpy()
        inflow = inputs["inflow"].loc[timestamps, reservoirs].sum(axis=1, skipna=False).to_numpy().reshape(-1, HOURS)
        inflow_million_m3 = inflow * 3600 / 1_000_000

        frame = common.copy()
        frame["generator_id"] = generator_id
        frame["initial_volume_million_m3"] = np.repeat(volume, HOURS)
        frame["initial_fill_fraction"] = np.repeat(volume / capacity, HOURS)
        frame["terminal_water_value"] = np.repeat(water_value, HOURS)
        frame["price_minus_water_value"] = frame["price"] - frame["terminal_water_value"]
        if use_favourable_periods:
            for name, values in favourable_periods(price, water_value).items():
                frame[name] = values.ravel()
        frame["natural_inflow_m3_s"] = inflow.ravel()
        # Natural inflow only: these features exclude releases and upstream transfers.
        frame["cumulative_natural_inflow_million_m3"] = inflow_million_m3.cumsum(axis=1).ravel()
        frame["week_natural_inflow_million_m3"] = np.repeat(inflow_million_m3.sum(axis=1), HOURS)
        if use_topology:
            for name, values in topology_features(plant, starts, timestamps, inputs, topology).items():
                frame[name] = np.repeat(values, HOURS)
        if use_water_context:
            for name, values in water_context(plant, volume, capacity, inflow, timestamps, inputs, topology).items():
                frame[name] = values
        if use_economic_context:
            for name, values in economic_context(price, water_value, topology["model"]["generator"][generator]).items():
                frame[name] = values
        frames.append(frame.astype("float32"))

    features = pd.concat(frames, ignore_index=True)
    assert np.isfinite(features.to_numpy()).all(), "Missing or invalid feature values."
    return features


def flatten_labels(cases, label_columns, generators):
    """Match build_features' generator → case → hour row order."""
    labels = cases[label_columns].to_numpy()
    assert np.isin(labels, [0, 1]).all()
    return labels.reshape(len(cases), len(generators), HOURS).transpose(1, 0, 2).ravel()


def prediction_frame(probabilities, cases, label_columns, generators):
    """Restore the competition's case → generator → hour column order."""
    values = probabilities.reshape(len(generators), len(cases), HOURS)
    values = values.transpose(1, 0, 2).reshape(len(cases), -1)
    frame = pd.DataFrame(values, columns=label_columns)
    frame.insert(0, "Run No", cases["Run No"].to_numpy())
    return frame
