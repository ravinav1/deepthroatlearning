"""Training-only Tokke G1/G2 label-order audit before any shrinkage selection."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shlex
import sys
import time

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    """Return one input's SHA256."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def summarize(first: np.ndarray, second: np.ndarray) -> dict:
    """Describe pair asymmetry without assuming monotonic commitment order."""
    code = 2 * first + second
    count = np.bincount(code.ravel(), minlength=4)
    discordant = int(count[1] + count[2])
    return {"hours": int(code.size), "counts_00_01_10_11": count.tolist(),
            "rates_00_01_10_11": (count / code.size).tolist(),
            "g1_on_rate": float(first.mean()), "g2_on_rate": float(second.mean()),
            "discordant_hours": discordant,
            "g1_only_share_of_discordance": float(count[2] / discordant) if discordant else None,
            "g2_only_share_of_discordance": float(count[1] / discordant) if discordant else None,
            "cases_with_any_g1_only": int((code == 2).any(axis=1).sum()),
            "cases_with_any_g2_only": int((code == 1).any(axis=1).sum()),
            "cases_g1_only_all_hours": int((code == 2).all(axis=1).sum()),
            "cases_g2_only_all_hours": int((code == 1).all(axis=1).sum())}


def main() -> None:
    """Read exactly the 2015–2020 prefix and save descriptive label evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="tokke-exchangeability-001")
    args = parser.parse_args()
    out = ROOT / "artifacts" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source = ROOT / "sintef_project/data/kernel/Unit_commitment_decisions.csv"
    topology = ROOT / "sintef_project/data/extended/Tokke_Vinje_topology.yaml"
    units = ["Tokke_G1", "Tokke_G2"]
    columns = {unit: [f"result_committed_{unit}_t{hour}" for hour in range(168)] for unit in units}
    # nrows prevents parsing any 2021/2022 holdout label values in this first stage.
    data = pd.read_csv(source, nrows=2192, usecols=["Run No", "starttime", *columns[units[0]], *columns[units[1]]])
    dates = pd.to_datetime(data.starttime, utc=True)
    assert dates.min() == pd.Timestamp("2015-01-01", tz="UTC")
    assert dates.max() == pd.Timestamp("2020-12-31", tz="UTC")
    np.testing.assert_array_equal(data["Run No"].to_numpy(), np.arange(1, 2193))
    first, second = (data[columns[unit]].to_numpy(dtype=np.uint8) for unit in units)
    assert np.isin(first, [0, 1]).all() and np.isin(second, [0, 1]).all()
    model = yaml.safe_load(topology.read_text())["model"]
    report = {"run_id": args.run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "status": "complete",
              "command": shlex.join([sys.executable, *sys.argv]),
              "source": {str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))},
              "data": {str(path.relative_to(ROOT)): digest(path) for path in [source, topology]},
              "environment": {"python": sys.version, "platform": platform.platform(),
                              "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "PyYAML"]}},
              "train_case_ids": data["Run No"].tolist(), "date_range": [str(dates.min()), str(dates.max())],
              "label_parse_nrows": 2192, "holdout_predictions_read": False, "holdout_labels_parsed": False,
              "pair": units, "generator_attributes_identical": model["generator"][units[0]] == model["generator"][units[1]],
              "overall": summarize(first, second),
              "year": {str(year): summarize(first[dates.dt.year == year], second[dates.dt.year == year]) for year in range(2015, 2021)},
              "horizon_day": {str(day): summarize(first[:, day * 24:(day + 1) * 24], second[:, day * 24:(day + 1) * 24]) for day in range(7)},
              "month": {str(month): summarize(first[dates.dt.month == month], second[dates.dt.month == month]) for month in range(1, 13)},
              "interpretation": "Descriptive training-case evidence, not independent hourly samples because weekly cases overlap. No hard unit-order rule or new probability transformation inferred automatically.",
              "validation_score": None, "kaggle": None}
    g2only = (first == 0) & (second == 1)
    order = np.argsort(-g2only.sum(axis=1), kind="stable")[:10]
    report["largest_g2_only_cases"] = [{"case_id": int(data.iloc[i]["Run No"]), "date": str(dates.iloc[i]),
                                      "g2_only_hours": int(g2only[i].sum()), "g1_only_hours": int(((first[i] == 1) & (second[i] == 0)).sum())}
                                     for i in order if g2only[i].any()]
    np.savez_compressed(out / "training_pair.npz", case_ids=data["Run No"].to_numpy(), first=first, second=second, units=np.asarray(units))
    report["runtime_seconds"] = time.monotonic() - started
    report["artifacts"] = {"training_pair.npz": digest(out / "training_pair.npz")}
    (out / "manifest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in ["generator_attributes_identical", "overall", "year", "runtime_seconds"]}))


if __name__ == "__main__":
    main()
