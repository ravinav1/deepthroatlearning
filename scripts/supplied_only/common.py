"""Shared helpers for the supplied-data-only attempt (November 2026).

Information boundary: every file this package reads must come from the
supplied competition materials or from locally computed, audited caches whose
ancestry is supplied-only. Paths under the EXCLUDED prefixes are refused.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

CLONE = Path(__file__).resolve().parents[2]
WORKSPACE = Path(os.environ.get("TDT4173_WORKSPACE", "/Users/heap/programming/tdt4173"))
ARTIFACTS = WORKSPACE / "artifacts"
RAW = WORKSPACE / "sintef_project"
OUT = CLONE / "models/supplied_only"
REPORTS = CLONE / "reports/supplied_only"

HORIZON = 168
UNITS = ["Hogga_G1", "Tokke_G1", "Tokke_G2", "Tokke_G3", "Tokke_G4", "Lio_G1", "Byrte_G1",
         "Vinje_G1", "Vinje_G2", "Vinje_G3", "Haukeli_G1", "Songa_G1", "Kjela_G1", "Vesle Kjela_G1"]
PLANTS = ["Hogga", "Tokke", "Lio", "Byrte", "Vinje", "Haukeli", "Songa", "Kjela", "Vesle Kjela"]
UNIT_PLANT = [u.rsplit("_G", 1)[0] for u in UNITS]

# Directories that contain or derive from external solver outputs. Never read.
EXCLUDED_PREFIXES = ("external-", "metric-trick-shop", "metric-trick-public", "metric-trick-retrospective",
                     "metric-trick-nomip", "metric-trick-mip", "physics-trick-shop", "structural-trick-public",
                     "provided-only-latent")
EXCLUDED_TOKENS = ("solver_committed", "all_mip", "no_mip", "shop17", "external")

SUPPLIED_FILES = [
    RAW / "data/kernel/Historical_day_ahead_price_2015_2025.csv",
    RAW / "data/kernel/Historical_inflow_1958_2025.csv",
    RAW / "data/kernel/Historical_volume_2015_2024.csv",
    RAW / "data/kernel/Synthetic_water_value_2015_2024.csv",
    RAW / "data/kernel/Unit_commitment_decisions.csv",
    RAW / "data/extended/Constraint_min_flow.csv",
    RAW / "data/extended/Constraint_min_volume.csv",
    RAW / "data/extended/Tokke_Vinje_topology.yaml",
    RAW / "prediction_mapping.csv",
    RAW / "sample_submission.csv",
]


def check_allowed(path: Path) -> Path:
    """Refuse any path that touches an external-derived artifact."""
    path = Path(path).resolve()
    rel = str(path)
    try:
        rel = str(path.relative_to(ARTIFACTS))
    except ValueError:
        pass
    low = rel.lower()
    if any(low.startswith(p) for p in EXCLUDED_PREFIXES) or any(t in low for t in EXCLUDED_TOKENS):
        raise PermissionError(f"Refusing external-derived input: {path}")
    return path


def sha256(path: Path) -> str:
    with open(check_allowed(path), "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class InputLedger:
    """Record every input path and hash used by a run."""

    def __init__(self) -> None:
        self.records: dict[str, str] = {}

    def add(self, path: Path) -> Path:
        path = check_allowed(path)
        key = str(path)
        if key not in self.records:
            self.records[key] = sha256(path)
        return path

    def write(self, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"inputs": self.records}, indent=1, sort_keys=True))


def micro_auc(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Pooled binary AUC with average ranks for ties (same as sklearn micro)."""
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.asarray(labels).ravel(), np.asarray(predictions, dtype=np.float64).ravel()))


def load_cases(ledger: InputLedger | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (training cases with labels, test mapping). Labels shape (n, 14, 168)."""
    path = RAW / "data/kernel/Unit_commitment_decisions.csv"
    mapping = RAW / "prediction_mapping.csv"
    if ledger is not None:
        ledger.add(path); ledger.add(mapping)
    columns = [f"result_committed_{u}_t{h}" for u in UNITS for h in range(HORIZON)]
    frame = pd.read_csv(path, usecols=["Run No", "starttime", *columns],
                        dtype={c: "uint8" for c in columns})
    frame["starttime"] = pd.to_datetime(frame["starttime"], utc=True)
    frame = frame.sort_values("Run No").reset_index(drop=True)
    labels = frame[columns].to_numpy().reshape(len(frame), len(UNITS), HORIZON)
    cases = frame[["Run No", "starttime"]].copy()
    cases["labels_index"] = np.arange(len(cases))
    test = pd.read_csv(mapping)
    test["starttime"] = pd.to_datetime(test["starttime"], utc=True)
    return cases.assign(labels=list(labels)), test


def load_labels_array(ledger: InputLedger | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    cases, _ = load_cases(ledger)
    labels = np.stack(cases.pop("labels").to_list())
    return cases, labels


def validation_npz(path: Path, ledger: InputLedger | None = None) -> dict:
    path = check_allowed(path)
    if ledger is not None:
        ledger.add(path)
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files}
    assert list(out["units"]) == UNITS, path
    return out


def fold_ids(year: int) -> tuple[int, int]:
    """Validation case-ID range for the two development years."""
    return {2021: (2193, 2557), 2022: (2558, 2922)}[year]
