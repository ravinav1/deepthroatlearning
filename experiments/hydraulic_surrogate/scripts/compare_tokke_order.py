"""Select violation-only Tokke correction on2021, then lock and gate on2022."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shlex
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import roc_auc_score

from sintef_project.lp_features import UNIT_ORDER
from sintef_project.pair_order import soften_pair_order_violations


def digest(path: Path) -> str:
    """Hash an artifact without decoding its arrays."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_fold(path: Path, year: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Check known chronological IDs, unit order, values and existing projection."""
    with np.load(path, allow_pickle=False) as data:
        prediction, labels, ids, units = data["predictions"], data["labels"], data["case_ids"], data["units"].tolist()
    expected_start = {2021: 2193, 2022: 2558}[year]
    np.testing.assert_array_equal(ids, np.arange(expected_start, expected_start + 365))
    assert units == UNIT_ORDER
    assert prediction.shape == labels.shape == (365 * 168, 14)
    assert np.isfinite(prediction).all() and ((prediction >= 0) & (prediction <= 1)).all()
    assert np.isin(labels, [0, 1]).all()
    vinje = prediction[:, [units.index(f"Vinje_G{j}") for j in [1, 2, 3]]]
    assert (np.diff(vinje, axis=1) <= 0).all()
    return prediction, labels, ids, units


def scores(labels: np.ndarray, probabilities: np.ndarray, units: list[str]) -> dict:
    """Report pooled AUC and the two affected unit AUCs."""
    return {"micro_auc": float(roc_auc_score(labels.ravel(), probabilities.ravel())),
            "pair_unit_auc": {unit: float(roc_auc_score(labels[:, units.index(unit)], probabilities[:, units.index(unit)]))
                              for unit in ["Tokke_G1", "Tokke_G2"]}}


def main() -> None:
    """Freeze the four-alpha protocol before loading prediction arrays."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="tokke-order-comparison-001")
    args = parser.parse_args()
    out = ROOT / "artifacts" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    parent = ROOT / "artifacts/lp-joint-comparison-001"
    analysis_path = parent / "analysis.json"
    proof_path = ROOT / "artifacts/verify-lp-joint-comparison-001/manifest.json"
    analysis = json.loads(analysis_path.read_text())
    proof = json.loads(proof_path.read_text())
    assert analysis["accepted"] and proof["status"] == "passed"
    assert digest(analysis_path) == proof["analysis_sha256"]
    manifest = {"run_id": args.run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
                "command": shlex.join([sys.executable, *sys.argv]), "source": {}, "inputs": {},
                "environment": {"python": sys.version, "platform": platform.platform(),
                                "packages": {p: importlib.metadata.version(p) for p in ["numpy", "scikit-learn"]}},
                "alphas": [0., .25, .5, 1.], "pair": ["Tokke_G1", "Tokke_G2"],
                "formula": "Only where p_G1<p_G2, replace each p by (1-alpha)*p+alpha*(p_G1+p_G2)/2. All other values, including the previously applied Vinje projection, remain bitwise unchanged.",
                "selection": "Maximum pooled2021 ROC AUC; exact ties prefer smaller alpha. Write immutable selection before decoding2022. Evaluate only baseline and locked selected alpha on2022; accept only strict AUC improvement on BOTH years.",
                "baseline": {"comparison": str(parent.relative_to(ROOT)), "weights": analysis["selected_weights"],
                             "already_projected": "VinjeG1>=G2>=G3"},
                "scope": "Empirical soft label-order hypothesis despite rare exceptions; no physical hard-order claim, fitted model, subset-specific rule or threshold.",
                "training": "None. No early stopping or seed. Selection reuses adaptively researched chronological2021;2022 is acceptance only and not a new independent holdout.",
                "kaggle": {"submission_id": None, "score": None, "rank": None, "observation_time": None}}
    for source in [Path(__file__), ROOT / "sintef_project/pair_order.py", ROOT / "tests/test_pair_order.py"]:
        relative = str(source.relative_to(ROOT))
        manifest["source"][relative] = digest(source)
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for path in [analysis_path, proof_path, ROOT / "artifacts/tokke-exchangeability-001/manifest.json"]:
        manifest["inputs"][str(path.relative_to(ROOT))] = digest(path)
    for year in [2021, 2022]:
        path = parent / f"selected_validation_{year}.npz"
        assert digest(path) == analysis["artifacts"][path.name]
        manifest["inputs"][str(path.relative_to(ROOT))] = digest(path)
    manifest["data"] = analysis["data"]
    for relative, expected in manifest["data"].items():
        assert digest(ROOT / relative) == expected
    (out / "protocol.json").write_text(json.dumps(manifest, indent=2))
    manifest["protocol_sha256"] = digest(out / "protocol.json")
    try:
        prediction, labels, ids, units = load_fold(parent / "selected_validation_2021.npz", 2021)
        pair = [units.index(name) for name in manifest["pair"]]
        untouched = [j for j in range(14) if j not in pair]
        violations = prediction[:, pair[0]] < prediction[:, pair[1]]
        results = []
        for alpha in manifest["alphas"]:
            transformed = soften_pair_order_violations(prediction, units, alpha)
            np.testing.assert_array_equal(transformed[:, untouched], prediction[:, untouched])
            np.testing.assert_array_equal(transformed[~violations], prediction[~violations])
            results.append({"alpha": alpha, **scores(labels, transformed, units)})
        chosen = max(results, key=lambda result: (result["micro_auc"], -result["alpha"]))
        selection = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "alpha": chosen["alpha"],
                     "scores_2021": results, "selection_rule": manifest["selection"],
                     "protocol_sha256": manifest["protocol_sha256"], "prediction_2021_sha256": digest(parent / "selected_validation_2021.npz"),
                     "2022_arrays_decoded": False, "violation_hours_2021": int(violations.sum())}
        (out / "selection_2021.json").write_text(json.dumps(selection, indent=2))
        manifest["selection_sha256"] = digest(out / "selection_2021.json")
        manifest["selection_locked_utc"] = selection["timestamp_utc"]
        manifest["selected_alpha"] = chosen["alpha"]
        manifest["folds"] = {"2021": {"grid": results, "baseline": results[0], "selected": chosen,
                                            "violation_hours": int(violations.sum()), "case_ids": ids.tolist()}}
        transformed = soften_pair_order_violations(prediction, units, chosen["alpha"])
        np.savez_compressed(out / "selected_validation_2021.npz", predictions=transformed, labels=labels, case_ids=ids, units=np.asarray(units))
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        # The only held-out transformation evaluated is the already locked alpha.
        manifest["2022_decode_started_utc"] = datetime.now(timezone.utc).isoformat()
        prediction, labels, ids, units = load_fold(parent / "selected_validation_2022.npz", 2022)
        transformed = soften_pair_order_violations(prediction, units, chosen["alpha"])
        violations = prediction[:, pair[0]] < prediction[:, pair[1]]
        np.testing.assert_array_equal(transformed[:, untouched], prediction[:, untouched])
        np.testing.assert_array_equal(transformed[~violations], prediction[~violations])
        baseline, selected = scores(labels, prediction, units), scores(labels, transformed, units)
        manifest["folds"]["2022"] = {"baseline": baseline, "selected": selected, "violation_hours": int(violations.sum()), "case_ids": ids.tolist()}
        manifest["accepted"] = chosen["micro_auc"] > results[0]["micro_auc"] and selected["micro_auc"] > baseline["micro_auc"]
        manifest["accepted_alpha"] = chosen["alpha"] if manifest["accepted"] else 0.
        manifest["meets_migration_gate"] = manifest["accepted"] and min(chosen["micro_auc"], selected["micro_auc"]) >= .998
        np.savez_compressed(out / "selected_validation_2022.npz", predictions=transformed, labels=labels, case_ids=ids, units=np.asarray(units))
        manifest["status"] = "complete"
        print(json.dumps({"selected_alpha": chosen["alpha"], "2021_grid": results, "2022": manifest["folds"]["2022"] | {"case_ids": "2558..2922"}, "accepted": manifest["accepted"]}))
    except BaseException:
        manifest["status"] = "failed"
        manifest["error"] = traceback.format_exc()
        raise
    finally:
        manifest["runtime_seconds"] = time.monotonic() - started
        manifest["artifacts"] = {str(path.relative_to(out)): digest(path) for path in out.rglob("*") if path.is_file() and path.name != "manifest.json"}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
