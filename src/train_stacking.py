"""Fit a small combination model from chronological out-of-fold predictions."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from features import HOURS, ROOT


def load_predictions(suffix, members, columns):
    frames = [pd.read_csv(ROOT / f"data/processed/stacking_{member}_{suffix}_predictions.csv")
              .set_index("Run No") for member in members]
    reference = frames[0]
    for frame in frames:
        assert frame.index.is_unique and frame.index.equals(reference.index)
        assert frame.columns.tolist() == columns
        values = frame.to_numpy()
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
    # Each column is one model; rows remain case -> generator -> hour.
    probabilities = np.column_stack([frame.to_numpy().ravel() for frame in frames])
    return reference.index, probabilities


def design_matrix(logits, mean, scale, generator_count):
    case_count = len(logits) // (generator_count * HOURS)
    generator_ids = np.tile(np.repeat(np.arange(generator_count), HOURS), case_count)
    indicators = sparse.csr_matrix(
        (np.ones(len(logits)), (np.arange(len(logits)), generator_ids)),
        shape=(len(logits), generator_count),
    )
    return sparse.hstack([sparse.csr_matrix((logits - mean) / scale), indicators], format="csr")


def metrics(labels, predictions, generators):
    labels = labels.reshape(-1, len(generators), HOURS)
    predictions = predictions.reshape(labels.shape)
    return {
        "micro_roc_auc": float(roc_auc_score(labels.ravel(), predictions.ravel())),
        "generator_auc": {
            generator: float(roc_auc_score(labels[:, i].ravel(), predictions[:, i].ravel()))
            for i, generator in enumerate(generators)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/stacking.json"))
    parser.add_argument("--final", action="store_true", help="Refit meta on all OOF years and create a submission.")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    suffix = "final" if args.final else str(config["evaluation_year"])
    model_path = ROOT / f"models/stacking_meta_{suffix}.joblib"
    report_path = ROOT / f"reports/stacking_{suffix}_metrics.json"
    output = ROOT / ("submissions/submission_stacking.csv" if args.final
                     else f"data/processed/stacking_{suffix}_validation_predictions.csv")
    for path in (model_path, report_path, output):
        assert not path.exists(), f"Artifact already exists: {path}"

    members = config["profiles"] + ["cnn"]
    template = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
    columns = template.columns[1:].tolist()
    generators = list(dict.fromkeys(column.removeprefix("result_committed_").rsplit("_t", 1)[0]
                                   for column in columns))
    truth = pd.read_csv(ROOT / "data/raw/kernel/Unit_commitment_decisions.csv",
                        usecols=["Run No", "starttime", *columns],
                        dtype={column: "uint8" for column in columns}).set_index("Run No")
    truth["starttime"] = pd.to_datetime(truth["starttime"], utc=True)
    assert truth.index.is_unique and not truth.index.isin(template["Run No"]).any()

    fit_years = [year for year in config["fold_years"] if args.final or year < config["evaluation_year"]]
    fit_values, fit_targets, fit_ids, purged_ids = [], [], [], []
    boundary = pd.Timestamp(f"{config['evaluation_year']}-01-01", tz="UTC")
    for year in fit_years:
        ids, probabilities = load_predictions(year, members, columns)
        cases = truth.loc[ids]
        assert (cases["starttime"].dt.year == year).all()
        keep = np.ones(len(cases), dtype=bool) if args.final else (
            cases["starttime"] + pd.Timedelta(hours=HOURS) <= boundary
        ).to_numpy()
        purged_ids.extend(ids[~keep].tolist())
        fit_ids.extend(ids[keep].tolist())
        fit_values.append(probabilities.reshape(len(ids), len(columns), len(members))[keep]
                          .reshape(-1, len(members)))
        fit_targets.append(cases.loc[keep, columns].to_numpy().ravel())
    probabilities = np.concatenate(fit_values)
    labels = np.concatenate(fit_targets)
    assert len(set(fit_ids)) == len(fit_ids)
    clip = config["probability_clip"]
    logits = logit(np.clip(probabilities, clip, 1 - clip))
    mean, scale = logits.mean(axis=0), logits.std(axis=0)
    scale[scale < 1e-6] = 1
    model = LogisticRegression(**config["meta"])
    print(f"Meta fit: {len(fit_ids)} OOF cases, {len(labels):,} outputs, {len(purged_ids)} purged", flush=True)
    with threadpool_limits(limits=4):
        matrix = design_matrix(logits, mean, scale, len(generators))
        model.fit(matrix, labels)
        fit_auc = float(roc_auc_score(labels, model.predict_proba(matrix)[:, 1]))
    assert model.classes_.tolist() == [0, 1]
    del matrix, logits, probabilities, fit_values, fit_targets

    # Save the fitted combination before reading evaluation predictions or scoring 2022.
    with model_path.open("xb") as stream:
        joblib.dump({"model": model, "mean": mean, "scale": scale, "members": members,
                     "generators": generators, "config": config}, stream, compress=3)
    ids, probabilities = load_predictions(suffix, members, columns)
    assert not set(fit_ids).intersection(ids), "Evaluation cases must stay out of meta fitting."
    if args.final:
        assert ids.equals(pd.Index(template["Run No"], name="Run No"))
    else:
        assert (truth.loc[ids, "starttime"].dt.year == config["evaluation_year"]).all()
        assert (truth.loc[fit_ids, "starttime"] + pd.Timedelta(hours=HOURS)).max() <= truth.loc[ids, "starttime"].min()
    logits = logit(np.clip(probabilities, clip, 1 - clip))
    with threadpool_limits(limits=4):
        predictions = model.predict_proba(design_matrix(logits, mean, scale, len(generators)))[:, 1]
    assert np.isfinite(predictions).all() and ((predictions >= 0) & (predictions <= 1)).all()
    frame = pd.DataFrame(predictions.reshape(len(ids), len(columns)), index=ids, columns=columns).reset_index()
    assert frame.columns.tolist() == template.columns.tolist()
    frame.to_csv(output, index=False, mode="x")

    report = {
        "config": config, "fit_years": fit_years, "fit_cases": len(fit_ids), "fit_outputs": len(labels),
        "fit_run_ids": fit_ids, "purged_run_ids": purged_ids, "meta_training_auc": fit_auc,
        "meta_training_auc_note": "In-sample for the meta-model; base predictions are chronological OOF.",
        "members": members, "logit_mean": mean.tolist(), "logit_scale": scale.tolist(),
        "standardized_logit_weights": dict(zip(members, model.coef_[0, :len(members)].tolist())),
        "generator_offsets": dict(zip(generators, model.coef_[0, len(members):].tolist())),
        "iterations": int(model.n_iter_[0]), "prediction_cases": len(ids),
        "model": str(model_path), "predictions": str(output),
    }
    if not args.final:
        evaluation_labels = truth.loc[ids, columns].to_numpy().ravel()
        ensemble = probabilities[:, :-1].mean(axis=1)
        fixed_blend = .8 * ensemble + .2 * probabilities[:, -1]
        report["evaluation"] = {
            "stacking": metrics(evaluation_labels, predictions, generators),
            "fixed_80_20_same_bases": metrics(evaluation_labels, fixed_blend, generators),
            "equal_weight_xgboost_same_bases": metrics(evaluation_labels, ensemble, generators),
            "base_models": {member: metrics(evaluation_labels, probabilities[:, i], generators)
                            for i, member in enumerate(members)},
        }
        previous = json.loads((ROOT / "reports/sequence_ensemble_comparison.json").read_text())
        report["previous_selected_budget_reference"] = next(
            row for row in previous["results"]
            if row["year"] == config["evaluation_year"] and row["model"] == "xgboost_cnn_80_20"
        )
        stack_auc = report["evaluation"]["stacking"]["micro_roc_auc"]
        report["delta_vs_same_bases"] = stack_auc - report["evaluation"]["fixed_80_20_same_bases"]["micro_roc_auc"]
        report["delta_vs_previous_reference"] = stack_auc - report["previous_selected_budget_reference"]["micro_roc_auc"]
        print(f"Stacking AUC: {stack_auc:.6f}; same-base 80/20: {report['evaluation']['fixed_80_20_same_bases']['micro_roc_auc']:.6f}", flush=True)
        print(f"Previous selected-budget 80/20: {report['previous_selected_budget_reference']['micro_roc_auc']:.6f}", flush=True)
    with report_path.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(f"Saved {len(ids)} cases to {output}", flush=True)


if __name__ == "__main__":
    main()
