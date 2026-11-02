# TDT4173 hydropower scheduling

Predict hourly ON/OFF probabilities for **14 hydropower generators over a
168-hour week**: **2,352 outputs per case**, evaluated with **micro ROC-AUC**.
The labels are schedules produced by an optimizer, and the model receives the
supplied prices, water values, reservoir conditions and operating constraints.

## Start here

- [Project handoff](docs/HANDOFF.md): current model, results, known weaknesses and how to continue.
- [Data and artifact downloads](docs/DATA_AND_ARTIFACTS.md): raw-data sources, release assets, restore commands and checksums.
- [Competition goals and Codex continuation](GOALS.md): durable objective and a fresh-session starting prompt.
- [Automatic Codex startup and agents](docs/CODEX_AUTOSTART.md): Astra/ultra preference, model checks, agent roles and resource ownership.
- [Best result and exact replay](docs/best_result.md): the hydraulic LGB664–SHOP ensemble.
- [Run on another computer](docs/REPRODUCING.md): setup, data placement and a new validation run.
- [Supplied files](docs/PROJECT_FILES.md): original task, dataset and topology documents.
- [Latest experiments](reports/round10_report_20261018.md): price ranks, temporal correction and hydraulic timing.

## Current status

Latest stored observation: **2 November 2026, 22:04:37 UTC**. The best recorded
public Kaggle score is **0.99652, rank 18**, from the hydraulic LGB664–SHOP nested
ensemble. Reused purged 2021/2022 AUC is **0.995104208197 / 0.993871771073**.
Private score is unknown; first place remains the objective. All 98 component
checkpoints, predictions and useful research state are available through the
[artifact release](https://github.com/ravinav1/deepthroatlearning/releases/tag/hydraulic-surrogate-20261102).

| Candidate | Historical result | Public Kaggle result |
| --- | --- | --- |
| Hydraulic LGB664–SHOP | Purged 2021/2022: 0.99510421 / 0.99387177 | **0.99652, rank 18**, dated UI observation |
| Round5 reference | Established comparison baseline | 0.99172, user-reported |
| Round10 price ranks | +0.00011 to +0.00016 AUC across three checks | Not recorded |
| Round10 temporal correction | +0.00002 to +0.00003 on the two 2022 checks | Not recorded |

The newer candidates are separate experiments. Historical checks reuse
validation years; their gains do not guarantee a public or private leaderboard
gain. See the [score ledger](reports/kaggle_scores.json) and
[Round10 report](reports/round10_report_20261018.md).

## First run

For a fresh Codex session, use `bash scripts/codex-start.sh check`, then
`bash scripts/codex-start.sh start`. After data/artifact recovery, `improve` starts
the documented local improvement workflow. The configuration requests Astra/ultra;
the startup guide explains project trust and actual runtime verification.

For the current best, follow [REPRODUCING.md](docs/REPRODUCING.md) to set up the
separate Python 3.11 environment and restore artifacts, then run:

```sh
.venv-hydraulic/bin/python src/hydraulic_surrogate.py export-best --run-id first-export-001
.venv-hydraulic/bin/python src/hydraulic_surrogate.py replay-best --run-id first-replay-001
```

Export reconstructs the submitted CSV from saved probabilities; replay rebuilds
features and executes all 98 classifiers on CPU. Both require the original checksum.

For the existing Round10 pipeline, use Python 3.12 and start from a
**newly named copy** of `configs/round10_ranks_seed299_2021.json`, using
`src/round5_lightgbm.py`. Existing run names already have reports in Git and
will deliberately refuse to overwrite them. `src/train_baseline.py` is the
older pooled baseline, not the complete current ensemble.

A fresh clone can train new models after the supplied data is added. Replaying
the complete historical ensemble also needs ignored checkpoints or prediction
caches transferred separately, or regenerated from their component recipes.

## Repository layout

| Path | Purpose |
| --- | --- |
| `data/raw/` | Unchanged supplied CSVs; obtain separately |
| `data/interim/`, `data/processed/` | Generated data, prediction caches and intermediate files |
| `src/` | Features, trainers, evaluation and submission code |
| `experiments/hydraulic_surrogate/` | Frozen hydraulic code, tests, protocols and research history |
| `configs/` | Topology and named experiment recipes |
| `evaluation/kaggle_metric.py` | Supplied metric; preserve unchanged |
| `notebooks/` | EDA and supplied visualization notebook |
| `models/`, `submissions/` | Generated checkpoints and competition CSVs |
| `reports/` | Metrics, experiment decisions, audits and small figures |
| `docs/` | Task materials, setup and handoff documentation |

Git includes code, configs, documentation, notebooks and report summaries.
Datasets, models, submissions, environments and console logs stay out of Git.
Hydraulic checkpoints/predictions and useful solver state restore from checksum-indexed
release assets; supplied raw data is obtained separately. Root `AGENTS.md` gives
fresh Codex sessions the objective, working method, handoff and checks.

## Validation rules

Keep cases chronological and purge training cases whose 168-hour targets
cross the validation boundary. Construct complete-week features before
sampling training hours; evaluate every hour and generator. Fit only the
supplied labelled cases, never competition IDs or true neighbouring validation
labels. Preserve the template's case/column order and submit probabilities.

The complete earlier README is preserved in
[earlier experiment notes](docs/EXPERIMENT_HISTORY.md). Later reports under
`reports/` supersede historical statements about which model was best.
