# TDT4173 hydropower scheduling

Predict hourly ON/OFF probabilities for **14 hydropower generators over a
168-hour week**: **2,352 outputs per case**, evaluated with **micro ROC-AUC**.
The labels are schedules produced by an optimizer, and the model receives the
supplied prices, water values, reservoir conditions and operating constraints.

## Start here

- [Project handoff](docs/HANDOFF.md): current model, results, known weaknesses and how to continue.
- [Run on another computer](docs/REPRODUCING.md): setup, data placement and a new validation run.
- [Supplied files](docs/PROJECT_FILES.md): original task, dataset and topology documents.
- [Latest experiments](reports/round10_report_20261018.md): price ranks, temporal correction and hydraulic timing.

## Current status

Status recorded **2 November 2026**. The best user-reported public Kaggle score
in the project records is **0.99172**, for `submission_round5_lgb_generators.csv`.
Its recipe is 50% of the Round3 ensemble plus 25% each of two separate-generator
LightGBM ensembles, seeds 299 and 673.

| Candidate | Historical result | Public Kaggle result |
| --- | --- | --- |
| Round5 reference | Established comparison baseline | 0.99172, user-reported |
| Round10 price ranks | +0.00011 to +0.00016 AUC across three checks | Not recorded |
| Round10 temporal correction | +0.00002 to +0.00003 on the two 2022 checks | Not recorded |

The newer candidates are separate experiments. Historical checks reuse
validation years; their gains do not guarantee a public or private leaderboard
gain. See the [score ledger](reports/kaggle_scores.json) and
[Round10 report](reports/round10_report_20261018.md).

## First run

Use Python 3.12 and follow [REPRODUCING.md](docs/REPRODUCING.md). Start from a
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
| `configs/` | Topology and named experiment recipes |
| `evaluation/kaggle_metric.py` | Supplied metric; preserve unchanged |
| `notebooks/` | EDA and supplied visualization notebook |
| `models/`, `submissions/` | Generated checkpoints and competition CSVs |
| `reports/` | Metrics, experiment decisions, audits and small figures |
| `docs/` | Task materials, setup and handoff documentation |

Git includes code, configs, documentation, notebooks and report summaries.
Datasets, processed data, models, submissions, environments and console logs
remain local. Empty directories retain `.gitkeep` placeholders.

## Validation rules

Keep cases chronological and purge training cases whose 168-hour targets
cross the validation boundary. Construct complete-week features before
sampling training hours; evaluate every hour and generator. Fit only the
supplied labelled cases, never competition IDs or true neighbouring validation
labels. Preserve the template's case/column order and submit probabilities.

The complete earlier README is preserved in
[earlier experiment notes](docs/EXPERIMENT_HISTORY.md). Later reports under
`reports/` supersede historical statements about which model was best.
