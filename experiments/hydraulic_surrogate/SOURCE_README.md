# TDT4173 hydro scheduling

Reproducible models for the supplied SINTEF hydro unit-commitment data. The objective is micro-average ROC AUC across 14 generators and 168 forecast hours per daily case.

## Setup with uv

```sh
uv venv --python 3.11 .venv-ml
uv pip install --python .venv-ml/bin/python -r pyproject.toml
.venv-ml/bin/python -m unittest discover -s tests -v
```

This workspace already has `.venv-ml`. Setup provenance and network workarounds are in [docs/environment.md](docs/environment.md). Generated artifacts, caches and environments are ignored by Git. Raw data is unchanged.

## Experiments

```sh
.venv-ml/bin/python scripts/train.py --run-id my-context-run --years 2021 2022 --feature-set full --trees 350 --sample-fraction 0.4 --threads 6
.venv-ml/bin/python scripts/train.py --run-id my-physics-run --years 2021 2022 --feature-set physics --trees 350 --sample-fraction 0.4 --threads 6
```

The runner records the command, timestamps, environment, source snapshots, file hashes, model parameters, sampled rows, train/validation case IDs, metrics and failures. Each run ID must be new. It trains one classifier per unit with fixed boosting rounds; no outer-fold early stopping. Unit prevalence is the baseline. Validation purges six days of overlapping target windows before each held-out year.

`--final` additionally fits on every supplied training case and creates a schema-checked `artifacts/<run-id>/submission.csv`. `--years` with no following years skips validation for a previously selected configuration. `--inference-from artifacts/<trained-run>` loads saved final checkpoints; pass the same `--feature-set`. Inference verifies input hashes and feature source integrity. Exact chosen commands and checksums are in [docs/best_result.md](docs/best_result.md).

## Input semantics

Features use storage only at case start, water values at start plus seven days, and the known hourly prices/inflows/constraints within the optimization horizon. Neither future realized storage, calculation time nor any commitment labels are model inputs. Timestamp joins and submission mapping are explicit. [docs/data_audit.md](docs/data_audit.md) documents the schema and chronological validation; [docs/physics_features.md](docs/physics_features.md) explains the approximate hydraulic features.

[docs/experiments.md](docs/experiments.md) records measured results. [docs/consultation.md](docs/consultation.md) records competition observations and the unavailable named GPT-6 Pro selection. Local validation scores are distinct from Kaggle scores.
