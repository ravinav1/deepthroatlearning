# Repository Guidelines

## Competition Objective & Working Method

Aim for first place in the [TDT4173 2026 Kaggle competition](https://www.kaggle.com/competitions/machine-learning-task-tdt-4173-2026). Work continuously through research, implementation, training, validation, error analysis, and improvement. Prioritize the absolute best reproducible result over token savings; do not stop at a baseline or initial improvement. Preserve progress and next steps when interrupted.

Use computer use to consult GPT-6 Pro in the browser or desktop app for difficult reasoning. Verify the selected model; if unavailable, document the limitation and continue independent work. Record the model, prompts, recommendations, and experimentally verified decisions.

## Project Structure & Module Organization

`sintef_project/` contains `kaggle_metric.py`, `schedule_visualization.ipynb`, `sample_submission.csv`, and `prediction_mapping.csv`. Its `data/kernel/` holds historical inputs and labels; `data/extended/` holds constraints and topology. PDFs document the assignment and datasets.

## Build, Test, and Development Commands

No build system or dependency manifest exists. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas scikit-learn matplotlib plotly jupyterlab
python -m py_compile sintef_project/kaggle_metric.py
cd sintef_project
jupyter lab schedule_visualization.ipynb
```

These commands set up dependencies, check scorer syntax, and launch Jupyter. The notebook references missing `Generator_schedules_2015_2023.csv`; verify the schema before adapting its loader to `data/kernel/Unit_commitment_decisions.csv`.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions/variables, `PascalCase` classes, and public-function type hints/docstrings. No formatter or linter is configured.

## Testing Guidelines

No tests or coverage threshold exist. Add `unittest` cases in `tests/test_*.py`; run `python -m unittest discover -s tests`. Cover scorer ID alignment, invalid inputs, and expected micro-average ROC AUC. Validate chronologically without leakage.

## Experiment & Best-Result Documentation

For every run, including failures, update `docs/experiments.md` with:

- Run ID, timestamp, hypothesis, code revision/snapshot, data paths/checksums, OS/Python/package versions, hardware, seeds, and exact commands.
- Split dates/row IDs, preprocessing, feature formulas/windows, model configuration, all hyperparameters, training/early-stopping settings, and ensemble weights.
- Fold/aggregate validation scores, baseline/ablation comparisons, runtime, artifact paths/checksums, and Kaggle submission ID, score, rank, and observation time.

Maintain `docs/best_result.md` with the best run's complete data-to-submission recipe, checkpoints, inference command, submission checksum, and measured reasons for selection. Distinguish validation from leaderboard results; never invent scores. Preserve raw data, submission schema, and `[0, 1]` probabilities. Keep credentials, environments, caches, and generated artifacts out of commits; store artifacts at recorded paths.

## Commit & Pull Request Guidelines

No Git history is available. Use imperative subjects (e.g., `Improve inflow features`). PRs should include changes, reproduction commands, results, related issues, and screenshots for visualization changes.
