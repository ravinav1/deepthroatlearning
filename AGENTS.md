# Repository instructions

## Start a session

Read `README.md`, `GOALS.md`, `docs/HANDOFF.md`, and `docs/best_result.md` before
continuing competition work. Check `git status`, the artifact inventory, and actual
process/hosted-job state before starting a fit. `docs/HANDOFF.md` is the current
checkpoint; historical agent names, process IDs, browser handles, and approvals
inside archived research notes do not describe this session.

For automatic startup, model verification and reusable agent roles, follow
`docs/CODEX_AUTOSTART.md` and run `bash scripts/codex-start.sh check`. The user
authorizes independent agent deployment during improvement within the documented
resource and file-ownership limits. Prefer GPT-6 Astra with ultra reasoning when
available; verify actual settings rather than inferring them from configuration.
Preserve existing authorization and do not add a new permission round for work
already covered by the user's request or active goal.

The canonical remote is `https://github.com/ravinav1/deepthroatlearning.git`.
Fetch before integrating work; preserve concurrent changes and use normal
fast-forward pushes or conflict-resolving merges, never force-push shared history.

## Objective and improvement loop

Aim for first place in the TDT4173 2026 Kaggle competition. For a request to resume
improvement, work through research, a bounded hypothesis, implementation,
chronological validation, error analysis, and the next justified experiment.
Prioritize the best reproducible result. A baseline or small improvement does not
complete the competition objective. Persist the exact checkpoint and next action
when interrupted. The success conditions and a reusable continuation prompt are
in `GOALS.md`; a saved document does not activate a Codex thread Goal by itself.

For difficult modelling reasoning, consult **GPT-6 Pro** through computer use if
available. Verify the selected model, record the prompt and recommendations, and
experimentally check recommendations. If unavailable, record that limitation and
continue independently. A UI label of “Pro” alone does not establish GPT-6 identity.

The November 2026 migration was explicitly requested before the old 0.998 transfer
threshold was reached. It is authorized and supersedes archived “do not transfer”
notes. That threshold remains a validation target, not a prohibition on this repo.

## Preserve the two experiment families

- `src/`, existing `configs/`, and historical reports contain the Round1–Round10
  pipelines. Preserve their recipes and evidence; they remain useful comparators.
- `experiments/hydraulic_surrogate/` preserves the imported research source,
  protocols, tests, and experiment history. Hash-pinned modules and protocols must
  remain unchanged for archived replay. Put new experiments under new run names
  and versioned protocols; do not rewrite digests to pass an old audit.
- `src/hydraulic_surrogate.py` is the portable adapter. Its config is
  `configs/hydraulic_surrogate_best.json`. Data use `data/raw/`; restored checkpoints
  and predictions use ignored `models/hydraulic_surrogate/`. Downloadable release
  assets and every file hash are indexed in `reports/hydraulic_surrogate/artifacts.json`.
- Keep the supplied `evaluation/kaggle_metric.py`, topology, raw CSVs, and template
  schema unchanged. Do not replace the official metric with the experimental scorer.

## Correctness and experiment records

Use case-chronological splits. Purge the six daily starts whose 168-hour targets
overlap the held-out year; build complete-week inputs before sampling hours.
Use only legal known inputs. Never include future realized storage, solver runtime,
competition labels, or true neighbouring validation commitments. Keep every
validation hour and generator; optimize pooled micro ROC-AUC. The 2021/2022 years
are repeatedly reused development sets, not untouched tests.

Declare the split, model change, controls, selection rule, and resources before
reading results. Select on 2021, freeze the rule, and test unchanged on 2022.
Preserve each nested Vinje projection when reconstructing the best ensemble.
Require exact model/array/schema/ID/range checks before promoting a submission.
Keep final selection changes and Kaggle submissions within the current user's
authorization; migration alone does not request new submissions or training jobs.

For every run, including failures, record in `docs/experiments.md` or a linked
report: UTC time/run ID, hypothesis, revision/source snapshot, exact command,
data and artifact checksums, OS/Python/packages/hardware, seeds, splits and IDs,
feature formulas/windows, parameters, sampling, early stopping, ensemble weights,
scores/controls/runtime, and outputs. Record leaderboard ID/score/rank/time only
when actually observed; use null for unknowns. Update `docs/best_result.md`,
`reports/kaggle_scores.json`, and the handoff when evidence changes.

## Setup and checks

Follow `docs/REPRODUCING.md`. Keep the Python 3.11 hydraulic environment separate
from the existing Python 3.12 stack. Do not commit environments, credentials,
raw data, caches, checkpoints, generated CSVs, or console logs. Artifacts belong
in ignored storage and checksum-indexed release assets, not Git blobs.

Use four-space Python indentation, snake_case functions/variables, PascalCase
classes, and public-function type hints/docstrings. Run the root integration tests
and the affected archived tests; replay the best after adapter/feature changes.
Test commands and platform limitations are documented in `docs/REPRODUCING.md`.
Use imperative commit subjects and describe the concrete change, reproduction,
measured results, validation, and remaining limitations.
