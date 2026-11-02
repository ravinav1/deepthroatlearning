# Running the project on another computer

Checked against the repository and local Python 3.12.11 environment on 2026-11-02.
Start with [HANDOFF.md](HANDOFF.md) for the current model and experiment results.
Run commands from the repository root. Nothing below requires Kaggle test labels.

## Environment

```bash
git clone git@github.com:ravinav1/deepthroatlearning.git
cd deepthroatlearning
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-model.txt
python -m pip check
```

`requirements-model.txt` includes `requirements-eda.txt` and the tree/neural
libraries used by the existing scripts. Versions are pinned to the local
environment; this handoff did not test a new installation on another machine.
On macOS, a missing OpenMP runtime when importing LightGBM can require
`brew install libomp`; see the [LightGBM installation guide](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#macos).
The separate-generator experiment below runs on CPU, with two LightGBM threads.

## Obtain the supplied data separately

Git contains source, configurations, reports and the supplied documentation.
It deliberately excludes raw/processed data, trained models and submissions.
Obtain `project materials.zip` from the course materials or a teammate and copy
its contents into these locations, removing only the outer `sintef_project/`
directory. Preserve the files themselves unchanged.

```text
data/raw/
  prediction_mapping.csv
  sample_submission.csv
  kernel/
    Historical_day_ahead_price_2015_2025.csv
    Historical_volume_2015_2024.csv
    Historical_inflow_1958_2025.csv
    Synthetic_water_value_2015_2024.csv
    Unit_commitment_decisions.csv
  extended/
    Constraint_min_volume.csv
    Constraint_min_flow.csv
```

The topology is `configs/Tokke_Vinje_topology.yaml`; the official, unchanged
metric is `evaluation/kaggle_metric.py`. See [PROJECT_FILES.md](PROJECT_FILES.md)
for the original import inventory. The authored EDA notebook is
`notebooks/one_generator_one_week.ipynb`. The separate supplied visualization
notebook is preserved unchanged; it needs Plotly and has historical data paths
described in the import inventory.

Check imports and data layout without training:

```bash
python src/round5_lightgbm.py --help
PYTHONPATH=src python - <<'PY'
import pandas as pd
from features import ROOT, load_data
cases, columns, generators, inputs, topology = load_data(use_extended=True)
test = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv")
assert (len(cases), len(columns), len(generators), len(test)) == (2922, 2352, 14, 725)
assert not cases["Run No"].isin(test["Run No"]).any()
print("Data loaded; labelled and competition cases are disjoint.")
PY
```

## First independent validation run

Use the current separate-generator LightGBM recipe with the two price-rank
features. Give the copied configuration a new name: existing report names are
already occupied even on a fresh clone, while model/prediction files are absent.

```bash
python - <<'PY'
import json
from pathlib import Path
source = Path("configs/round10_ranks_seed299_2021.json")
destination = Path("configs/round10_teammate_2021_v1.json")
config = json.loads(source.read_text())
config.pop("deadline_utc", None)
assert not config.get("full_fit", False)
with destination.open("x") as handle:
    json.dump(config, handle, indent=2)
    handle.write("\n")
PY
python src/round5_lightgbm.py --config configs/round10_teammate_2021_v1.json
```

Choose `v2`, etc. for later runs; the trainer requires a `round5_` or `round10_`
prefix. It refuses existing output paths and leaves completed checkpoints if
interrupted. Keep interrupted runs and use a new name to restart.
Some older configs contain expired `deadline_utc` values: remove/reset that field
in the **new copy** before fitting. Frozen inference ignores old fitting deadlines.

This run trains 14 classifiers on 2,186 cases ending before 2021, leaving six
overlapping case starts out. It evaluates all 365 cases starting in 2021.
Weekly inputs are computed first; training then selects 42 of the 168 hours per
case/generator using the fixed rotating stride-four rule. Evaluation keeps all
168 hours. Tree counts are already fixed from earlier 2019/2020 experiments;
this configuration does not use early stopping on 2021.

Outputs use the copied config's stem:

- `models/round10_teammate_2021_v1/`: one model per generator.
- `data/processed/round10_teammate_2021_v1_validation_predictions.csv`.
- `reports/round10_teammate_2021_v1_metrics.json`: official micro ROC-AUC and provenance.

That score measures the separate-generator component, **not the full ensemble**.
For the next year, copy `configs/round10_ranks_seed299_2022.json` under another
new name. Keep settings fixed when comparing years; do not split hourly rows
randomly or train on `prediction_mapping.csv` cases.

## What is needed to reproduce the complete ensemble?

The reported Kaggle best is defined by `configs/round5_submission_lgb_generators.json`:
50% Round3 ensemble plus 25% each of separate-generator seeds 299 and 673.
The newer rank candidate uses `configs/round10_submission_ranks.json` and replaces
only those two separate-generator components. The independent run above does
not reconstruct either complete submission.

| Operation | Additional artifacts absent from Git |
| --- | --- |
| Build the Round3 branch | All six prediction CSVs listed under `sources` in `configs/round3_submission_temporal_ranks.json`, or their original model checkpoints and preprocessing state to regenerate them. |
| Build the Round5/rank candidate | Round3 submission plus both full-fit separate-generator prediction CSVs and their uniform mean; corresponding model/report artifacts for replay verification. Full-fit configs end in `_full.json`. |
| Compare against the historical ensemble | `data/processed/round6_native_2021_baseline.npz`, `round6_native_2022_baseline.npz`, `round6_frozen_2022_baseline.npz`, **and every original file listed in each cache's `metadata.source_sha256`**. Cache loading verifies those files, so transferring just the NPZ is insufficient. |
| Export the temporal candidate | `models/round10_temporal_temporal.json`, the original Round5 and Round3 submissions, the full-fit seed mean and its two source CSVs, the native-2021 baseline NPZ, and the reports/configs explicitly listed in `src/round10_temporal_export.py`. |

Ask a teammate to transfer the relevant ignored artifacts outside Git, preserving
paths, together with the matching committed reports. Alternatively, retrain the
components under new names and create new provenance reports. Exact historical
hashes are not guaranteed across environments or reruns; do not edit recorded
hashes just to make an archived replay pass.

`src/round5_average.py` averages aligned prediction files; `src/round6_evaluate.py`
compares candidates with the cached ensemble. These scripts do not produce
missing base-model predictions. `src/round10_temporal_export.py` is an archived
export with hard-coded paths and exclusive output guards, not a fresh-clone
training command; a new export needs deliberately updated output/provenance paths.

Keep new reports/configs in Git. Keep data, checkpoints and generated submissions
in their ignored directories. Never replace the original metric or raw files.
