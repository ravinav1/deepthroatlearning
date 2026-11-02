# Getting the data and release artifacts

There are two separate downloads. **Raw supplied competition inputs are not in
Git or in the release assets.** The release contains trained models, generated
predictions, feature caches and experiment evidence. Validation archives may
include aligned training labels for scoring; the original supplied CSV files
remain a separate download. A fresh clone needs both raw
inputs and the `best` artifact group to run the full best-model replay.

## 1. Obtain the supplied raw data

The original import came from the course-issued **`project materials.zip`**, as
recorded in [PROJECT_FILES.md](PROJECT_FILES.md). Download that archive from the
TDT4173 course materials, or obtain an unchanged copy from a teammate with course
access. The [Kaggle competition Data tab](https://www.kaggle.com/competitions/machine-learning-task-tdt-4173-2026/data)
is the competition download entry point; sign in and accept the rules if required.
Its current file listing could not be read during migration, so these instructions
do not assume that a specifically named ZIP is hosted there. Use the supplied
files named below and let the adapter verify their hashes.

From the repository root, extract the course archive into ignored local storage:

```sh
python3 -m zipfile -e "/path/to/project materials.zip" data/interim/project-materials
```

Find the extracted `sintef_project` folder (the one containing `data/`,
`sample_submission.csv`, and `prediction_mapping.csv`), then import it:

```sh
python3 src/hydraulic_surrogate.py import-data --source data/interim/project-materials/sintef_project
```

Adjust `--source` if the ZIP has a different outer folder. On the original machine,
the existing workspace can supply the exact same files without another download:

```sh
python3 src/hydraulic_surrogate.py import-data --source /Users/heap/programming/tdt4173/sintef_project
```

The import checks every file's SHA256, refuses conflicting existing files, and
copies only the required inputs. It preserves the already tracked topology. The
resulting layout is:

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
configs/Tokke_Vinje_topology.yaml
```

Manual copying to the same layout also works. Run this afterward:

```sh
python3 src/hydraulic_surrogate.py setup
```

Expected raw-file sizes and SHA256s are in the `data` section of
[`configs/hydraulic_surrogate_best.json`](../configs/hydraulic_surrogate_best.json).
Do not substitute the missing historical `Generator_schedules_2015_2023.csv` name;
the provided target file is `Unit_commitment_decisions.csv`.

## 2. Download release assets

Release: **[Hydraulic surrogate models and research — 2026-11-02](https://github.com/ravinav1/deepthroatlearning/releases/tag/hydraulic-surrogate-20261102)**.
The assets are public and need no Kaggle credentials. Use Python 3.11 or later;
downloading and restoring uses only the standard library, so model packages are
not needed until inference or training.

| Group | Archive count | Compressed download | Contents |
| --- | ---: | ---: | --- |
| `best` | 2 | 0.352 GB | All 98 best-model checkpoints, saved predictions, required LP feature caches and evidence |
| `research` | 9 | 2.675 GB | Other trained models, validation predictions, rejected experiments, source snapshots and hosted returns |
| `physics` | 9 | 4.125 GB | Completed solver caches and explicitly incomplete integer-cache continuation state |

The default group is `best`:

```sh
python3 src/hydraulic_surrogate.py fetch-artifacts --group best
python3 src/hydraulic_surrogate.py verify --group best
```

For all useful continuation state:

```sh
python3 src/hydraulic_surrogate.py fetch-artifacts --group all
python3 src/hydraulic_surrogate.py verify --group all
```

You can also request `--group research` or `--group physics` separately. Those
groups add to `best`; they are not substitutes for it. Full restoration uses about
10 GB for extracted files plus about 7.2 GB for cached archives; leave additional
space for replay stages and new experiments. Physics includes a partial snapshot,
not an assertion that every numerical cache passed its completion gate.

Files download into `models/hydraulic_surrogate/.downloads/` and restore into
`models/hydraulic_surrogate/<original-run-id>/`. The best submitted CSV is:

```text
models/hydraulic_surrogate/submission-lgb-shop664-anchor-blend-001/submission.csv
```

Its SHA256 is
`1c1d173816127997e711d54c8269406830666af66b6f4eb6d24f4f227b70d496`.
The adapter checks archive sizes/hashes before extraction and checks every restored
file against its inventory. It refuses unexpected paths and conflicting local
files; rerunning after a download interruption safely reuses verified archives.

The small committed [artifact index](../reports/hydraulic_surrogate/artifacts.json)
contains direct asset URLs, sizes and SHA256s. Best-file hashes are in that index;
research/physics per-file inventories are separately compressed release assets,
themselves pinned by the committed index. A Git clone alone does not download them.

## Browser or GitHub CLI download alternative

Open the release link above and expand **Assets**, or use GitHub CLI:

```sh
mkdir -p models/hydraulic_surrogate/.downloads
gh release download hydraulic-surrogate-20261102 --repo ravinav1/deepthroatlearning --pattern 'hydraulic-best-*.tar.gz' --dir models/hydraulic_surrogate/.downloads
python3 src/hydraulic_surrogate.py fetch-artifacts --group best
```

For all groups, download both the `hydraulic-*.tar.gz` archives and the
`hydraulic-*-inventory.json.gz` files to that same directory. The normal adapter
command then verifies and restores the local downloads without fetching them
again. Do not merely unpack an unchecked archive over an existing experiment.

## 3. Use the recovered data and models

Create the Python 3.11 environment in [REPRODUCING.md](REPRODUCING.md), then:

```sh
.venv-hydraulic/bin/python src/hydraulic_surrogate.py export-best --run-id recovered-export-001
.venv-hydraulic/bin/python src/hydraulic_surrogate.py replay-best --run-id recovered-replay-001
```

`export-best` reconstructs from saved probabilities; `replay-best` rebuilds features
and replays all 98 checkpoints. LightGBM must match bitwise; CPU XGBoost uses an
explicit 1e-5 absolute portability bound and preserves the original hash-verified
CUDA probabilities for export. Both commands require the exact recorded best CSV
checksum. See REPRODUCING.md for the distinction and measured evidence.
Use fresh run IDs. See [HANDOFF.md](HANDOFF.md) before resuming incomplete research.
Private hosted upload bundles and duplicate staging trees were omitted; rebuild
those locally with the archived packaging code when needed. Historical Round1–Round10
artifacts not present in the source workspace are not supplied by this release;
their separate recovery/retraining instructions remain in
[PRE_HYDRAULIC_REPRODUCING.md](PRE_HYDRAULIC_REPRODUCING.md).
