# Reproduce on another computer

```sh
git clone https://github.com/ravinav1/deepthroatlearning.git
cd deepthroatlearning
uv venv --python 3.11 .venv-hydraulic
uv pip install --python .venv-hydraulic/bin/python -r experiments/hydraulic_surrogate/pyproject.toml
uv pip check --python .venv-hydraulic/bin/python
```

The hydraulic environment was Python 3.11.15 with the exact archived package pins.
Keep it separate from the existing Round pipeline's Python 3.12 environment and
`requirements-model.txt`. A fresh network installation was not assumed during
migration; the verification report records the actual environment tested. On macOS,
LightGBM may require a system OpenMP runtime. If using venv/pip instead of uv, install
the dependency list from the archived project file into a Python 3.11 environment.

## Download data and models

Follow **[DATA_AND_ARTIFACTS.md](DATA_AND_ARTIFACTS.md)** for raw-data sources,
exact file placement, release download commands, sizes, and integrity checks.
The short version, after placing supplied raw files, is:

```sh
.venv-hydraulic/bin/python src/hydraulic_surrogate.py setup
.venv-hydraulic/bin/python src/hydraulic_surrogate.py fetch-artifacts --group best
.venv-hydraulic/bin/python src/hydraulic_surrogate.py verify --group best
```

## Exact best export and checkpoint replay

```sh
.venv-hydraulic/bin/python src/hydraulic_surrogate.py export-best --run-id teammate-export-001
.venv-hydraulic/bin/python src/hydraulic_surrogate.py replay-best --run-id teammate-replay-001
```

Use fresh IDs each time. Export composes the saved component predictions. Replay
rebuilds the frozen feature table, executes all **98 classifiers on CPU**, compares
all component arrays, and reconstructs every nested projection and exact CSV byte.
LightGBM must match bitwise. XGBoost CPU inference is checked separately within
1e-5 maximum absolute error against the original CUDA outputs; exact export uses
those hash-verified CUDA arrays. The report distinguishes CPU portability from the
original CUDA bitwise replay; it never replaces the archived probabilities.
Allow roughly 8–12 GB of free RAM headroom; avoid another large feature build at the
same time. Timings and actual checks are recorded in
`reports/hydraulic_surrogate/migration_verification.json`.

Outputs and success/failure manifests are under
`models/hydraulic_surrogate/reproductions/<run-id>/`. Expected submission SHA256:
`1c1d173816127997e711d54c8269406830666af66b6f4eb6d24f4f227b70d496`.
The shape is 725 × 2353 including Run No, in template order, with [0,1] probabilities.
A replay mismatch must be investigated; do not revise archived hashes to hide it.

## Tests and further experiments

From the repository root:

```sh
.venv-hydraulic/bin/python -m unittest discover -s tests -v
cd experiments/hydraulic_surrogate
../../.venv-hydraulic/bin/python -m unittest discover -s tests -v
```

The imported suite expects its package root. Artifact-dependent tests require the
appropriate release groups; root adapter tests run in a source-only checkout.
Archived commands can run from the repository root through the adapter:

```sh
.venv-hydraulic/bin/python src/hydraulic_surrogate.py run train.py --help
```

Read the current handoff and protocol before long fits. Some frozen historical
resource/audit scripts call macOS `memory_pressure` or `system_profiler`, or assert
the original whole-environment identity. Preserve them and use a separately named
protocol/runner for another platform. Portable replay records relocation separately
and checks source/data/checkpoint/prediction identity without claiming identical OS.

For the existing Round10 pipeline, follow the preserved
[Python 3.12 instructions](PRE_HYDRAULIC_REPRODUCING.md). Its older full ensemble
may need historical artifacts that were absent from the hydraulic source workspace.
