# Selected-recipe migration plan

Prepared 2026-11-02; planning only. No destination, reference snapshot, remote branch, or migration bundle has been changed. The selected recipe is still pending: migration requires its measured micro ROC-AUC to reach **0.998 on both purged 2021 and 2022 holdouts**, with one frozen selection rule. A promising partial generator result does not satisfy this gate. The separate leaderboard goal is 0.999.

The destination is `ravinav1/deepthroatlearning`. The inspected read-only snapshot is `artifacts/reference_repo/deepthroatlearning-main`. Its established code lives in `src/`, JSON experiment recipes in `configs/`, original topology in `configs/Tokke_Vinje_topology.yaml`, original metric in `evaluation/kaggle_metric.py`, and raw data in `data/raw/`. Preserve its existing models, reports, ensemble recipes and official metric. Add this experiment as an independent recipe; do not replace its `src/features.py` or existing trainer.

## Minimal integration after the gate

Create an isolated clone under this writable workspace, for example `artifacts/migration_workspace/deepthroatlearning`, on a new branch based on the current remote HEAD. The existing `/Users/heap/programming/deepthroatlearning` is outside this session's writable roots and reportedly has an unborn checkout; it is not needed. Verify the remote's canonical owner/name rather than relying on its old redirected origin. User authorization to move the best experiments already covers the intended branch/draft PR; no new data request is necessary.

Stage an independent package at `experiments/hydraulic_surrogate/`, keeping the frozen relative layout (`scripts/`, `sintef_project/`, `tests/`) inside it. This avoids collisions and preserves LP source hashes. A small new launcher/config adapter can live beside the package and translate a named destination JSON recipe into the archived CLI arguments. Do not refactor the frozen feature modules merely to match the destination layout.

| Source manifest | Destination purpose |
| --- | --- |
| `sintef_project/features.py`, `physics.py`, `normalized.py`, `context_features.py` | Exact selected feature implementation; include context only when selected |
| `sintef_project/lp_features.py`, `scripts/build_lp_features.py`, `scripts/repair_lp_cache_schema.py` | Include if the selected model uses LP v1; preserve byte identity and schema-repair provenance |
| `sintef_project/lp_joint_plant.py`, `lp_head_update.py`, `lp_shared_loss_update.py`, `scripts/build_lp_joint_features.py` | Include if the joint-plant candidate is selected: the head/shared files are frozen runtime helper dependencies even though their separate iterative experiments were rejected |
| `scripts/train.py` | Archived trainer and inference entry point, with its matching source snapshot |
| `scripts/export_submission.py` or `export_ensemble.py`, and `sintef_project/postprocess.py` | Include only the selected export/ensemble dependencies and fixed weights, not every historical experiment |
| `tests/test_physics.py`, `test_normalized.py`, `test_context_features.py`, `test_lp_features.py`, `test_pipeline.py`, `test_postprocess.py` | Include tests relevant to the selected dependency closure |
| Strict scorer and `tests/test_metric.py` | Rename/package as an experimental validation module; adjust only new test imports. Never overwrite `evaluation/kaggle_metric.py` |
| New `configs/hydraulic_surrogate_<selected>.json` | Exact feature set, trees/leaves/rate, seed, sampled rows or sampling recipe, split, checkpoints, cache and frozen ensemble/postprocessing parameters |
| New report plus supporting feature documentation | Selection scores, comparisons, limitations, original manifests and artifact digests; link from handoff without rewriting historical score claims |
| New isolated requirements file | Exact versions from the selected run manifest: Python, NumPy, pandas, SciPy, LightGBM, scikit-learn, PyYAML; do not change the existing global pins without testing compatibility |

A later LP v2 or joint-plant recipe is a different dependency closure and needs its own source/cache manifest. Select it only with measured evidence; this plan does not assume its inclusion.

## Data and artifact path adapter

Use ignored, local symlinks created by the new adapter, with absolute resolved targets checked against the destination's existing data. They provide the archived package's expected paths without duplicating or editing raw CSVs:

| Archived path inside package | Existing destination target |
| --- | --- |
| `sintef_project/data/kernel/` | `data/raw/kernel/` |
| `sintef_project/data/extended/Constraint_min_volume.csv` | `data/raw/extended/Constraint_min_volume.csv` |
| `sintef_project/data/extended/Constraint_min_flow.csv` | `data/raw/extended/Constraint_min_flow.csv` |
| `sintef_project/data/extended/Tokke_Vinje_topology.yaml` | `configs/Tokke_Vinje_topology.yaml` |
| `sintef_project/prediction_mapping.csv`, `sample_submission.csv` | Their namesakes in `data/raw/` |
| `artifacts/lp-full-001/` | Ignored `data/processed/hydraulic_surrogate/lp-full-001/` |
| Other `artifacts/<run>/` | Ignored `models/hydraulic_surrogate/<run>/` or a dedicated ignored experiment artifacts directory |

Keep small structured manifests/reports in Git and large arrays, model files, raw data, submissions and logs out of Git. The destination currently does not ignore every arbitrary `artifacts/` directory, so add explicit package-local ignore rules before any staging. Transfer the complete selected cache/checkpoint manifests with their source snapshots; their validators compare relative paths and hashes. Do not rewrite recorded digests to make a moved cache pass. Either preserve the archived internal layout exactly or build a separately recorded new cache with the adapter. Existing local supplied data are sufficient; historical destination ensemble caches are necessary only if that ensemble becomes a selected component.

The trainer now compares complete reconstructed feature specifications during replay. Some specifications contain absolute cache/topology paths, so relocation needs an explicit recorded path mapping in the new launcher while preserving every non-path field and all source/data/artifact digests. This must be tested against original predictions; do not silently weaken the comparison or rewrite the archived training manifest. Keeping both original and relocated path identities in a migration report makes that exception reviewable.

## Acceptance and publication

1. Freeze the selected recipe, both complete fold scores, code/data/artifact hashes and selection history before migration. Training ends 2020-12-25 for 2021 and 2021-12-25 for 2022; validation retains every hour in the respective year.
2. Verify all raw-input hashes and the original destination metric hash. Test imports and selected unit tests in the destination environment. Preserve the current working environment; its pins differ from the destination's documented Python 3.12 stack.
3. Compare original and relocated feature columns, case IDs and exact numerical values on a small subset, then reproduce archived validation predictions from the same checkpoints. Confirm the official destination metric agrees with the experimental micro-AUC computation.
4. Recreate final probabilities from frozen checkpoints, validate 725 IDs and all 2,352 ordered prediction columns, finite values in [0,1], and record the submission digest. A new training rerun is recorded separately if library changes prevent bitwise replay.
5. Review the staged diff and ignored-file inventory. Push only the new branch and open a draft PR describing the measured result, integration boundaries and reproduction commands. Leave existing branches, recipes, reported leaderboard history and official evaluation file unchanged.

No patch, clone, branch, commit, push, or PR has been prepared by this planning task. The root coordinator decides when the threshold is satisfied and which exact recipe to transfer.
