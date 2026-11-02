# GPU XGBoost and large-LP recovery

Recovered on 2026-11-02. Root owns Kaggle/browser actions and the overall best-result/handoff documents. This branch owns review of the frozen GPU XGBoost experiment and completion checks for the existing large-LP comparison. No extra heavy local model or feature process was launched.

## GPU XGBoost readiness

Independent audit `artifacts/cuda-xgb-audit-recovery-001/manifest.json` passed in 1.414 seconds. It verifies every code-package ZIP member and hash, the complete trainer/launcher/shared-loader/selector/projection source closure, the fixed earlier joint anchor and both unchanged anchor probability files, original sampled-row arrays for both years, requested learning parameters against the earlier CPU specialists, and four CPU fixtures. The audit uses chronological metadata and sampled-row arrays; it builds no feature matrix and fits no model.

The initial `uv run` command failed because the default user cache was outside writable roots. The successful command uses a task-specific writable cache and the existing uv environment; no package installation occurred:

```sh
UV_CACHE_DIR=/private/tmp/tdt4173-xgb-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python scripts/audit_cuda_xgb_recovery.py
```

That exact run ID already exists and must not be repeated. Source snapshot, environment, seeds, commands, fixture log and output checksums are retained in its artifact directory. The immutable original package remains `artifacts/cuda-xgb-package-001/cuda-xgb-code.zip` (36,324 bytes), SHA256 `29116fc622237b83065478fb407e67320dd45e263736c0af535eb0269d188139`. Protocol SHA256 is `7317e241c9a579531def94aec898b4cb3ae3192a5f5818062ee12b94db5b9b7c`.

Four hosted artifacts were prepared without launching remote work:

- `setup_cell.py` reconstructs and verifies the immutable package in `/kaggle/working/tdt4173-xgb-code-29116fc62223`. It reuses the observed offline uv environment `/kaggle/working/tdt4173-cuda-system-env` and changes no TCN files or variables.
- `smoke_cell.py` explicitly runs `cuda-xgb-smoke-001` on `cuda:1` with synthetic inputs only. Inspect the complete successful manifest before progressing. Do not rerun a completed ID.
- `fit_2021_cell.py` explicitly runs only `cuda-xgb-2021-001`, after checking the smoke and the already observed private dataset root `/kaggle/input/datasets/ravinav/tdt4173-private-hydraulic-training-inputs`.
- `cuda_xgb_review.ipynb` combines these cells with the real fold disabled by default. Add cells to the existing private notebook or use a separate private notebook; preserve the existing TCN session and cells.

After downloading all 2021 artifacts, local selection is:

```sh
UV_CACHE_DIR=/private/tmp/tdt4173-xgb-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python scripts/select_cuda_xgb_blend.py --year 2021 --gpu-run artifacts/cuda-xgb-2021-001 --protocol docs/cuda_xgb_protocol.json --run-id cuda-xgb-select-2021-001
```

The path in this command is conditional on placing the downloaded complete output directory there; no download is claimed. Selection verifies returned artifact hashes, native GPU configuration, exact chronological IDs, raw AUC and all fourteen checkpoint-replay reports. It tests weights `[0, .1, .25, .5]` only on 2021 against the predeclared `lp-joint-comparison-001` anchor, using `project_vinje((1-w)*anchor_projected+w*xgb_raw)` in float64. The anchor remains 18.75% base, 56.25% context, 25% joint even though a newer 608-feature candidate exists. The 2022 GPU command requires the resulting local `selection_2021.json` and its explicit byte SHA256. Do not train 2022 before this lock. There is no final-fit/export/submission code in this branch yet.

Frozen learning recipe: 395 LP-v1 features, exact baseline 40% samples (146,899 / 171,427 rows), 600 estimators, depth 6, learning rate .05, min child weight 20, L2=4, L1=0, max bin 128, subsample/column sample .9, seed 4173, two CPU threads, no early stopping or validation-selected checkpoint. GPU model/device compatibility still requires the actual hosted synthetic smoke; four CPU tests are not GPU evidence.

## Large-LP state

The original `lgb-normalized-lp-large-001` process and its existing `lp-large-blend-001` watcher completed before the 14:26:23 UTC observation. No duplicate was launched. The existing local heavy-process slot is now free.

The existing 2021 lock remains 25% large model plus 75% of its older context/base anchor, flattened to `[.1875, .5625, .25]` in base/context/large order. It passes its original strict improvement rule:

| Year | Locked blend | Original context/base anchor | Gain |
| --- | ---: | ---: | ---: |
| 2021 | 0.994817594504 | 0.994781111905 | 0.000036482599 |
| 2022 | 0.993542917367 | 0.993470245918 | 0.000072671449 |

Independent verification `artifacts/verify-lp-large-recovery-001/manifest.json` passed in 1.135 seconds. It hashes every comparator input and artifact, all three model-component artifacts and source snapshots, and raw data; it independently reconstructs both chosen arrays and the original anchor exactly in float64. It verifies that the 2021 lock and source/model configuration are unchanged from the snapshot taken before the 2022 fold existed. The verifier is `scripts/verify_lp_large_recovery.py` and its exact uv command, environment, case IDs and checksums are in the verification manifest.

The large blend scores below both the earlier joint blend and newer 608-feature blend on **both** years. Preserve it as an accepted ablation against its predeclared older anchor; do not final-fit it as a replacement or silently change its anchor. The standalone large model's 2021 AUC is `0.9941794368650142`, below the LP base. Full configuration, split IDs, checkpoints and data/source hashes are in its original training manifest and the comparator's saved lock. Both years are reused development validation.

No result in this recovery reaches the two-year `.998` transfer gate or establishes `.999` on Kaggle or a top-three rank. No repository transfer, Kaggle upload or submission was performed by this branch.
