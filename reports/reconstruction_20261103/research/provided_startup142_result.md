# Supplied-only startup142 result

The frozen 25% startup142 correction passed both chronological checks and produced an independently verified final CSV. It has **not been submitted or assigned a leaderboard score by this branch**. The earlier external-solver first-place result is a separate recipe.

| Check | Clean canonical115 anchor | Frozen startup142 blend |
|---|---:|---:|
| 2021 selection | 0.9952596524075921 | 0.9953329489195831 |
| 2022 unchanged 25% check | 0.9940509264719491 | 0.9940802862111509 |

Numerical inputs are exclusively the supplied kernel/extended files, original training labels, and their audited local caches. The canonical115 prefix comes from the existing clean chain verified by `artifacts/provided-only-lineage-audit-002/manifest.json`. The 27 additions use supplied unit Pmin/Pmax, startup/shutdown costs, monetary normalization, existing switch margins, and within-case prior/future price means. Every original115 feature byte is preserved. No external outputs, teachers, pseudo-labels or additional constraints enter the model.

The recipe is `docs/provided_startup142_recipe.json`, SHA256 `63b79ceaf39181b81aff1e775408e1fb0a511554a2e8ad90945228be2c962893`. It binds all supplied data hashes, source hashes, original115 prefix manifests, clean anchors and the pre-existing economics feature specification. Source is `scripts/train_provided_startup142.py` plus `sintef_project/canonical_startup_features.py`; exact frozen snapshots accompany every run.

The model is pooled binary LightGBM: 1,400 trees, 31 leaves, learning rate .06, minimum leaf100, column fraction .9, L2=1, max_bin255, deterministic/force_col_wise, seed4173 and four threads. All other parameters are stored in the recipe. Original40% sampled case-hours are reused, expanding each to14 unit rows; unit/plant identities111/112 remain categorical. 2021 uses only cases ending before2021; 2022 uses the identical six-day purge, updated training prefix, same model and locked weight. Final training uses all2,922 provided labeled cases through2022 with the original final40% sample. No early stopping. The selection grid was `[0,.1,.25,.5,1]` on2021 only;2022 evaluated identity and unchanged.25.

Final probabilities are `project_vinje(.75 * clean_canonical115 + .25 * raw_startup142)`. Columns flatten case→unit→hour exactly as the supplied template. The projection enforces the previously accepted decreasing Vinje generator probabilities; all other unit assignments are preserved.

- Final CSV: `artifacts/provided-startup142-final-001/submission.csv`
- CSV SHA256: `498c2f7b9acd3a3974b855b816f94da4ba52ea6ce1de86153cd294cafb5b1953`
- Checkpoint: `artifacts/provided-startup142-final-001/startup142.txt`
- Checkpoint SHA256: `480c6da3ab48c17cfc433cde13a635e3e9906e41eb7145359c88997f64b4ab83`
- Final manifest SHA256: `728d0a0e046bab8867f2bc3d8b3c400d58078229c2c2f05eb2661f0bf40730f4`
- Final independent audit: `physics-provided-startup142-final-audit-001`, SHA256 `17d4fb4aab21ea4672af9c2930131e2d4fd81d29fb17d8bac39825be8ef612cb`
- Complete standalone inference replay: `physics-provided-startup142-final-inference-001`, SHA256 `3a24af261959f550cde839c991f9aa36874ba0235abeb249b41c515967d2e5e1`

The final audit reconstructed every added feature from supplied constants, verified original training labels/sample/cutoff, replayed512 checkpoint rows, and independently rebuilt selected probabilities and CSV bytes. The separate47.65-second inference run rebuilt every27-column extension and replayed all1,705,200 scalar predictions bitwise, producing the identical CSV. It relies on pinned audited original115 features and anchor rather than rebuilding that earlier chain from raw files.

Run the no-fit reproduction from the repository root, choosing a new output run ID:

```sh
.venv-ml/bin/python scripts/infer_provided_startup142.py --final-run artifacts/provided-startup142-final-001 --final-manifest-sha256 728d0a0e046bab8867f2bc3d8b3c400d58078229c2c2f05eb2661f0bf40730f4 --audit artifacts/physics-provided-startup142-final-audit-001/manifest.json --audit-sha256 17d4fb4aab21ea4672af9c2930131e2d4fd81d29fb17d8bac39825be8ef612cb --run-id provided-startup142-reproduction-NEW
```

Exact final training command:

```sh
/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/train_provided_startup142.py --recipe docs/provided_startup142_recipe.json --year final --run-id provided-startup142-final-001 --selection-lock artifacts/provided-startup142-2022-001/selection_2022.json
```

The recipe and manifests preserve source/data/feature/sample/environment hashes, complete parameters, case IDs and commands. The development years have been reused across experiments; these are chronological development results, not a fresh holdout guarantee. Unit-level gains vary: Haukeli improves in both years, while the2021 Kjela gain reverses in2022. No per-unit rescue was selected. Recorded 2026-11-03T06:11:45.857334+00:00.
