# Fixed normalized LP capacity and coverage experiment

Run `lgb-normalized-lp-large-001` retains the frozen 395 normalized LP-v1 inputs. It tests a jointly larger tree budget, more leaves, and greater training-hour coverage. The earlier larger unnormalized physics model tested a different feature representation and does not answer this question. This experiment cannot separate the contributions of its several fixed changes.

```sh
.venv-ml/bin/python scripts/train.py --run-id lgb-normalized-lp-large-001 --years 2021 2022 --feature-set normalized-lp --trees 700 --leaves 63 --learning-rate 0.04 --min-child-samples 100 --sample-fraction 0.7 --seed 4173 --threads 2
```

There is no final fit, early stopping, or interim hyperparameter adjustment. The purged chronological case IDs match `lgb-normalized-lp-001`; sampling covers 70% of eligible training hours rather than the baseline's 40%, using the existing seed/order logic and saving exact row indices. All remaining LightGBM settings match the baseline: binary objective, column subsampling 0.9, row subsampling parameter 1, L2 penalty 1, max bin 255, deterministic column-wise training, and baseline defaults. Two threads are permitted alongside the root's joint LP fit. The past-state process ended before this model started, preserving the two-heavy-process cap.

Baseline LP350 took 1,014 seconds for both folds at four threads. No measured runtime for this larger recipe was available at declaration; the greater tree/leaf/sample budget and two-thread limit can make it a multi-hour run. Runtime remains an observed outcome, not a promised completion time.

## Blend rule fixed before either large-model fold finishes

The prior is the accepted 75% context / 25% base candidate from `lp-candidate-comparison-001`, which produced the public 0.99604 submission (root reported rank 28 after direct UI verification). That leaderboard observation does not enter the scalar selection.

Evaluate scalar large-model weights `w` in `[0, 0.25, 0.5, 0.75, 1]`. On the full 14-unit raw matrices, the coefficients in source order `[base LP, context LP, large LP]` are:

```text
[0.25 * (1-w), 0.75 * (1-w), w]
```

Apply the fixed Vinje probability projection once after this raw convex blend; do not smooth, rescale, rank-transform, choose units, or tune individual-unit weights. Weight zero must reconstruct the prior accepted projected matrix bit for bit. This operation order also matches the generic saved-array exporter.

`scripts/compare_lp_large.py --run-id lp-large-blend-001` snapshots this protocol and code before waiting for the 2021 fold. It polls metadata at 15-second intervals, then checks the model/data/feature recipe and source integrity. It opens 2021 predictions only, selects maximum pooled micro AUC, breaks exact ties toward the smaller large-model weight, and writes `selection_2021.json` containing only the 2021 fold. It records the 2021 prediction, sample, checkpoint and partial-training-manifest hashes before waiting for completed 2022 training.

The script opens 2022 predictions only after the lock exists and the source model has completed. It verifies source stability and evaluates only the frozen scalar and the fixed prior as blend candidates. If a nonzero selected weight fails to improve 2022 strictly, retain the prior; never select a different weight based on 2022. Independently report raw larger-model AUC versus the original raw LP-v1 baseline and the fixed projected larger-model AUC, without allowing these diagnostics to alter the selected recipe.

The final generic selection metadata includes `source_order`, `selected_weights`, `selected_candidate`, `accepted_candidate`, and boolean `accepted`. A failed second-fold gate keeps the originally selected weights for audit but sets `accepted:false` and the accepted candidate to the prior, preventing export by the root's guard. `selected_validation_YYYY.npz` preserves the 2021 choice; `validation_YYYY.npz` holds the accepted predictions. No full-data fit or submission is authorized by selection alone; the parent decides subsequent work.

Both years have been used repeatedly in research. Their scores are development evidence, not fresh holdout estimates. Migration still requires at least 0.998 on both chronological folds, while the overall objective remains at least 0.999 Kaggle score and a verified top-three rank.
