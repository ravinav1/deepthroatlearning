# CPU multilabel MLP protocol

This separate experiment uses installed scikit-learn on the CPU. The optional PyTorch/MPS branch remains blocked by the recorded package-download approval rejection; this experiment uses no GPU, new dependency, package installation or network request. The existing environment was created with uv.

Hypothesis: a shared neural representation of the same legal hydraulic inputs can provide complementary probability errors to the accepted tree ensemble. No validation score is assumed. The parent measured only a synthetic smoke test: `cpu-mlp-smoke-001`, 4096 synthetic rows ×395 columns, three epochs, 0.0415s. That measurement does not establish real-data runtime, convergence, memory use or accuracy.

Use exactly the395 normalized LP-v1 features, unchanged feature formulas and cache, and the stored40% hourly training samples from `lgb-normalized-lp-001` for each fold. These are solver inputs and input-only hydraulic relaxation outputs; no learned predictions enter the network. Predict all fourteen binary generator commitments jointly with fourteen independent logistic outputs. Hidden layers128/64 use ReLU. The network contains59,854 float32 trainable parameters.

The fixed learner is scikit-learn `MLPClassifier`: Adam, initial learning rate0.001, alpha0.0001, batch512,24epochs, shuffleTrue, random_state4173, tolerance0, n_iter_no_change1000 and early_stoppingFalse. There is no validation set, class reweighting, epoch search, learning-rate schedule search or adaptive stopping. Every expanded default, convergence warning and24-point loss curve is recorded. Limit BLAS and other native numerical pools to two threads.

For each fold, fit feature-wise float64 mean and population standard deviation(ddof0) only on the exact sampled training rows. Replace exactly zero standard deviations by scale1. Apply those frozen statistics to training and validation inputs in float64, clip standardized values to[-10,10], then store the matrix asfloat32. Save mean, original standard deviation, effective scale, feature order and sampled-row count in a new per-fold NPZ before fitting; no validation or test rows influence these statistics.

Chronological train/validation case IDs match the baseline exactly. Training starts end on December25 of the prior year, ensuring that the final167-hour target falls before January1. A local CSV-read wrapper exposes all case metadata but only actual targets strictly before2022 to the frozen feature builder. Later target rows are zero sentinels, discarded before constructing the training target matrix. The2022 validation labels and anchor NPZ are decoded only after the2021 selection lock is persisted and its checksum recorded. The wrapper does not edit any raw file or frozen feature module.

The fixed anchor is the newly accepted `lp-joint-comparison-001` candidate:18.75% LP-v1 base,56.25% LP temporal-context and25% joint-plant model, already Vinje-projected. Compare neural weights `[0, .1, .25, .5]` with

`project_vinje((1-w) * accepted_anchor_projected + w * neural_raw)`.

Promote predictions tofloat64 before mixing. Select highest whole-matrix2021 micro AUC; exact ties choose smaller neural weight. Write a new exclusive `selection_2021.json` before2022 label access. Evaluate only that locked weight and zero on2022. Accept only a nonzero weight that strictly improves both years over the frozen anchor. The anchor and both years have been repeatedly used during development; this is not an independent holdout. No per-unit selection or posthoc2022 retuning is allowed in this run. Main-repository transfer still requires at least0.998 on both years.

Save each model as a locally created trusted pickle and as portable NPZ coefficients/intercepts, feature/unit order, classes and activation names. Reconstruct every validation probability from both saved representations and require exact equality. No external pickle is loaded. Preserve full source/data/cache/sample/normalization/model hashes, exact split IDs/dates, seed, hardware/software, configuration, warnings, timing, probabilities, labels and unit diagnostics in the run manifest. Failed attempts receive a manifest and experiment-ledger entry.

Prepared command: `.venv-ml/bin/python scripts/cpu_mlp.py --run-id cpu-mlp-lp-001`. Preparation does not launch this command. The root agent controls heavy-process scheduling and must release a slot first. The script performs only chronological ablation; it never performs a full-data refit, export, submission or repository transfer.

Small-fixture checks: target-access boundary including deliberately nonnumeric withheld2022 target cells; train-only population statistics and heldout clipping; nonfinite feature rejection; exact zero-weight anchor preservation; and exact pickle/portable forward replay. Their test command is `.venv-ml/bin/python -m unittest discover -s tests -p 'test_cpu_mlp.py'`.

## Completed result: cpu-mlp-lp-001

Rejected. The fixed2021 grid selected neural weight0, retaining the accepted tree anchor exactly. No positive tested weight improved2021; no final fit or submission is warranted.

| Neural weight | 2021 projected micro AUC |
|---:|---:|
| 0.0 | 0.994888491751 |
| 0.1 | 0.994796424977 |
| 0.25 | 0.994206886767 |
| 0.5 | 0.991569944033 |

Raw neural AUC was0.975675653109 on2021 and0.973878969083 on2022. With locked weight0, accepted-anchor scores remain0.994888491751/0.993641964015. No alternative2022 weights were evaluated.

Run started`2026-11-02T12:59:10.542038+00:00`; total runtime25.903s. Actual24-epoch fit times were7.844s and9.219s. Both training loss curves declined throughout, from2.321413 to0.802232 and2.230399 to0.837942, and both produced the expected maximum-iteration convergence warning. This establishes only the failure of this fixed24-epoch candidate; declining training loss does not show that longer training would improve chronological validation.

Both folds had23 exactly zero-variance training columns, normalized with scale1. Both59,854-parameter float32 models reproduced all61,320 validation rows exactly after trusted-pickle reload and after portable NPZ reload. Independent verification additionally passed53 registered hashes, sample and normalization metadata, exact frozen blending/labels/IDs/units/AUC, and64 synthetic-row checkpoint-format replay. The independent verifier did not rebuild the large feature matrix.

Source manifest:`artifacts/cpu-mlp-lp-001/manifest.json`, SHA256`ef94fe60f828532c1d66894086a4f94fd8adcdefb4a8e1929e1a6fb1b737c30d`. Independent proof:`artifacts/verify-cpu-mlp-lp-001/manifest.json`. All source snapshots, raw data/cache hashes, full model settings, loss curves, warnings, training sample rows, normalization statistics, checkpoints and probabilities are retained. No failure occurred during either full run or verification.
