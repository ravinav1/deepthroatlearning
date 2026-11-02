# Economic threshold probe — economic-probe-001

The fixed logistic and additive-spline probe did **not** improve both validation years. Retain the physical LightGBM model. A small Haukeli logistic blend selected on 2021 improved that year's pooled AUC but reduced 2022 pooled AUC; it is rejected. No leaderboard submission was made.

## Reproduction and provenance

Command:

```sh
.venv-ml/bin/python scripts/economic_probe.py --run-id economic-probe-reproduce
```

The actual invocation used the default run ID: `.venv-ml/bin/python scripts/economic_probe.py`, with stdout/stderr saved in `artifacts/economic-probe-001.log`. Runtime was **27.329 seconds**. The manifest records the UTC timestamp, exact interpreter/command, OS/Python/package versions, machine hardware, CPU count, all input CSV/YAML SHA256 digests, exact source digests, and every artifact digest. Source snapshots are in `artifacts/economic-probe-001/source/`; fitted pipelines are `{year}_{unit}_{linear,spline}.joblib`. All eight fits converged within the fixed 400-iteration limit without warnings. No training or infrastructure failure occurred in this run.

- Full manifest: `artifacts/economic-probe-001/manifest.json`.
- Frozen choice recorded before 2022 evaluation: `artifacts/economic-probe-001/selection_2021.json`.
- Exact sampled row indexes: `train_sample_{2021,2022}.npy`.
- Predictions, labels, ordered IDs and unit names: `predictions_{2021,2022}.npz`.
- Baseline: raw `artifacts/lgb-physics-001/validation_{year}.npz`, with source-manifest and prediction-file hashes recorded in the probe manifest. No Vinje projection was applied in these comparisons.
- Kaggle submission ID, score, rank and observation time: none; not submitted.

The script imports the unchanged feature/physics builders. It adds no code to the production training/inference path. Generated models/artifacts remain outside committed source.

## Hypothesis and fixed settings

Hypothesis: simple economic thresholds expressed relative to each case's known price scale might transfer better than decision trees across the large 2021/2022 electricity-price changes. Only **Lio G1 and Haukeli G1** were fitted; the other twelve predictions stay identical to the physical LightGBM baseline.

Each case contributes 168 hourly rows. Initial reservoir volume is taken at case start; terminal water value at start+7 days; hourly price/inflow/constraints come from the documented optimizer inputs. No future historical reservoir volume or label enters any feature. The physical priors retain the approximations documented in `docs/physics_features.md`.

For both model families, monetary values are divided by `1+mean(abs(price))` over the same complete 168-hour horizon. Selected inputs include current/week/remaining price summaries, local/downstream terminal water values, local price-water margin, price/water ratios, known calendar/horizon fields, local/downstream/connected-pool filling and headroom, four minimum-volume/flow constraints, and physical head/supply/catchment/spill/switching priors. Explicit interactions multiply normalized local margin by initial head/100, initial pool filling or remaining-horizon fraction, and normalized price by head/100. The local/downstream EUR/MWh ratio is a statistical predictor, not an unconverted water-volume opportunity-cost subtraction.

Reservoir selections are fixed: Lio uses Byrtevatn, Botnedalsvatn and Bandak; Haukeli uses Vatjern, Langeidvatn and Vinjevatn. All exact feature names, formulas, upstream feature definitions and physical constants are embedded in the manifest and archived source.

- Linear model: training-fitted `StandardScaler` followed by L2 logistic regression, `C=0.1`, `solver='lbfgs'`, `max_iter=400`, `tol=1e-5`, seed 4173.
- Additive spline model: `SplineTransformer(n_knots=4, degree=3, knots='quantile', extrapolation='linear', include_bias=False)`, followed by a training-fitted `StandardScaler` and the same logistic model. Quantile knots are fitted only on the retained sampled training rows.
- Training: 80,000 case-hour rows sampled without replacement from each expanding chronological training partition, sample seed `4173+year`; two BLAS threads; no early stopping or hyperparameter search. The complete effective sklearn defaults are in the manifest.
- 2021 split: retained case starts 2015-01-01 through 2020-12-25; validation starts 2021-01-01 through 2021-12-31.
- 2022 split: retained case starts 2015-01-01 through 2021-12-25; validation starts 2022-01-01 through 2022-12-31.
- Six intervening case starts are purged at each boundary. Each last training target ends at December 31 23:00 UTC, before validation begins. Full split IDs and sampled row indexes are saved.
- Selection: for each weak unit independently, test linear/spline probability blends with weights `0, .1, .25, .5, 1` against unchanged baseline probabilities. Select highest 2021 full-matrix micro AUC, with ties resolved by first candidate. Freeze choices before reading/evaluating 2022 results. The chosen recipe is zero replacement for Lio and 10% linear logistic /90% baseline for Haukeli.

## Results

Unit AUCs for full replacements:

| Year | Unit | Physical LightGBM | Linear logistic | Additive spline |
|---|---|---:|---:|---:|
| 2021 | Lio G1 | 0.96107565 | 0.86336180 | 0.80326417 |
| 2021 | Haukeli G1 | 0.93828637 | 0.83397606 | 0.84042283 |
| 2022 | Lio G1 | 0.93615203 | 0.89057323 | 0.90454608 |
| 2022 | Haukeli G1 | 0.95575964 | 0.81986146 | 0.85442392 |

Full fourteen-unit micro AUC with one unit replaced at a time:

| Year | Baseline | Lio linear | Lio spline | Haukeli linear | Haukeli spline | Frozen combined blend |
|---|---:|---:|---:|---:|---:|---:|
| 2021 | 0.98470043 | 0.98161906 | 0.96427393 | 0.98304430 | 0.98180848 | 0.98499348 |
| 2022 | 0.98607836 | 0.98541566 | 0.98216370 | 0.98385001 | 0.98345475 | 0.98591413 |

The frozen blend's pooled gains are **+0.00029304** in 2021 and **−0.00016423** in 2022. It also reduces Haukeli's own AUC in both years, despite improving its log loss. Cross-unit probability calibration explains why the 2021 pooled result can improve while its unit ranking worsens. This is not a sufficiently robust improvement to select.

The spline extrapolation is particularly poor for Lio in 2021: log loss rises from 0.35006 to 1.95789. The models converged, so this is a modeling/generalization failure, rather than an optimizer warning.

## Regime diagnostic and next implications

The median within-case price scale increases from **28.76** in the 2021 training partition to **58.38** in 2021 validation. For the 2022 fold it rises from **31.31** in training to **180.92** in validation. These are `1+mean(abs(price))` in EUR/MWh; full min/10th/50th/90th/max quantiles are in the manifest.

Using each fold's own training 90th-percentile price scale as a diagnostic threshold:

| Year | Unit | High-scale rows | Label prevalence | Mean baseline probability | Baseline unit AUC |
|---|---|---:|---:|---:|---:|
| 2021 | Lio G1 | 50,904 | 0.51937 | 0.60764 | 0.95211778 |
| 2021 | Haukeli G1 | 50,904 | 0.32341 | 0.47664 | 0.93236440 |
| 2022 | Lio G1 | 59,640 | 0.54562 | 0.64095 | 0.93942271 |
| 2022 | Haukeli G1 | 59,640 | 0.13992 | 0.15765 | 0.95831203 |

Lio overprediction persists across both high-price partitions, around nine percentage points. Haukeli's overprediction changes from fifteen points to under two. This suggests any future cross-unit calibration must be fitted on earlier chronological out-of-fold predictions and checked for regime stability. A fixed correction inferred from the 2021 Haukeli bias is not justified by the 2022 measurements.

These results reject this particular small normalized-threshold family; they do not prove normalization or physical modeling is unhelpful. Tree normalization is being tested separately by the main experiment. More plausible remaining sources of weak-unit errors include nonlinear storage/cascade interactions and the unknown effect of upstream production, which the current cheap priors omit. That interpretation is a hypothesis, not an experimentally established cause. Do not perform further parameter selection on 2022 for this probe.
