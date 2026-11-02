# Joint-plant model comparison

Protocol prepared before the new joint-plant 2021 prediction array exists. Source model run: `lgb-normalized-lp-joint-001`, with 404 normalized input features, 350 trees, the existing fixed LightGBM settings, seed 4173, 40% sampled training hours and six-day-purged chronological validation. This comparison reads saved small prediction arrays; it does not train models or reconstruct the large feature matrix.

The current accepted recipe is 75% LP temporal-context plus 25% LP-v1 base, followed by fixed Vinje G1≥G2≥G3 projection. Compare global joint-model weights `[0, .25, .5, .75, 1]` using:

`project_vinje((1-w) * (.75 * context_raw + .25 * base_raw) + w * joint_raw)`

Promote each source array to float64 before weighting. The exact floating-point implementation flattens weights to `[.25*(1-w), .75*(1-w), w]` and accumulates base, context, joint into a zero float64 array in that order, matching the generic exporter. Select the highest whole-matrix micro AUC on 2021; exact ties prefer the smaller joint weight. Write `selection_2021.json` with exclusive file creation and record its hash before decoding any 2022 prediction or label array. On 2022, evaluate only the locked blend and the fixed current/base/joint reference models, without calculating an alternate blend grid. Accept a nonzero joint weight only if the locked blend strictly improves the current recipe on 2022. Both chronological years have been repeatedly reused during development; this check is not an independent holdout.

`scripts/lp_joint_compare.py` records labels/IDs/unit alignment, exact training sample equality, source/raw data/cache/checkpoint-manifest provenance, protocol/selection hashes, selected predictions, per-unit/calendar-month/lead-day AUC, commands, environment and any failed attempt. No final fit, export, submission or main-repository transfer follows automatically. The transfer threshold remains 0.998 on both years.

The hydraulic approximation uses convex time sharing among feasible common-loading plant vertices. Its LP on fractions are physical relaxation outputs, not calibrated unit-on probabilities; q/Qmin uses the baseline solo initial-head reference. The supervised model can learn from these proxies, but neither proxy proves exact SHOP commitment recovery. Full physical qualifications are in `docs/lp_joint_plant.md`.

Run with `.venv-ml/bin/python scripts/lp_joint_compare.py --run-id lp-joint-comparison-001`. It waits for the separately controlled trainer and preserves a failed manifest if the trainer fails or the two-hour deadline expires. No dependency installation is needed; the existing environment was created with uv.

## Completed comparison: lp-joint-comparison-001

The frozen joint weight **0.25** passes the declared 2022 gate. The accepted flattened recipe is **18.75% LP base +56.25% LP context +25% joint-plant model**, followed by unchanged Vinje projection. This improves both reused chronological development years; neither reaches0.998. No full fit, export, Kaggle action or repository migration was performed by this comparison.

| Model or recipe | 2021 micro AUC | 2022 micro AUC |
|---|---:|---:|
| LP-v1 base, raw | 0.994370012291 | 0.993348542164 |
| LP-v1 base, projected | 0.994372219552 | 0.993351463186 |
| Joint model, raw | 0.994242557230 | 0.993402020319 |
| Joint model, projected | 0.994247311938 | 0.993403139230 |
| Prior accepted blend | 0.994781111905 | 0.993470245918 |
| Locked joint25% blend | 0.994888491751 | 0.993641964015 |

The standalone joint model is weaker than the prior blend on both years; the improvement comes from complementary errors in the fixed mixture. The gain over the prior blend is +0.000107379846 on2021 and +0.000171718097 on2022.

| Joint weight | 2021 selection AUC |
|---:|---:|
| 0.0 | 0.994781111905 |
| 0.25 | 0.994888491751 |
| 0.5 | 0.994858412146 |
| 0.75 | 0.994688128392 |
| 1.0 | 0.994247311938 |

Selection lock: `2026-11-02T12:35:40.315580+00:00`, SHA256 `787cf5d98dd77afdf393dbb3dbd5fbec94ddfa1836b08a1b27c458cf8cec7359`. The alternate grid was never evaluated on2022. Independent verification confirms that the lock predates creation of the joint2022 NPZ.

The selected blend improves all seven lead-day slices in both years, all twelve calendar months in2021, and eleven of twelve in2022. March2022 declines0.00002779. These slices are descriptive diagnostics, not extra tuning targets. Unit AUC improves twelve of fourteen units in each year.

| Unit | 2021 AUC delta vs prior | 2022 AUC delta vs prior |
|---|---:|---:|
| Hogga_G1 | +0.000404987 | +0.000150574 |
| Tokke_G1 | +0.000176531 | +0.000100475 |
| Tokke_G2 | +0.000029320 | +0.000085000 |
| Tokke_G3 | +0.000063829 | +0.000145425 |
| Tokke_G4 | +0.000132452 | +0.000088211 |
| Lio_G1 | +0.000055092 | -0.000118731 |
| Byrte_G1 | -0.000003424 | +0.000004275 |
| Vinje_G1 | +0.000060995 | +0.000339113 |
| Vinje_G2 | +0.000072859 | +0.000358929 |
| Vinje_G3 | +0.000549830 | +0.000601421 |
| Haukeli_G1 | -0.000132855 | -0.001248937 |
| Songa_G1 | +0.000029747 | +0.000350005 |
| Kjela_G1 | +0.000039318 | +0.000456831 |
| Vesle Kjela_G1 | +0.000003063 | +0.000108753 |

Haukeli remains a counterexample, declining in both years; Lio also declines in2022. These results did not trigger posthoc per-unit replacement or another weight search. Whole-matrix micro AUC includes cross-unit ranking pairs and is not a weighted average of the unit AUCs.

Comparison command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/lp_joint_compare.py --run-id lp-joint-comparison-001`. Runtime649.17s, mostly waiting for externally owned training. Manifests/source snapshots contain exact split IDs, dates, raw input/cache hashes, per-model settings and checkpoints, same sampled training rows, all unit/month/lead diagnostics and environment. Analysis SHA256 `6b3ae133bcc8cb0b32edaaf45d4b8937f043a3bbdc52820a4b39ff338626bf13`.

Independent verification command: `/Users/heap/programming/tdt4173/.venv-ml/bin/python scripts/verify_lp_joint_comparison.py --comparison-run lp-joint-comparison-001 --run-id verify-lp-joint-comparison-001`. It passed42 registered hashes, exact float64 reconstruction and saved probability equality, IDs/labels/units, pooled AUC, protocol/selection locks and acceptance. Proof: `artifacts/verify-lp-joint-comparison-001/manifest.json`. The three comparison tests passed: exact tie handling, refusal to replace the selection file, and exact generic-export accumulation/projection arithmetic.

A paired circular28-day block bootstrap of whole daily cases used200 replicates and seed4173. For2021 the conditional95% percentile interval of the fixed-blend gain is[0.000067736045,0.000157452225], with all200 replicates positive. Provenance:`artifacts/lp-joint-uncertainty-2021-001/manifest.json`. This descriptive interval excludes adaptive candidate selection, repeated use of development years, model-training uncertainty and cross-year distribution shift; it does not create independent validation.

The matching2022 block-bootstrap report gives observed gain0.000171718097, conditional95% interval[0.000089625474,0.000280690140], and positive fraction1.000. Same200 replicates/28-day blocks/seed4173 and the same limitations. Proof:`artifacts/lp-joint-uncertainty-2022-001/manifest.json`. These diagnostics did not change weights or the fixed acceptance gate.
