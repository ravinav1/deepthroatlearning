# Hourly head and shared hydraulic loss investigation

These are independent physical ablations of frozen LP v1. No v1/v2 source, cache, raw input or trained model was changed. No full cache is authorized for these candidates. The head-only update was rejected after its benchmark; shared-loss consistency is investigated separately below.

## Evidence from all 2021 cases

`artifacts/lp-head-update-001/manifest.json` records the exact command, source snapshots, input/cache/model hashes, all365 case IDs, environment and output checksums. It completed on2026-11-02 at11:27UTC in5.076s. It reads the baseline LP's single-reservoir Bandak and Vinjevatn storage states. For each hour, interpolate water level at beginning and end storage and average the two levels. Initial storage comes only from the case start; later storage is an LP prediction. No pooled volume is allocated among constituent reservoirs and no future historical volume is read.

Hogga gross head is Bandak level minus its60m outlet. Tokke gross head is Vinjevatn level minus max(72m outlet, Bandak level). The latter formula follows SHOP's documented gross-head definition. The endpoint-average discretization is our explicit approximation, not a claim about SHOP's internal integration. [SHOP plant attributes](https://docs.shop.sintef.energy/objects/plant/plant.html)

| Across365 cases | Median | 95th percentile | Maximum |
| --- | ---: | ---: | ---: |
| Initial Hogga gross head | 12.150m | 12.350m | 12.520m |
| Maximum absolute Hogga head change within a case | 0.611m | 1.587m | 1.941m |
| Maximum relative Hogga head change | 4.97% | 13.21% | 16.03% |
| Maximum absolute Tokke head change | 1.368m | 3.429m | 3.989m |

The fixed-initial-head approximation is material for low-head Hogga. However, maximum relative head change has only0.036 correlation with case-level Hogga Brier error of the existing normalized-LP model. This descriptive association does not establish causality or prove that head is irrelevant.

For Tokke/Vinje, use the previous LP's actual unit discharge to estimate omitted shared loss:

`extra_loss_j = main_coefficient * (sum_plant(q)^2 - q_j^2) + penstock_coefficient * (sum_same_penstock(q)^2 - q_j^2)`.

The existing unit-alone curve already includes each unit's own quadratic losses. On active unit-hours, the omitted extra loss averages20.15m for Tokke (max21.504m) and5.58m for Vinje (max7.018m). Vinje's mean shared loss correlates0.356 with G3 case Brier error. This is exploratory and can be confounded by dispatch level and season. SHOP documents quadratic main-tunnel losses and shared penstocks, supporting the mechanism rather than a fitted capacity adjustment. [SHOP plant loss attributes](https://docs.shop.sintef.energy/objects/plant/plant.html#main-loss)

## Single Hogga head update: rejected

The new `sintef_project/lp_head_update.py` supports hourly generation-segment slopes and widths. Rebuild Hogga's four-segment PQ curve at each prior-LP hourly mean head, then solve the network once. Keep all other plant curves, initial volumes, inflows, constraints and the initial-head terminal-value conversion unchanged. This isolates head trajectory effects. The initial-head terminal conversion is deliberately retained because SHOP uses the initial best-point factor when none is supplied. [SHOP production factor](https://docs.shop.sintef.energy/objects/plant/plant.html#prod-factor)

The16-case benchmark contains eight evenly spaced2021 cases, four high model-error Hogga cases and four high head-change cases, with at least seven start days between added cases. Exact IDs are in the manifest. The untrained diagnostic remains the fixed monotone score `(q/Qmin)/(1+q/Qmin)`; it is not a calibrated commitment model.

| Subset / metric | Frozen v1 | One head update |
| --- | ---: | ---: |
| All16, micro AUC | 0.950147 | 0.945623 |
| All16, Hogga AUC | 0.810997 | 0.730357 |
| Eight spread controls, Hogga AUC | 0.920363 | 0.851609 |
| Four high model-error cases, Hogga AUC | 0.609434 | 0.559990 |
| Four high head-change cases, Hogga AUC | 0.366923 | 0.264615 |

Every subset worsened. The largest disagreement between the head used for the second solve and head implied by its new storage was0.698m. One fixed-profile re-solve therefore does not establish nonlinear consistency. There is no basis for expanding this candidate to a full cache now.

All slacks were zero, power caps passed and equality residuals were within solver tolerances. The original aggregate diagnostic reported up to0.000103Mm3 because it summed float32 source inputs in float32. A separate audit using float64 accumulation found a maximum true aggregate residual of7.05e-12Mm3; the original results/manifest were preserved. Audit: `artifacts/lp-head-update-001-conservation-review.json`.

Five tests cover constant versus hourly coefficient equivalence, time-varying energy dispatch, conservation, negative-price per-hour width-weighted conversion, rejection of invalid/nonconcave3D segment arrays, and legal initial/predicted boundary-level calculation. At negative prices, every segment receives that hour's capacity-weighted average energy conversion, preserving a physical linear discharge/production relation.

## Separate shared-loss update

`sintef_project/lp_shared_loss_update.py` subtracts only the formula's extra loss from the original initial gross head for Tokke and Vinje, then calls the same unit curve routine. It does not combine this with the rejected time-varying reservoir-head update. Terminal water values stay fixed. Reusing exact repeated head values avoids recomputing identical curves without rounding or tuning.

This is exact only for the value of the shared loss at the reference flows. Once dispatch changes, the loss is stale. Its dependence on another unit's discharge also introduces marginal effects on every connected unit's production, which this frozen-extra-loss construction omits. Thus it is an iterative approximation, not a full coupled nonlinear hydraulic model.

The32-case `lp-shared-loss-001` benchmark reused the same16 cases and separately evaluated16 evenly spaced training IDs from1 through2186 (2015–2020), chosen without labels or error measures. Runtime4.820s; median about0.095s/case. All slacks were zero, power caps passed and float64 conservation residual was at most1.26e-11Mm3. The first-step extra-loss mismatch after redispatch reached21.504m. The real selected cases contained no negative prices; negative-price correctness is covered by the synthetic tests rather than claimed from this sample.

| Fixed LP-only score | V1 | Shared-loss update |
| --- | ---: | ---: |
| Selected16 from2021, micro AUC | 0.950147 | 0.955552 |
| Independent spread16 from2015–2020, micro AUC | 0.932784 | 0.930452 |

Tokke G2/G4 improved strongly, while G1/G3 and Vinje G1 worsened. Shared losses change unit allocation; a favorable selected subset does not establish a general gain. Both positive and negative outcomes are preserved in `artifacts/lp-shared-loss-001/manifest.json`, with raw updated schedules and per-unit scores. Two additional tests verify no double counting of own-flow losses and recovery of original curves under zero extra loss while terminal values remain unchanged.

## Four damped shared-loss iterations: no convergence established

At the root coordinator's request, `lp-shared-loss-iterations-001` performed exactly four iterations on the same two16-case groups. The reference flow initially equals v1. After each solve, set `reference_flow = 0.5 * reference_flow + 0.5 * solved_flow`. Damping0.5 and four iterations were fixed before results; no iteration was selected by score. Initial reservoir heads and terminal values remain unchanged. All32 first iterations reproduce the prior single-update dispatch exactly.

| Iteration | Selected2021 micro AUC | Independent spread2015–2020 micro AUC | Mean loss inconsistency | Maximum inconsistency | On-state change from previous solution |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0: frozen v1 | 0.950147 | 0.932784 | — | — | — |
| 1 | 0.955552 | 0.930452 | 0.880m | 21.504m | 4.35% |
| 2 | 0.956436 | 0.933454 | 0.489m | 18.480m | 2.99% |
| 3 | 0.957346 | 0.934083 | 0.541m | 12.096m | 3.07% |
| 4 | 0.956397 | 0.933089 | 0.477m | 18.480m | 3.45% |

Loss inconsistency compares shared loss implied by the new solution against the loss used to build its curves. The mean falls, but the worst discrepancy and unit allocations oscillate. For example, case2186 alternates maximum discrepancies12.096,18.480,12.096,18.480m. Approximately3% of all unit-hour on/off diagnostics still change each step. There is no evidence of convergence within the prescribed budget. Even convergence of this construction would not repair the omitted cross-unit marginal derivatives.

Both groups' fourth-iteration diagnostic scores exceed baseline, but that does not satisfy the physical-consistency test and is not a supervised improvement. **Do not expand this iterative construction to a full cache on current evidence.** Preserve the useful finding that shared losses matter and investigate a formulation with coupled marginal effects separately if warranted. Exact iteration-by-iteration per-unit scores, schedules, reference provenance, source/data/artifact hashes and runtime are in `artifacts/lp-shared-loss-iterations-001/manifest.json`.

## Reproduction

```sh
.venv-ml/bin/python -m unittest discover -s tests -p 'test_lp_head_update.py' -v
.venv-ml/bin/python -m unittest discover -s tests -p 'test_lp_shared_loss_update.py' -v
.venv-ml/bin/python scripts/analyze_lp_head_update.py --run-id NEW_HEAD_INVESTIGATION
.venv-ml/bin/python scripts/benchmark_lp_shared_losses.py --run-id NEW_SHARED_LOSS_BENCHMARK
.venv-ml/bin/python scripts/iterate_lp_shared_losses.py --run-id NEW_SHARED_LOSS_CONSISTENCY
```

Use new artifact IDs. These scripts rely on the frozen baseline cache and model diagnostic archives identified in their manifests. No models were fitted, no thresholds tuned, and no Kaggle submissions or migration actions occurred.
