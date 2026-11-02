# Joint plant operating-point relaxation

The32-case benchmark improved the fixed discharge diagnostic on both predetermined subsets, unlike the previous fixed-flow shared-loss construction. The root coordinator approved a full input-only cache after reviewing these results. `artifacts/lp-joint-full-001` is complete and strict-verified; its sources are frozen. No supervised model result is claimed here.

## Mathematical formulation

For Tokke's four units and Vinje's three units, enumerate all15 and7 nonempty active-unit subsets. The common discharge fractions are fixed at1/128,2/128,…,128/128. An active unit receives that fraction of the largest discharge appearing in its turbine-efficiency curves; inactive units receive zero. This exact grid was chosen before benchmark scores, without a grid search.

At each joint point, calculate effective head using total plant flow in the main tunnel and total flow through the unit's assigned penstock:

`H_eff,j = H_initial - K_main * Q_plant² - K_penstock,j * Q_penstock,j²`.

Interpolate turbine efficiency in discharge and effective head; apply the same eight generator-efficiency fixed-point steps as the frozen baseline. Reject any point violating positive head, turbine minimum/maximum flow at that head, or active-unit Pmin/Pmax. The all-off point is included. Each retained point stores total Q/P and every unit's Q/P/active indicator. This simultaneously evaluates shared losses rather than treating another unit's flow as a stale external input. Quadratic shared main-tunnel and penstock losses are supported by the supplied topology and primary SHOP documentation. [SHOP plant attributes](https://docs.shop.sintef.energy/objects/plant/plant.html#main-loss)

Construct both upper and lower convex boundaries in total discharge versus production. The lower boundary is retained for negative-price hours. Per plant/hour, assign nonnegative weights to these actual operating vertices with total weight at most one; unused mass represents all-off. Water balance receives the weighted total discharge exactly once, and market revenue receives price times weighted total production. The old independent Tokke/Vinje turbine variables have upper bound zero. All other seven plants, routing, storage constraints and initial-head terminal-value conversion remain frozen v1 behavior.

Unit discharge, production and on-fraction use the same convex weights. Their sums equal plant discharge/production. This is a continuous time-sharing approximation; fractional on indicators are not integer commitments or calibrated probabilities. Because the LP directly chooses the weights, there is no external shared-flow iteration or consistency oscillation.

## Limits

The common-load grid is a restricted set of candidate operations. Convexification then permits mixtures across discrete subsets within an hour. This combination is not an exact convex hull of every physically feasible continuous operating point, nor an exact MIP or SHOP reconstruction. Fixed initial reservoir levels, efficiency interpolation and missing startup/command-state mechanisms remain approximations. Equal-value vertices or operating combinations can produce ambiguous unit allocation. Including both hull boundaries prevents representing negative-price discharge by unsupported segment skipping, but does not remove the finite-grid restriction.

## Fixed32-case benchmark

`scripts/benchmark_lp_joint_plant.py --run-id lp-joint-plant-001` completed on2026-11-02 at11:57UTC in8.329s, median0.2021s/case, one HiGHS thread. The selected2021 group and independent16 evenly spaced2015–2020 training cases are exactly the earlier shared-loss groups. No case, grid size or threshold was selected using these new scores.

| Diagnostic | Selected16 from2021 | Independent spread16 from2015–2020 |
| --- | ---: | ---: |
| V1 fixed q/Qmin proxy micro AUC | 0.950147 | 0.932784 |
| Joint fixed q/Qmin proxy micro AUC | **0.964731** | **0.951693** |
| Joint on-fraction hybrid micro AUC | 0.945444 | 0.940089 |
| Tokke count MAE: baseline → joint | 0.3030 → 0.2025 | 0.3549 → 0.1450 |
| Vinje count MAE: baseline → joint | 0.2515 → 0.1655 | 0.5076 → 0.2895 |

Both discharge proxies use the same original initial-head Qmin denominator and fixed mapping `r/(1+r)`. The separately reported hybrid replaces joint-unit scores with mixture on-fractions, preserving the weaker result rather than selecting it away; its pooled AUC also changes score scaling. Count MAE compares sum(clipped baseline q/Qmin) with sum(joint on-fractions), without fitting a threshold.

Unit gains are not universal: on the independent subset, Tokke G3 AUC falls from0.98637 to0.97509 while the other Tokke units and all Vinje units improve. These small subsets support a supervised-feature ablation, not a leaderboard claim or full-holdout guarantee. No hidden labels were accessed.

All32 solves succeeded. All soft slacks were zero; maximum absolute water-balance residual from float64 aggregate accounting was1.21e-11Mm3. Convex weight sums never exceeded1, old independent joint-unit discharge was exactly0, unit allocation sums matched plant totals and generator caps passed. Tokke retained29–35 nonzero vertices and Vinje18–35; augmented LP sizes were27,384–30,240 variables.

Complete mathematical settings, input/source/cache hashes, selected IDs/dates, environment, raw solutions, convex weights, per-case vertex allocations and output hashes: `artifacts/lp-joint-plant-001/manifest.json`.

## Tests and implementation

Six tests pass in `tests/test_lp_joint_plant.py`: simultaneous shared-loss arithmetic; upper/lower hull coverage; physical lower-vertex selection at negative price; exact partial-time-sharing allocations; real vertex generator limits; and full-network87-column feature/count/routing checks. A first test preflight incorrectly required retaining a collinear boundary point. The fixture was corrected to make the point strictly below the edge; the implementation did not change for that correction. Failed source and explanation are preserved in `artifacts/lp-joint-tests-001/manifest.json`; the exact original start time was not captured.

The implementation is `sintef_project/lp_joint_plant.py`. `result_joint_features` exports87 dimensionless `lpjoint_*` columns:78 v1-comparable dispatch, dual-margin and storage features;7 joint-unit on-fractions;2 Tokke/Vinje active-unit counts. The baseline unit-solo energy factors remain the economic-margin denominators for comparable coordinates; these are not exact unit derivatives of joint production.

`add_lp_joint_features(x, root, cache_dir=None)` defaults to `artifacts/lp-joint-full-001`. It checks the mathematical version and87-column prefix/schema, then applies v1's full source/data/artifact/checksum/case-order verification. Replace the v1 loader for the matched ablation. The expected normalized table with the existing context features has404 columns, nine more than the395-column v1 comparator. All `lpjoint_*` quantities are dimensionless and remain unchanged by monetary normalization.

```sh
.venv-ml/bin/python -m unittest discover -s tests -p 'test_lp_joint_plant.py' -v
.venv-ml/bin/python scripts/benchmark_lp_joint_plant.py --run-id NEW_JOINT_BENCHMARK
.venv-ml/bin/python scripts/build_lp_joint_features.py --run-id NEW_JOINT_CACHE --all
```

The full-cache command was separately authorized after the benchmark. It saves both LP results and the exact retained joint-vertex arrays for every case. No v1/v2 source or cache is modified, and no supervised training is launched by this investigation.

## Complete cache and all-year diagnostics

All3,647 cases completed in792.912s, median0.18634s/case, using one solver thread. Shape:3,647×168×87. Every solve succeeded, all soft slacks were zero, and the maximum aggregate conservation residual was1.42e-9Mm3. The data include268 cases containing negative-price hours. Final feature archive SHA256: `5fd1de4d08bf5db95892dc66a870431f38c3571e1bcc65ec62cbbe8882eb5db6`. Complete provenance and raw solutions/vertices are in `artifacts/lp-joint-full-001/manifest.json`.

A memory-bounded verification loaded all612,696 rows using only hour/price base columns, checked exact feature order, source/data/artifact hashes, finiteness, normalization invariance, on-fraction bounds and plant-count sums in2.272s. It deliberately did not create another large full model matrix while other fits were active. Expected full normalized/context feature count404 is the exact231+72+13+87+1 dependency count; actual full-frame verification belongs to the trainer. Report: `artifacts/lp-joint-verification-001/manifest.json`.

The unchanged fixed discharge proxy was evaluated over all365 cases,168hours and14units in each year:

| LP-only micro AUC | 2021 | 2022 |
| --- | ---: | ---: |
| Frozen pooled v1 | 0.946965 | 0.952251 |
| Directed-Haukeli v2 | 0.947610 | 0.954115 |
| Joint Tokke/Vinje q proxy | **0.962105** | **0.960667** |
| Joint on-fraction hybrid, secondary diagnostic | 0.946685 | 0.952597 |

All results, including the weaker hybrid, are retained. These are untrained diagnostics with no fitted thresholds or calibrated probabilities, not supervised validation or leaderboard scores. Exact full-year predictions, labels, IDs, source/cache hashes and environment: `artifacts/lp-joint-diagnostic-001/manifest.json` (runtime1.329s).

Joint Tokke/Vinje gains are broad but not universal: Tokke G3 falls from0.98452 to0.98265 in2021 and0.98780 to0.98487 in2022. Haukeli remains weaker than directed v2: joint0.87770/0.73187 versus v2's0.88253/0.78804. The known dry all-on Haukeli cases2653/2655 remain zero-discharge;2656 has one nonzero hour. This joint construction corrects a different mechanism and does not replace the evidence for Haukeli directionality. Combining those physical corrections is a distinct future ablation, not an unrecorded change to this cache.

```sh
.venv-ml/bin/python scripts/verify_lp_joint_cache.py --run-id NEW_JOINT_VERIFICATION
.venv-ml/bin/python scripts/score_lp_joint_diagnostic.py --run-id NEW_JOINT_DIAGNOSTIC
```
