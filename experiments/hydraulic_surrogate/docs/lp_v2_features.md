# Haukeli directional-storage ablation

Benchmark and full cache completed 2026-11-02. Frozen LP v1 source and cache remain unchanged. The new module is `sintef_project/lp_features_v2.py`; the benchmark is `scripts/benchmark_lp_v2.py`. The complete cache is `artifacts/lp-v2-full-001`.

The supplied topology has `Langeidvatn -> r_Langeidvatn -> Vatjern -> Haukeli`, with Langeidvatn spill also entering Vatjern. Their separate storage capacities are 32.1 and 0.3 Mm3. LP v1's freely pooled 32.4 Mm3 can incorrectly retain direct Vatjern inflow in upstream Langeidvatn. This is a physically justified correction independent of label agreement. It does not explain all observed Haukeli mistakes.

V2 splits those two storage states, inputs and capacities. The Haukeli turbine and its existing bypass/spill drain Vatjern. A new nonnegative, zero-cost river connects Langeidvatn to Vatjern; upstream emergency spill takes the same destination. The river retains the v1 generic 1,000 m3/s upper bound. Both reservoirs use the same global terminal value; their supplied local water-value columns are checked to be identical. The free forward river deliberately isolates directionality from the nonlinear river cost table, which remains omitted. The other eight pools, turbine curves and economic assumptions are unchanged.

There are 19,992 variables, 1,680 equalities and 5,376 inequalities. Split input preparation promotes balance arrays to float64, eliminating v1's first-hour RHS rounding discrepancy; other pool sums retain the existing float32 source aggregation. All 81 exported features are dimensionless: 78 v1-comparable `lp2_*` predictors plus separate Vatjern/Langeidvatn storage fractions and river discharge divided by Haukeli minimum discharge. Aggregated Haukeli fill remains available for comparison, while its unit economic margin uses the actual downstream Vatjern shadow price.

## Preselected 16-case benchmark

Run `lp-v2-benchmark-001` completed in 3.041 s, median 0.09663 s/case with one HiGHS thread. Every solve succeeded, all soft slacks were zero, and the maximum absolute aggregate conservation residual was 4.51e-12 Mm3. Feature shape is 16 × 168 × 81. The manifest records code snapshots, all CSV/YAML digests, prior-cache digests, exact inputs/order, parameters, environment and artifact hashes at `artifacts/lp-v2-benchmark-001/manifest.json`.

The eight spread-out controls were IDs 1, 418, 835, 1252, 1669, 2086, 2503 and 2922. Haukeli's predicted on-hour counts were unchanged. The other eight cases were selected from prior error analysis, so their diagnostic improvement is not an independent validation estimate.

| Run | Label on-hours | V1 on-hours | V2 on-hours | V1 false negatives | V2 false negatives | V2 false positives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2325 | 137 | 7 | 110 | 130 | 34 | 7 |
| 2326 | 151 | 14 | 104 | 137 | 47 | 0 |
| 2690 | 103 | 7 | 61 | 96 | 53 | 11 |
| 2698 | 94 | 4 | 88 | 90 | 8 | 2 |
| 2404 | 151 | 0 | 0 | 151 | 151 | 0 |
| 2653 | 168 | 0 | 0 | 168 | 168 | 0 |
| 2655 | 168 | 0 | 0 | 168 | 168 | 0 |
| 2656 | 168 | 0 | 0 | 168 | 168 | 0 |

The diagnostic above uses `q >= Qmin`, with 1e-8 numerical tolerance in the ratio, not a fitted threshold or commitment assertion. The manifest also records the fixed 0.5 threshold suggested by SHOP's documented relaxed-to-incremental rounding. No thresholds were tuned. The four wet cases reduce false negatives from 453 to 142, adding 20 false positives. The four dry exceptions retain all 655 false negatives. This supports a partial explanation and a supervised feature ablation; it does not establish a leaderboard or full-holdout gain.

Six tests pass: direct downstream inflow cannot move uphill; upstream water can move downstream without water creation; split construction leaves the base inputs unchanged; labels, solver runtime metadata and post-start volume do not affect the first case; changed cache source is rejected; and incomplete/reordered case hours are rejected. The simple fixture explicitly contrasts pooled storage retaining all water against the directed model releasing excess downstream inflow.

```sh
.venv-ml/bin/python -m unittest discover -s tests -p 'test_lp_features_v2.py' -v
.venv-ml/bin/python scripts/benchmark_lp_v2.py --run-id NEW_LP_V2_BENCHMARK
```

## Complete cache and fixed diagnostic

After root approval, `scripts/build_lp_features_v2.py --run-id lp-v2-full-001 --all` solved all3,647 cases in425.189s, median0.09185s/case. Every solve succeeded and all soft slacks were zero. Maximum aggregate conservation residual was1.47e-11Mm3. The 3,647×168×81 archive SHA256 is `5dc566d5dd2f2e07b5dc4e5ce8ec85430a45c79cf439b3ecbeff4e1a32709084`. Source, data, dates, per-case outputs and artifact hashes are frozen in `artifacts/lp-v2-full-001/manifest.json`. Unlike v1, Unicode metadata is serialized directly and no repair is needed.

`add_lp_features_v2(x, root, cache_dir=None)` verifies the v2 version then uses v1's strict input/source/artifact/case-order checks. The default cache is `artifacts/lp-v2-full-001`. A complete pipeline smoke verified exact81-column order, finiteness, base+physics+v2 shape612696×384 and normalized shape612696×385 in3.659s. Its report is `artifacts/lp-v2-full-001-loader-verification.json`. For a matched ablation, replace the v1 loader with this loader; adding both is a different larger experiment.

The untrained diagnostic remains fixed `p=(q/Qmin)/(1+q/Qmin)`, evaluated over all365 cases,168hours and14units per year. Results are in `artifacts/lp-v2-diagnostic-001/manifest.json`:

| Metric | V1 | V2 |
| --- | ---: | ---: |
| 2021 micro AUC | 0.946964869 | 0.947609525 |
| 2022 micro AUC | 0.952251082 | 0.954115118 |
| 2021 Haukeli AUC | 0.857379 | 0.882528 |
| 2022 Haukeli AUC | 0.711749 | 0.788038 |

This strengthens evidence for the routing correction; supervised gains still require measurement. No raw data, frozen v1 source, migration destination or remote repository has been changed. Migration requires at least0.998 on both chronological years under the user's latest threshold; leaderboard success remains separate.
