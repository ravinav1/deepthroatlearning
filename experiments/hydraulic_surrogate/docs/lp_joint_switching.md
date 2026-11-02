# Per-subset joint curves with fractional switching costs

This is a separately authorized prototype. Frozen joint v1, LP v1/v2 and their caches remain unchanged. `sintef_project/lp_joint_switching.py` and `scripts/benchmark_lp_joint_switching.py` implement a fixed three-arm comparison; no full cache or supervised training is authorized by this benchmark.

## Predeclared formulation

Retain the existing 128 common-discharge fractions and physical evaluation for all 15 Tokke and 7 Vinje active subsets. Compute upper and lower Q/P boundaries separately inside each subset, preserving that subset's binary unit mask. Unlike a global two-dimensional hull, this retains operating alternatives with different transition costs. Do not include an off point inside any active subset. Unused convex mass is the sole all-off representation.

For every plant/hour, nonnegative vertex weights sum to at most one. Unit on-fraction is exactly `u[j,t] = sum_v(mask[v,j] * lambda[v,t])`; it is not an independently adjustable state. Flow and power use those same weights. Every active vertex respects minimum generation, so `power[j,t] >= Pmin[j] * u[j,t]`: remaining on through a low-price hour requires physical production and water.

For hours 1 through 167, introduce nonnegative start and stop fractions satisfying `s[j,t] >= u[j,t]-u[j,t-1]` and `d[j,t] >= u[j,t-1]-u[j,t]`. Subtract `startcost_const[j]*s + stopcost_const[j]*d` from the same water-budget objective. With nonnegative costs this is the exact adjacent-hour positive/negative variation penalty of the relaxed on-fraction. Only Tokke and Vinje are changed. Other plants, initial heads, terminal values, release costs and routing remain frozen.

There is no transition charge before hour 0 or after hour 167 because those generator states were not supplied. Hour 0 still requires physical operation; free boundary states do not create a free permanently-on indicator. This is an explicit boundary assumption, not a reconstruction of unknown initial commitment. The finite-grid, fractional time-sharing formulation does not charge within-hour cycling, implement minimum up/down times or recreate an exact MIP.

SHOP documents fractional startup/shutdown costs outside MIP, conditional on the relevant commands/plant flags. The competition command sequence is unavailable, so the experiment tests that mechanism without claiming it was enabled. The topology's cost numbers are used in the market's currency, consistent with earlier priors; official attribute tables label costs NOK whereas supplied price data are EUR. No exchange-rate adjustment is inferred. [SHOP linear startup](https://docs.shop.sintef.energy/commands/set_linear_startup.html), [generator states and costs](https://docs.shop.sintef.energy/objects/generator/generator.html#initial-state)

## Protocol and checks

The exact protocol and source snapshots are written to `artifacts/lp-joint-switching-001/manifest.json` before solving or reading evaluation labels. The same 16 selected 2021 cases and 16 independent spread training cases from `lp-shared-loss-001` are used. All three arms are reported: frozen global hull, per-subset hull without switching costs, and per-subset hull with supplied costs. The unchanged primary score is `r/(1+r)` for `r=q/baseline_Qmin`; the weaker or stronger on-fraction hybrid is reported separately. No grid, threshold, scale or initial-state choice is selected from scores.

The no-cost subset model must reproduce the global-hull objective within numerical tolerance. The model with switching costs cannot exceed that objective. Water balance, generator caps, faithful on-fraction bounds, zero old joint dispatch and exact reconstructed transition cost are checked in every case. Solutions, vertices, predictions, labels and artifact hashes are saved. One HiGHS thread is used.

Six tests passed in 3.948 seconds of unittest execution: free initial/final boundaries; smoothing only through physical operation; exact fractional transition cost; negative-price forced release through a lower physical vertex; preservation of every feasible active subset; and zero-cost global/subset objective equivalence and conservation. Test command, source snapshots and output are in `artifacts/lp-joint-switching-tests-001/manifest.json` (4.912 seconds including process startup). No frozen source changed.

```sh
.venv-ml/bin/python -m unittest discover -s tests -p 'test_lp_joint_switching.py' -v
.venv-ml/bin/python scripts/benchmark_lp_joint_switching.py --run-id NEW_SUBSET_SWITCHING_BENCHMARK
```

Benchmark results will be appended after completion; none are inferred from the passing tests.

## Measured result: retain evidence, do not expand this unit-dispatch branch

`lp-joint-switching-001` completed all 32 cases in 171.634 seconds. Median total time was 5.293 seconds/case; median switching solve plus assembly was 4.343 seconds. The largest LP had 122,458 variables, compared with roughly 30,000 in the frozen global-hull model. A naive full-cache extrapolation of the switching arm alone is approximately 4.4 hours; this is an estimate, not an observed full run.

| Diagnostic | Selected 16 from 2021 | Independent 16 from 2015–2020 |
| --- | ---: | ---: |
| Frozen global-hull q-proxy micro AUC | 0.964731 | 0.951693 |
| Per-subset, no-cost q-proxy micro AUC | 0.952839 | 0.925155 |
| Per-subset, switching q-proxy micro AUC | 0.957921 | 0.940160 |
| Frozen global-hull on-fraction hybrid AUC | 0.945444 | 0.940089 |
| Per-subset, no-cost on-fraction hybrid AUC | 0.922502 | 0.880579 |
| Per-subset, switching on-fraction hybrid AUC | 0.937392 | 0.916823 |

Switching costs improve both proxies relative to the no-cost subset control, but both remain materially below the frozen global hull. No full cache is recommended on that evidence. All three arms and all per-unit results remain in the manifest; the no-cost arm is not discarded as an implementation failure.

There is a useful, separately reported positive result: plant active-unit counts improve despite weaker unit allocation.

| Count MAE, global hull → switching | Selected 16 | Independent 16 |
| --- | ---: | ---: |
| Tokke | 0.2025 → 0.1815 | 0.1450 → 0.0976 |
| Vinje | 0.1655 → 0.1570 | 0.2895 → 0.2267 |

The total absolute adjacent-hour variation of the seven unit on-fractions falls from 1,168.67 in the frozen model (1,167.74 in the no-cost subset model) to 533.68 with switching costs. Across all 32 cases, all soft slacks remain zero and the maximum aggregate conservation error is 1.21e-11 Mm3. No-cost objectives match the frozen global-hull objectives; switching objectives are lower as required. The exact penalty reconstructed from physical on-fractions matches the LP penalty.

The deterioration of the no-cost unit diagnostic despite equal financial objectives demonstrates an important allocation/timing ambiguity. In particular, Tokke G1 and G2 have exactly identical supplied generator attributes, and neither Tokke nor Vinje has a supplied priority/linear-startup flag. Different equivalent representations can select different units or move water between equal-valued hours. This is evidence that physical objective equivalence does not establish commitment-label equivalence. The original global hull's deterministic duplicate-point elimination is consequential for its unit features. No learned or assumed priority was added after observing these results.

These are small, deliberately fixed proxy benchmarks, not supervised or leaderboard results. They support preserving the global-hull candidate and rejecting automatic expansion of this slower switching branch. Any future attempt to use the count signal alone or impose a documented unit priority is a distinct experiment requiring a new protocol.
