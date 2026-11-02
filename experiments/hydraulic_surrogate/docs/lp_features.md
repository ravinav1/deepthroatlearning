# Hydraulic LP feature prototype

Status on 2026-11-02: the complete input-only cache is ready. All 3,647 cases solved successfully. This is an approximate optimization-derived feature set, not an exact SHOP reproduction or a proposed standalone submission.

## Results and artifact locations

| Run | Purpose | Result |
|---|---|---|
| `lp-benchmark-001` | First 16-case structural check | Failed before the first solve: YAML scientific notation parsed one loss coefficient as a string. Fixed explicit numeric conversion. |
| `lp-benchmark-002` | Spread-out 16-case input benchmark | Passed; 2.726 s total, median 0.08688 s/case. |
| `lp-full-001` | All training and prediction cases | Passed; 376.342 s total, median 0.08209 s/case; cache shape 3,647 × 168 × 78. |
| `lp-diagnostic-001` | Initial soft-score diagnostic | Failed before label loading: pandas 3 column-index serialization produced object-dtype metadata. |
| `lp-diagnostic-002` | Fixed monotonic LP-only ranking | Micro AUC 0.946964869 for 2021 and 0.952251082 for 2022. No fitting or tuning. |

Each run has an artifact directory with a manifest. The full cache includes code snapshots, every input CSV/YAML checksum, solver configuration, exact selected IDs/dates, raw per-case flows/storage/duals/slacks, per-case runtimes, all output checksums and environment. The machine was an Apple M2 Max MacBook Pro, 12 CPU cores and 32 GB memory; each HiGHS solve used one thread.

Full-cache maximum equality residual: 1.137e-13 Mm3. All minimum-storage, capacity, mandatory-bypass and Vest-flow slacks were zero. The separate aggregate original-input conservation diagnostic has a maximum absolute difference of 0.000239663 Mm3, caused by float32 input aggregation/first-hour RHS rounding. This tiny numerical discrepancy is recorded rather than described as exact conservation of the unrounded inputs. Promoting all balance preparation to float64 is a possible later revision; the current cache and source remain frozen for the initial model ablation.

The cache stores 268 cases containing negative-price hours; those hours use the documented linear conversion approximation below.

## Inputs and topology

`HydroCaseData` reads initial volumes at case start, terminal water values at start plus seven days, and all 168 hourly prices/inflows/constraints. It reads only `Run No` and `starttime` from the training commitment CSV. No commitment values, solver runtime, future historical volume or learned statistic enters the LP.

The 17 reservoirs are grouped into the nine generation pools already documented in `physics.py`. Turbine routes follow the plant cascade. Bypass routes remain separate, notably Kjelavatn to the Vinje pool, Botnedalsvatn directly to Bandak, and the Totak pool directly to Bandak. The Vest river's own natural inflow enters Bandak once; its minimum-flow inequality includes that natural inflow and designated Lio/bypass/spill releases. Pooling omits hydraulic restrictions within a pool, which is an explicit approximation.

## Frozen mathematical model

There are 18,984 continuous variables, 1,512 water-balance equalities and 4,872 inequalities per case. Storage is nonnegative. Each balance is `storage[t] - storage[t-1] + outgoing water - upstream water = natural inflow[t]`, with initial storage added to the first right-hand side. Variable units are Mm3 per hourly interval. Minimum/maximum storage and mandatory flow violations have expensive slacks; no water-creation slack is allowed.

Each of the 14 units has four incremental generation segments. Initial reservoir head is interpolated from individual reservoir volume. Turbine efficiency is interpolated over reference heads and discharge; generator efficiency is evaluated iteratively. Unit-alone main-tunnel and penstock losses are quadratic. The physical minimum discharge is the maximum of the power-limit-derived discharge and the turbine-curve minimum at the relevant head. The production cap and discharge-curve maximum bound the feasible curve. A concave upper hull is sampled beginning at its best-efficiency point.

At nonnegative prices, decreasing incremental energy slopes favor loading efficient segments first. At negative prices, every segment uses the same average full-discharge conversion; otherwise a continuous LP could skip efficient segments to dispose of water without corresponding production. This safeguard is a deliberate approximation rather than an exact nonconvex PQ graph.

The objective maximizes hourly market revenue plus global terminal water value minus flow/constraint costs. Global terminal value is the local EUR/MWh water value times plant best-point MWh/Mm3 plus downstream global terminal value. Bypass costs are `10/0.0036` EUR/Mm3; emergency spill costs are `1000/0.0036`; bypass flow capacity is 1,000 m3/s. Soft-constraint penalties are `1e7 + 100 * max(abs(global terminal value))` EUR/Mm3 per hour. The exact competition currency/configuration commands are unavailable; treating YAML costs consistently with price EUR is an assumption.

The entire objective is divided by `1 + mean(abs(case price))` for conditioning. Reported economic dual values are multiplied back by that scale and use the sign verified by independent finite differences. HiGHS uses a 20-second per-case time limit and primal/dual feasibility tolerances of 1e-7. No random seed, training, early stopping or ensemble weight applies to cache generation.

Omitted mechanisms include integer commitments, startup/shutdown costs, minimum run times, ramping, changing reservoir head, shared-penstock interactions, tunnel capacities, individual reservoir storage within pools, convex river cost tables, and head-dependent overflow curves. Unknown SHOP command settings are not inferred from absent YAML flags. Primary sources and the reasoning behind these choices are in `docs/web_research.md`.

## Exported features and integration

The 78 exported features are all dimensionless:

- For each of 14 units: discharge divided by physical minimum discharge, output divided by feasible maximum output, and `(price - local dual opportunity cost) / (1 + mean(abs(case price)))`.
- For each of nine pools: total plant output fraction, storage filling fraction, minimum-storage slack fraction and maximum-storage slack fraction.

`add_lp_features(x, root, cache_dir=None)` defaults to `artifacts/lp-full-001`. It verifies raw input and source checksums, the feature artifact checksum, exact case IDs and dates, complete 168-hour ordering, column order, dimensions and finite values. It returns an enriched frame and full provenance specification. Append these after the physical feature table and before optional monetary normalization. The normalizer should retain `lp_*` unchanged because these features are already dimensionless.

The verified integration is 612,696 × 381 before normalization and 612,696 × 382 afterward. The complete load/build/verify smoke took 2.809 seconds. Source checks are intentionally strict: changing `features.py`, `physics.py`, `lp_features.py` or the cache builder invalidates this frozen cache's current-source check.

## Reproduction and serialization repair

```sh
.venv-ml/bin/python -m unittest discover -s tests -p 'test_lp_features.py' -v
.venv-ml/bin/python scripts/build_lp_features.py --run-id NEW_LP_RUN --all
.venv-ml/bin/python scripts/repair_lp_cache_schema.py artifacts/NEW_LP_RUN
.venv-ml/bin/python scripts/score_lp_diagnostic.py --run-id NEW_DIAGNOSTIC_RUN --cache artifacts/NEW_LP_RUN
```

Use fresh run IDs; existing artifact directories are preserved. The repair is needed for the frozen v1 builder under pandas 3. It reads numeric arrays without pickle, rebuilds Unicode column names from the recorded manifest and verifies the numeric feature-byte hash is unchanged. The original/repaired archive hashes, command and repair script are recorded in `lp-full-001/manifest.json`. It does not rerun or modify optimization results. The corrected feature archive SHA-256 is `0ebba7d10ffb29294e36e5e8742c73165e202ac1d96e6d15fbef7ea9d7cca7b9`.

Nine independent tests pass. They cover water-value break-even and dual signs, two-reservoir terminal conversion, exact small-fixture conservation, no water creation under infeasible minimum storage, bypass destinations, negative-price segment behavior, final-hour inflow, physical curve limits, and mocked-input provenance. Arbitrary target/runtime metadata and post-start historical volumes leave the first case's LP inputs unchanged. Terminal value responds to exactly start plus seven days.

## Diagnostic interpretation

The LP-only diagnostic uses the predeclared monotone score `p = (q/Qmin)/(1 + q/Qmin)`. It is bounded but not claimed to be calibrated; its AUC is identical to ranking by `q/Qmin`. All 2021/2022 IDs and per-unit scores are preserved in `lp-diagnostic-002/manifest.json`. These scores are not directly comparable to a trained classifier's intended final performance: the experiment is to add the LP features and learn its errors.

Haukeli is weak in this approximation (unit AUC 0.857379 in 2021 and 0.711749 in 2022), while several other units exceed 0.97. Plant-total features are included to reduce sensitivity to alternative allocations among similar units. There is no measured evidence yet that adding the features improves the supervised model. Kaggle submission ID, score, rank and observation time for these LP runs: none.
