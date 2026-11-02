# Compact relaxed-state cache proposal

Owner: `tcn_audit`. Prepared after the independently verified 36-case diagnostic; no full cache or new model has run. The immutable diagnostic and all four control arms remain in `artifacts/lp-shop-relaxation-diagnostic-001/`. Independent results, including numerical and per-unit caveats, are in `docs/lp_shop_relaxation_result_audit.md`.

The raw Q-proxy screen improves from .925327975264 to .944652174548, while the zero-cost linked-state control reaches .925744293690. Thus startup costs account for most of the measured gain, but a small part is representation/tie selection rather than cost. All soft slacks are zero on these 36 cases; true float64 conservation is within 1.137e-11Mm3. On-fraction error and several units' cutoff errors worsen. None of these numbers establishes trained-model or leaderboard improvement.

The proposed cache uses only the exact already reviewed supplied-cost solver, SHA `1ac82e04d2df6571c71acfbc6be6c4ac482ea69de441471804c2b1bce1ce1aaf`, with its fixed heads, linked fractional states, nominal Pmin, original Q/P limits and all costs unchanged. Preserve the 36 verified result/replay/input files by copying their exact bytes; do not solve them again. For the remaining 3,611 cases, solve once and repeat with a fresh solver; all result arrays/scalars must agree bitwise. Preserve every raw solution, replay, assembled input and failure. A failed cache may supply completed, hash-verified cases to a new run using the identical protocol; the failed run remains unchanged.

The complete 3,647-case scope includes the provided training and test inputs. A rejecting `pandas.read_csv` guard permits only `Run No` and `starttime` from the commitment CSV. No target value is decoded. Source/raw-input/baseline/diagnostic/audit hashes and the exact IDs/dates are checked before solving. The existing 608-column feature matrix is not rebuilt locally during this cache generation.

Append exactly four dimensionless columns per unit, in the existing 14-unit order, totaling 56 new columns and **664 overall**:

| New field | Fixed formula |
| --- | --- |
| State | Linked fractional `u`, clipped to `[0,1]` only for numerical tolerance; raw state preserved. |
| Flow change | `(Q_supplied_cost - Q_cached_base) / Qmin_initial`. |
| Power change | `(P_supplied_cost - P_cached_base) / Pmax_initial_case_feasible`. |
| Economic-margin change | `(nu_cached_base - nu_supplied_cost) / (Ebest_initial * (1 + mean(abs(case_price))))`; `nu=lambda_up-lambda_down`. |

The baseline reference is the original saved v1 cache used by the existing 608 features, including its small float32 RHS effect. The diagnostic's separate float64 control showed that effect does not explain the proxy improvement. There are no new time windows, cutoff thresholds, unit subsets, state reassignments, feature selections or learned normalizers. Features are stored as float32, with raw LP outputs retained as float64.

The transform must verify cache/source/data lineage and exact case/hour/unit/column order, then append columns without changing any original value, dtype or column position. The original 521 context columns and 87 joint columns remain the exact first 608. Independent synthetic tests check the economic sign and units, common-dual cancellation, raw-array/target independence, byte-sensitive replay, ID schema and preservation of every original column. Before launch, the new feature formulas must also be computed on all 36 preserved cases and compared against their replay outputs without re-solving or loading labels.

The earlier 24.61-second diagnostic included four solves per case (fresh base, zero cost, supplied cost and its replay), plus I/O and scoring. Scaling half that average work to 3,611 new cases suggests roughly 20 minutes; allow **20–35 minutes** because supplied-cost solves can be harder than controls. This is an estimate, not a measured full-cache time. One process/one HiGHS thread occupies a single local slot, with a137MB feature array plus input tables and one case at a time. No second full-cache process is allowed for this branch.

After the cache is complete and independently verified, a separate GPU protocol should keep the accepted 608-XGBoost parameters, 600 trees and exact chronological 40% sample. The new 664-column run must have the old 608 values bitwise identical and the same labels/IDs/sample/seed. Its matched control is the already verified 608 model, so completed control training need not be repeated. No 2022 launch occurs before a separately reviewed 2021 selection lock.

Proposed model comparison, to be frozen separately before training: use the accepted 50%608-GPU projected blend as the explicit new anchor; evaluate the same global grid `[0,.1,.25,.5]` for raw664 and, as a matched control only, raw608 against that anchor. Select one664 weight on2021 with lower-weight ties. Require its score to exceed the unchanged anchor and the matched608 control at that same weight. Lock the choice and compare those same quantities on2022 without retuning. Passing the cache build alone does not authorize this model protocol, a final fit, submission or repository transfer.
