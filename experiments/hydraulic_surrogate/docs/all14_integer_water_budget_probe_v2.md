# All-unit binary feasibility probe — revision2

The existing integer water-budget cache makes Hogga_G1, Lio_G1 and Haukeli_G1 binary while eleven other unit-state vectors remain continuous. This separate label-free probe changes only `unit_indices` to all fourteen original units. It retains the exact supplied-cost linked network, curves, reservoir balances, initial heads, terminal economics, state transitions, solver tolerances and fixed-state LP replay. The ongoing three-unit cache is untouched.

The parent prespecified case IDs1 and1826 from the earlier metadata-selected physical diagnostic. Their saved original continuous inputs/results are read from `artifacts/lp-shop-full-001`; their proven three-unit results/proofs come from `artifacts/integer-water-budget-completion-001`. The protocol pins every physical source, raw data file, comparison artifact and bundled solver environment. Each input is reconstructed from metadata and compared by shape, dtype and raw bytes before solving. The target CSV may only supply `Run No` and `starttime`; no commitment value is decoded, and there is no model fitting or label metric.

There is one fresh single-thread process per case, run sequentially. Each performs exactly one60-second MILP with requested zero relative gap, followed by the existing two1-second conditional LP calls. No retries, relaxed fallback or adaptive case selection occurs. Native bounds, statuses, nodes, incumbents and every returned conditional LP solution/residual/dual array are retained even on failure. Overall wall time includes input loading and process/solver overhead; the nominal solve budget is124seconds. Per-process peakRSS is measured with platform-correct `ru_maxrss` units.

Both cases must have native proven-optimal status, exact complete fixed-LP array replay, zero physical slack, objective no greater than either the original relaxed or proven three-unit optimum (the existing1e-4EUR subset tolerance), and peakRSS strictly below2GiB. Report runtime and objective/bound differences from the three-unit controls. Unproven feasible incumbents remain evidence but fail advancement. Passing permits discussion and a separately frozen/audited later diagnostic; it does not automatically authorize twelve-case labels or a full cache. Failure closes this bounded recipe without rescue.

The runner requires exact protocol and independent-preflight-proof hashes before solving. Its `--preflight` mode checks closure/reconstruction without any solve. Four focused tests verify byte identity, strict proof/resource gates and failure evidence preservation.

```sh
UV_CACHE_DIR=/private/tmp/tdt4173-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python scripts/probe_all14_integer_water_budget_v2.py --freeze-protocol
UV_CACHE_DIR=/private/tmp/tdt4173-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python scripts/probe_all14_integer_water_budget_v2.py --preflight --protocol-sha256 EXACT_SHA --run-id all14-integer-water-budget-v2-preflight-001
```

Revision2 is frozen before any competition solve. Independent source review found that a signed-maximum zero-slack check could hide a negative nonzero value. The gate now uses maximum absolute slack; a focused synthetic fixture rejects both signs. The solver, cases, inputs, costs, budgets and all other gates are unchanged. Unrun v1 source/protocol/no-solve preflight remain immutable and are pinned as provenance.
