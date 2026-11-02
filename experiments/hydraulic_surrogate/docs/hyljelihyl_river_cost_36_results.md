# River-cost diagnostic on 36 fixed development cases

The broader test completed in 180.48 seconds. It preserved the twelve earlier case directories byte for byte and solved the remaining 24 cases with two fresh processes for each of three arms. All 216 physical outputs passed independent primal/dual, projection, exact replay and metric verification. These are reused pre-2021 development cases, not model validation or leaderboard scores.

| Metric | A: original | B: directed, free | C: supplied river cost |
| --- | ---: | ---: | ---: |
| Discharge-ratio AUC | 0.944652174548 | 0.946370548265 | 0.947432922837 |
| Power-ratio AUC | 0.943782610203 | 0.945384826739 | 0.946289352970 |
| Cutoff errors | 7894 | 7628 | 7269 |
| Affected-five AUC | 0.944889567379 | 0.944849514395 | 0.946586276139 |
| Affected-five errors | 2832 | 2820 | 2632 |

C improves aggregate discharge and power ranking and reduces cutoff errors against both controls. Against B, the 2019 discharge AUC declines from 0.928660369803067 to 0.9282112111727606, while cutoff errors fall from 2608 to 2601. The 2015–2018 and 2020 blocks improve on both measures. This mixed result supports a bounded feature and execution pilot; it does not establish a learned-model gain or automatically authorize a full cache.

The physical formulation, cost curve, one-way routing, shared bypass cap and unbounded main-river tail remain unchanged from the 12-case v2 study. No parameter was chosen from the new results. The fixed case population comes from the earlier SHOP 36 protocol.

Actual result manifest: `artifacts/hyljelihyl-river-cost-36-diagnostic-001/manifest.json`, SHA `f403ed698047e6e47e94160fac245b019e6e286f8d9256014187bf6d1d29b4e7`. Frozen protocol SHA `380911fa5b18556cf18a734f6f38e24d3aea01b7705bdc95ed6e9e30b03966a8`; independent preflight SHA `e52b665b57f3e0b4c102ee3924d3760f4c504495b03e3dba10e1c0b855226544`.

Independent actual-result audit: `artifacts/hyljelihyl-river-cost-36-result-independent-audit-001/manifest.json`, SHA `0be8e29f96f08fe25c20a5b69182f1efff2a289e6a60ebafe715c9f13b14b85f`. Auditor: `scripts/audit_hyljelihyl_river_36_results.py`. It recomputed every reported metric using independent average-rank AUC and decoded only the 36 permitted rows after checking all physical evidence.

Next authorized preparation: 62 dimensionless C-relative-A/routing features preserving the 664-column prefix, and an input-only persistent-worker order/history pilot. Actual pilot execution requires independent protocol review. A full 3647-case cache and any supervised fit remain subsequent decisions.
