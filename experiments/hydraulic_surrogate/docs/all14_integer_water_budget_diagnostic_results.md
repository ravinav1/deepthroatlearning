# All-unit integrality: completed proof, negative prediction diagnostic

The twelve fixed pre-2021 cases are now physically proven, but expanding commitment integrality from Hogga, Lio and Haukeli to all fourteen units did not improve the declared prediction proxy. Do not expand this all-unit branch to a timing pilot or full cache on this evidence. The separate three-unit cache continues unchanged. This is a compute-priority decision, not a proof that every possible learned feature derived from the all-unit solution is useless.

The original run, `artifacts/all14-integer-water-budget-twelve-001`, remains closed with its physical gate failed. Eleven cases were proven; case 2186 reached its 60-second limit with a feasible incumbent, native relative gap 1.125174082e-6 and absolute bound gap EUR 86.100112. No labels or scores were decoded in that run. Its manifest SHA256 is `f971f8def3cf7259b820402393babd08aafde7ea992441dd5cc4bdb0a3af042f`; the independent physical audit SHA256 is `0e5c05d2f6448ec7509162605c2a35ae4b47884a9deaffe1cfa34c642d062b74`.

Root separately authorized a single 600-second attempt for case 2186 under protocol `63fa814bf8db436194272bf0596ad848e841625deb7358bd081825fbbc2b7b7d`. The other eleven case directories were preserved byte for byte. The new attempt used the exact original mathematical inputs, all-unit indices, native solver settings and two one-second conditional LP calls. Independent preflight `5a58d1ca44f706f86c450078f9adcb364aebae4c8e351c9810c057a0ccf023d2` preceded execution; the entry resource check recorded 60% system-wide free memory and no swap growth.

Case 2186 reached native optimality in 72.389966 seconds, 72.651828 seconds including its saved conditional solves. Its incumbent objective remained EUR 76,521,591.76886413, identical to the prior timeout incumbent. The native relative and absolute optimality gaps became zero, with 331 branch-and-bound nodes. The conditional LP objective differs from the native upper bound by approximately EUR 5.26e-7, a numerical discrepancy that is reported separately from native optimality. Peak worker RSS was 596,705,280 bytes. Thus the extra computation certified the existing incumbent; it did not discover a better physical schedule objective.

All twelve cases passed native proven status, literal conditional-LP replay, exact zero absolute slacks, objective subset checks and the frozen 2 GiB RSS gate. The maximum RSS across the combined twelve proofs was 686,931,968 bytes. The completion coordinator took 76.160188 seconds; this excludes the earlier immutable eleven solves and the original failed attempt. The final proof hashes were sealed before the original selected targets were decoded. Every target interval ends by 2021-01-01; no 2021 or 2022 target values were parsed.

| Fixed descriptive metric | Continuous control | Three binary units | All fourteen binary units |
| --- | ---: | ---: | ---: |
| Discharge-ratio micro AUC | 0.940867220949 | 0.942198277718 | 0.938095018405 |
| Power-ratio micro AUC | 0.940021934012 | 0.941168649207 | 0.937192249458 |
| Weak-three discharge-ratio AUC | 0.897624045074 | 0.907562263522 | 0.907554337237 |
| Power-cutoff hourly errors | 2,805 | 2,643 | 2,646 |
| Weak-three hourly errors | 844 | 701 | 701 |
| Predicted on-hours | 16,539 | 16,715 | 16,624 |

There are 15,604 target on-hours, including 4,174 for the weak three units. The fixed power cutoff is 0.5 times the supplied minimum power. Proxy AUC uses the predeclared monotone discharge/minimum-discharge score. These twelve previously used development cases are a mechanistic diagnostic, not a fresh chronological model-validation estimate. No model was trained, no threshold selected, and no Kaggle result follows from this experiment.

The completed run is `artifacts/all14-integer-water-budget-completion-001/manifest.json`, SHA256 `7c3ea0d07408b0f91d7c0d34b3b877fb68c3357eae186790bdba9c555300f26d`. Its full metrics, per-unit/block counts, raw arrays, source/data/protocol hashes and exact command are preserved there. Independent result audit `artifacts/all14-integer-water-budget-completion-result-independent-audit-001/manifest.json`, SHA256 `d4bc01a1b79425bebed753d5b4b5441f9d0aa39d7d81be2c25f6c475897cd458`, verified all eleven preserved directories, the sole new solve, physical feasibility, fixed-LP KKT conditions, replay, bounds, seal ordering, selected labels, diagnostic arrays and independent rank-sum metrics without another optimization.
