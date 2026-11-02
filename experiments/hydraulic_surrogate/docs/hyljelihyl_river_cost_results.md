# Hyljelihyl river costs: physically verified producer result

The fixed twelve-case diagnostic completed in 88.518868 seconds. All 72 fresh processes passed the declared physical and replay checks, and the physical seal was saved before the original selected pre-2021 labels were decoded. The supplied-cost arm improves the aggregate descriptive proxy, but one date block declines. This is a development diagnostic, not chronological model validation, a leaderboard result, or automatic authorization for a full cache.

| Fixed descriptive metric | A: original continuous SHOP | B: split, free transfer | C: split, supplied river cost |
| --- | ---: | ---: | ---: |
| Discharge-ratio micro AUC | 0.940867220949 | 0.941090191937 | 0.942493269433 |
| Power-ratio micro AUC | 0.940021934012 | 0.940183507744 | 0.941409513023 |
| Power-cutoff hourly errors | 2,805 | 2,764 | 2,641 |
| Affected-five discharge-ratio AUC | 0.927779660087 | 0.927459763288 | 0.932465320497 |
| Affected-five hourly errors | 1,127 | 1,127 | 1,030 |
| Weak-three discharge-ratio AUC | 0.897624045074 | 0.897624045074 | 0.897624045074 |

The affected five are Vinje G1/G2/G3, Kjela and Vesle Kjela. C reduces their combined cutoff errors by 97 against B, including 76 fewer Kjela errors. Vinje G1 gains three errors while Vinje G2/G3 improve by six and eighteen. Tokke G1 gains nineteen errors, with mixed changes in the other Tokke units. The weak three—Hogga, Lio and Haukeli—are unchanged. Preserve these counterexamples rather than presenting the aggregate gain as universal.

| Date block | B discharge-ratio AUC | C discharge-ratio AUC | B errors | C errors |
| --- | ---: | ---: | ---: | ---: |
| 2015–2018 | 0.953385484424 | 0.954720908220 | 559 | 515 |
| 2019 | 0.934052390598 | 0.933663971754 | 843 | 843 |
| 2020 | 0.932605506724 | 0.933995324031 | 1,362 | 1,283 |

B's economic objective equals A within numerical precision on all twelve cases, while some dispatch allocations and ranks change. Its apparent improvement cannot be attributed solely to correcting infeasible water movement; alternate optima and deterministic allocation matter. The primary physical contrast is C versus B, which uses the same split feasible region and adds the complete supplied cost curve.

All C optima keep the main-river flow at or below 40 m³/s. None uses a segment above the first breakpoint, even though the implementation has an unbounded final segment and its tests explicitly permit flow above 1000 m³/s. The marginal cost jump acts as an economic threshold in this sample. The river-cost term ranges from about 177.71 to 6720 numerical currency units per case. Economic objective loss relative to B ranges from about 177.71 to 83,731.90; the difference beyond the charged river cost reflects redispatch and other opportunity costs. This tests the entire supplied curve against free transfer, not a separate attribution of the first-segment cost and later marginal jumps.

A exactly reproduces every saved original result array. Both fresh replicas of every arm agree in dtype, shape and literal bytes for all result and full primal/dual arrays. All absolute physical slacks are exactly zero. Projected B/C solutions satisfy A's original balances, inequalities, bounds and bypass minimums, with the shared bypass cap preserved. A >= B >= C holds within the frozen EUR 1e-4 allowance. Maximum worker RSS is 245,743,616 bytes and maximum solver-worker elapsed time is 0.338852 seconds; the coordinator runtime includes 72 Python launches, hashing, preflight reconstruction and diagnostics.

Maximum residuals across the 72 proofs are: equality 1.20e-13, inequality 1.94e-10, bounds 1.27e-14, stationarity 7.28e-12, dual-sign violation 1.11e-10, complementarity 1.37e-12, and scaled primal/dual objective gap 2.33e-8. All are below the frozen tolerances. These are numerical certificates for the approximate continuous model, not claims about exact SHOP 17.17 reproduction.

Run `artifacts/hyljelihyl-river-cost-v2-diagnostic-001/manifest.json`, SHA256 `fbb1ac2019fd0a8bf81695d7bc7959dc5d887c6149ecd55f74be60383e3ed401`, contains the exact command, protocol/source/data hashes, matrix signatures, case metadata, all worker proof references, physical seal and full metrics. The frozen protocol SHA256 is `ba460ccb6085f9b032ea0647e9d19879801d0095ae5eece240ffbbdc33ab91f0`; independent source/preflight approval is `6268f5155b40869074ae3e673bef2c10e973c13d0e5d03302fd6c3c8b879e203`. Independent actual-result audit passed: `artifacts/hyljelihyl-river-cost-v2-result-independent-audit-001/manifest.json`, SHA `783a1d687a3e8e4a9833c0e7b23ca0a688738784630a0ed8aef7559594a25496`. Root independently reconstructed all72 physical/KKT/projection/replay outputs and all reported rank/threshold metrics, reading only the fixed twelve pre2021 label rows after physical verification. A broader unchanged36case diagnostic is authorized for preparation only, using the original metadata-selected36case list and preserving these twelve results; real execution requires independent protocol review. No full-cache or supervised-model promotion follows automatically.
