# Latent-shadow probability splice results

The fixed 2021 grid selected a **5%** probability splice on the three prespecified units, Haukeli_G1, Lio_G1 and Hogga_G1. Training used 2,186 cases through 2020-12-25, with six case-start days purged before the full 2021 validation year. The unchanged physical/learned family and exact four-source anchor are defined in the frozen splice protocol.

| Global weight | 2021 full 14-unit micro AUC |
| --- | ---: |
| 0 | .9949377186691472 |
| .025 | .9950257650606968 |
| .05 | **.9950284053858753** |
| .1 | .9949879208338457 |
| .25 | .9947387953562580 |

At the selected .05 weight, the constant-distribution control scored .9950173633431421 and the season-permuted control .9950167285562123. The selected real map strictly beat all comparisons, so the frozen gate passed. The total improvement over the anchor was .0000906867167281; only .0000110420427332 exceeds the constant control. Most of this gain is also attainable by smoothing over the physical schedule family. It is not evidence that the latent correction identifies the true water shadow value.

Run `latent-shadow-splice-2021-001` completed in 26.064104459 seconds with peak RSS 623,591,424 bytes. The fit lock preceded all 2021 input-grid inference, and the prediction lock preceded validation label access. Saved JSON coefficients reproduce all map distributions and raw marginal probabilities exactly. Independent verification in `artifacts/latent-shadow-splice-2021-result-audit-001/manifest.json` checked the full source/data/artifact closure, purged case membership, training-only normalization, season-preserving permutation, objectives/gradients, predictions, score grid and same-weight controls.

- Run manifest SHA256: `7fff1ceef26c384f513efa2836534ec6d67ef3f1fb66ee78b32a21c1aef33db9`.
- Selection lock SHA256: `7cea46df8d37773ed568dca0b588cec1f20bbb6a06f6d42a6719924f68c67232`.
- Independent result audit SHA256: `af1dba1873e7bad6635497364a64f17d8dffc6396fe2121815bc6a44a5dc7d7e`.

The separate 2022 confirmation protocol preserves the selected .05 weight, all three units and all model/physical hyperparameters. Its acceptance test requires strict improvement over the unchanged anchor and both same-weight controls. These reused development years do not establish leaderboard performance. No submission or final fit has been performed for this candidate.

## 2022 confirmation: rejected

The unchanged five-percent splice scored **.9937102130345562**, compared with .9936914879536517 for the anchor, .9937034478257145 for the constant control and **.9937230545979968** for the season-permuted control. Although the candidate improved the anchor by .0000187250809045, it failed the predeclared requirement to exceed both controls. No alternate weight was evaluated and no final model or submission was produced.

Training used 2,551 starts through 2021-12-25; validation used all 365 starts in 2022 and all 168 hours, with six purged start days. The run reused 2,186 targets and 359 input grids, generated six missing input grids and 365 new targets. New training labels were accessed only after checking the passing 2021 selection lock. Fit and prediction locks preceded 2022 label access. Runtime was 21.203420042 seconds; peak RSS was 592,068,608 bytes. Run manifest SHA256: `832e64f15c53d5a7f4d9b52e11ed89e6c726924367029470c863026604cf6c72`.

Independent result verification confirmed the rejection, exact target/model replay and all metrics. Proof: `artifacts/latent-shadow-splice-2022-result-audit-001/manifest.json`, SHA256 `12a8e215946f0ab1256ccc8e4c71b45ccf954eef67a613d70d1352d0dfcf738c`.

The result rejects this particular small learned map as a stable improvement over physical smoothing. It does not rule out an input-dependent economic discrepancy in a richer model. The finite-grid oracle and early-year learnability results remain valid diagnostics; they did not guarantee useful complementarity with the strong anchor.

## Residual structure and next hypothesis

A separate saved-array diagnostic allocates half of each erroneous ranking pair to each endpoint. The totals reproduce 1 minus micro AUC within 1e-13. No model was fit. In 2022, weeks with five or more true transitions comprise 37.16% of hours but 66.37% of error; in 2021 the corresponding shares are 25.58% and 48.83%. Hours within three of a true switch account for 50.69%/40.48% of error while occupying 15.12%/10.49% of hours. True episodes lasting at most 12 hours account for 39.25% of 2022 error. Constant weeks account for only 8.68% despite 39.04% of hours.

This supports investigating transition-sensitive residuals and joint unit-count sequences, particularly Vinje, whose G2/G3 errors are about two-thirds near switches in 2022. It also cautions against generic persistence smoothing, which can erase the short blocks carrying much error. Haukeli/Lio retain substantial errors away from switches and cannot be explained only by boundary timing. All truth-defined groups are descriptive; they must not enter inference features. The separate CRF protocol requires purged cross-fitted base predictions and a frozen earlier-year screen. Full tables and provenance: `artifacts/latent-shadow-residual-diagnostic-001/manifest.json`.
