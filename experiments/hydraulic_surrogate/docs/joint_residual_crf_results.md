# Joint residual CRF: rejected on the earlier-year screen

The prescribed 2020 screen rejected the joint residual model. All results below cover only the three Vinje generators, after the same fixed probability projection.

| Model | Pooled AUC | Interior-hour AUC | Sequence NLL/hour |
| --- | ---: | ---: | ---: |
| Unchanged 608-feature LightGBM | .9790386678233353 | .9792486227301465 | .6752701456549443 |
| Contextual unary correction | .9787562121876519 | .9789766262659656 | .6812207829740786 |
| Joint-state residual CRF | .9783957676026220 | .9786142159845935 | .5793405834628612 |

The CRF improves conditional sequence likelihood but worsens the requested ranking objective against both comparators. Excluding the first and last 12 hours does not reverse that result. All three individual projected unit AUCs also trail the unchanged baseline. No 2021/2022 model, final fit or submission is justified by this frozen screen.

The base probabilities came from nine new 350-tree models using the unchanged 608-feature recipe, three units for each of the 2018/2019/2020 purged prefixes. Every saved checkpoint reproduced its held-out probabilities exactly. Residual fitting used 724 genuinely forward-OOF cases from 2018 through 2019-12-25; its normalization also used those rows only. The screen contained 360 starts from 2020-01-01 through 2020-12-25. All 2020 and later label values were masked in the feature builder, then only the screen rows were decoded after residual fitting and screen predictions were sealed. No same-row fitted base predictions entered residual training.

The fixed joint model has 36 coefficients for pair co-activation and each unit's starts/stops, conditional on intercept, normalized price change, input-only hydraulic economic margin and time remaining. The matched unary control has 12 coefficients over the same covariates. Both used ridge .01, zero initialization, analytic gradients and deterministic L-BFGS-B. Exhaustive small-chain enumeration, finite-difference gradients, exact zero identity and case-isolation tests passed. This negative result concerns the declared small residual family and early-year transfer, not every possible structured learner. Better joint likelihood by itself does not establish better marginal ranking.

Run: `joint-residual-crf-screen-001`; runtime 464.426809 seconds, peak RSS 3,642,425,344 bytes. Manifest SHA256 `49a17b57f25d0a976a1cbfa06b2b15a036bf5155c0ecac3be9619500b38fdf4a`. Fit lock SHA256 `b824a196b689b50c994fd687c63c856f24d6e569832edf6003af69f98accfcd6`. Source/data hashes, exact case/sample IDs, checkpoints, covariates, fitted coefficients and all raw/projected predictions are retained in the run manifest. Protocol: `docs/joint_residual_crf_protocol.json`, SHA256 `aca67e40296d115cbccba113c5c55229bca74a3b57f2968be66be3e0202fdbed`.

Independent saved-array verification passed and confirmed rejection: `artifacts/joint-residual-crf-result-independent-audit-001/manifest.json`, SHA256 `11fb16eae1e5b50a3959f203f75e2b2fea4f462300cb2f4499e8faf0a015e73b`. It checked source/data/artifact hashes, exact purged samples and targets, training-only normalization, fitted objective/gradients, serialized marginal replay and all screen metrics. The reviewer did not independently rerun the nine base checkpoints; their source, saved hashes and the runner’s exact replay evidence were checked explicitly.
