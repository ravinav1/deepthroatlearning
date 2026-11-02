# SHOP664 accepted development result and final recipe

The input-only 56-column continuous SHOP extension improves both frozen chronological comparisons at the single 2021-selected weight of 0.25. No 2022 weight search was performed.

| Development year | Accepted GPU608 anchor | Same-weight raw608 control | 25% SHOP664 primary | Standalone projected SHOP664 diagnostic |
| --- | ---: | ---: | ---: | ---: |
| 2021 | 0.9950359172801482 | 0.9950177387934181 | 0.9951018232145258 | 0.9948267633302161 |
| 2022 | 0.9937529477072502 | 0.9937356307875711 | 0.9938546734526688 | 0.9936462172415264 |

The primary formula is `project_vinje(0.75 × accepted projected GPU608 anchor + 0.25 × raw SHOP664)`, calculated in float64. Preserve the intermediate anchor projection. The separately predeclared standalone diagnostic is `project_vinje(raw SHOP664)` from the identical final checkpoints. Its development scores are lower than the primary's; producing it for a separate submission does not automatically promote it or authorize other weights.

The 2021 independent audit used a separate generic PAVA projection and exactly reproduced all four predeclared grid scores and the chosen prediction array. Proof: `artifacts/cuda-xgb664-select-2021-independent-audit-002/manifest.json`, SHA `b52056d8ed6f9619f80ee242a2c88dd72ef2bb6f4f433b1c42e8775489ffa28c`. The preceding audit 001 preserved a strict floating-point equality assertion failure of 1.11e-16 between hosted and local raw AUC; audit 002 uses the existing frozen 1e-12 contract. No candidate artifact changed.

The locked 2022 independent audit evaluated only weights zero and 0.25, verified imported artifact/source/checkpoint/configuration proofs and exactly reproduced the chosen array. Proof: `artifacts/cuda-xgb664-select-2022-independent-audit-001/manifest.json`, SHA `f79267d4e8477a4d2d88ee9b1c5fd99240e56bf4837a7a62f3a7f9cf8d0e0df4`. The unchanged 2021 lock SHA is `1783327cd4ae46182f19ee902dac606f106115d66676518e8f3004e159b6cce5`.

The root agent separately computed a descriptive paired 28-day case bootstrap on these frozen arrays, with 200 replicates and seed 4173. The primary-minus-anchor 95% intervals were approximately `[0.00002414, 0.00010614]` in 2021 and `[0.00003161, 0.00019342]` in 2022; all sampled differences were positive. Artifacts are `cuda-shop664-uncertainty-2021-001` and `cuda-shop664-uncertainty-2022-001`. This does not account for adaptive reuse of development years, training uncertainty or distribution shift, and it does not change the original acceptance rule.

The final package preserves the original 600-tree CUDA1 model parameters, all 664 columns, exact seed-4173 40% full-training sample, original three disjoint input bundles and every nested anchor value. A fresh genuine synthetic smoke must match the accepted 2022 CUDA environment and source hashes. All 14 final saved models must reproduce every one of the 121,800 test rows bitwise. The local exporter independently checks samples, source and checkpoint/configuration hashes, model parameters, physical-feature provenance, test ordering and both CSV readbacks before producing either submission.

Frozen final protocol: `docs/cuda_xgb664_final_protocol.json`, SHA `e571c6607bd797f65c76db5d7617dec7b031774a41af44e65283ccf1ad964ab7`. Private code dataset: `artifacts/cuda-xgb664-final-package-001/cuda-xgb664-final-private-dataset.zip`, SHA `c6cee8e27fa12632c3ea9726b8305f504d3d72fa1f36915cab730838869b54d4`, 81,011 bytes. Package manifest SHA: `90bad8da07acf42d710c87fa4ceccd0f0006354f6e901aff467f137ec02fcd16`. Four focused tests and exact saved-anchor reconstruction passed. Final fitting, export and leaderboard measurements are separate subsequent records; no new public score is claimed here.
