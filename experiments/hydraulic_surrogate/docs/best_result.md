# Current measured public best — 2026-11-02T22:04:37.100265+00:00

**Kaggle0.99652, rank18**, independently observed in authenticated submission details and leaderboard. Primary file `artifacts/submission-lgb-shop664-anchor-blend-001/submission.csv`, SHA256 `1c1d173816127997e711d54c8269406830666af66b6f4eb6d24f4f227b70d496`. Submission time2026-11-02T22:02:46UTC; separate observation `artifacts/submission-lgb-shop664-anchor-blend-001/kaggle_observation.json`. Submission ID unavailable in observedUI; do not invent one.

Recipe: `project_vinje(0.9*saved_projected_SHOP664_anchor +0.1*raw_LGB664)`, preserving all previous nested projections. Exact data-to-checkpoint-to-submission commands, sources, inputs, final196358-row sample and replay evidence: `docs/lgb_shop664_final_execution.md`; final protocol `docs/lgb_shop664_final_protocol.json` SHA5513e67123a968ded3790ec6bcda898d50d3682685fa7f62020ea25108e322ad; fourteen checkpoints `artifacts/lgb-shop664-final-001/`; finalmanifest2d3e2f85e40a7bbf8d849664a063a09a61eb30306746018935689754a6295ada. All14×121800test predictions independentlyreplayed, exact CSV reconstructed. Model350trees/31leaves/.06/minleaf100/L2=1/.9featurefraction/seed4173/2threads.

Purged reused-development2021/2022 AUC **.9951042081969695/.9938717710733546**. Selected over priorSHOP664 by locked2021weight and unchanged2022gain, then measured publicimprovement0.00001. Both years remain below.998; no repository transfer or top-three claim. Latestleaders.99820/.99820/.99805. Existing prospective river/vector experiments keep their declared earlierSHOP664 anchor; no retroactive change.

Earlier records below are historical.

---

# Best results and reproduction

Updated 2026-11-02. The latest active goal is **at least top three, an actual Kaggle score above 0.998, and a score exceeding first place**. The current observed first-place score is0.99820. Main-repository transfer requires **at least 0.998 on both purged chronological validation years**, as raised by the user on November 2. Neither condition is met. No experiments have been transferred to `ravinav1/deepthroatlearning`.

## Current best — SHOP664 blend, 2026-11-02

**Public AUC 0.99651, team rank18**, verified in the authenticated submission row and leaderboard and independently confirmed by the user. This improves the previous0.99640 by0.00011. Top-three/above0.998 and both chronological transfer gates remain unmet; no final-selection checkbox changed.

Reused purged development AUC: **2021 0.9951018232145258 / 2022 0.9938546734526688**. The frozen recipe is `project_vinje(0.75 * saved_projected_GPU608_blend + 0.25 * raw_SHOP664)` in float64. It preserves the entire nested anchor recipe below. The new component has664features,600trees,depth6,lr.05,seed4173 and the same196358-row40% final sample. All14 final checkpoints replayed exactly on121800 test hours, and independent array/projection/CSV reconstruction passed.

Best CSV: `artifacts/submission-cuda-xgb664-anchor-blend-001/submission.csv`, SHA256 `898d29699cda7aa93cc5b22e410d63eb7fa371ce4afc9f0e9b5a176993e9e96a`. Final archive: `artifacts/cuda-xgb664-final-001-download.zip`, SHA256 `ea8fdec6dad3baaba01b034e41ef12b8a296f98b939aa490784d9119c2be15e7`; private savedv8/scriptVersionId364985464. Strict export manifest SHA256 `dcd1617cb60aa29dce9c9a64bff16fbc25f00ef569912ea1b80c21eea5aa973c`; independent proof `artifacts/cuda-dual-final-submissions-independent-audit-001/manifest.json`, SHA256 `ff88843e5eb8aa1d351ceb36b0640afb08dfc82d2490ecba294e3dacf272aa7d`. Full frozen protocol and reproduction commands: `docs/cuda_dual_final_execution.md`. Public observations are separate `kaggle_observation.json` files; export manifests remain unchanged.

## Newly accepted validation candidate — independently audited CSV ready

The LightGBM664 ablation selected a10% blend with the saved currentSHOP664anchor on2021, then passed locked2022. Its purged development AUC is **2021 0.9951042081969695 /2022 0.9938717710733546**. Both beat identity and the same-weight originalLGB608 control, with independently rebuilt features, original sampled rows, all14 exact checkpoint replays and metric verification. The gains are small and these remain reused development years. This candidate has no observed Kaggle score and does not replace the public best above.

Formula: `project_vinje(0.9 * saved_projected_SHOP664_anchor + 0.1 * raw_LGB664)`, with every nested projection preserved. Final protocol `docs/lgb_shop664_final_protocol.json`, SHA `5513e67123a968ded3790ec6bcda898d50d3682685fa7f62020ea25108e322ad`, uses the original196358-row40% sample,350trees/31leaves/lr.06/seed4173 and2threads. Actual2022 audit SHA `35f65d33e51cd3ee88d3b477f5b7cd3fa0e02c90ad9bfaf0bd318013e8a22d76`; exact finalprotocol audit SHA `191f57b14b4bf85d4b7f54fb44806dd3131dfe28f472e9d17991fb644272897f`.

The final fit completed in1481.774seconds with5.917GB peakRSS, manifest `artifacts/lgb-shop664-final-001/manifest.json`, SHA `2d3e2f85e40a7bbf8d849664a063a09a61eb30306746018935689754a6295ada`. All14 checkpoints independently replayed every121800 test hour exactly; raw replay proof `b63b5872411fa59eaea6c7c6cbc132803b6ea3ae445c7f871b3df67875d976b8`. Ready CSV: `artifacts/submission-lgb-shop664-anchor-blend-001/submission.csv`, SHA `1c1d173816127997e711d54c8269406830666af66b6f4eb6d24f4f227b70d496`. Strict export manifest `be4ec369caf7850c1259aee81b3344a738d62c51b9c547382fed9daa1cb6247c`; exact nested-projection/CSV proof `529c5b4e7a8bf1106882de2e0594c54de2d016ec979e5e69e84a723dbcfc3c99`. Reproduction commands and full closure are in `docs/lgb_shop664_final_execution.md`. No Kaggle score is observed yet: native Brave access still fails with `cgWindowNotFound`. Public best and transfer status remain unchanged.

## Previous best — GPU608 blend, 2026-11-02

**Public AUC 0.99640, team rank20**, independently verified in authenticated Kaggle UI. First/third remain0.99820/0.99805. The primary blend improves the prior public best0.99631 by0.00009; the separately predeclared standalone diagnostic scores0.99627 and is not promoted. Goal and both0.998 validation transfer gates remain unmet. No final-selection checkbox changed.

Reused purged development AUC: **2021 0.9950359172801482 / 2022 0.9937529477072502**. Recipe is `project_vinje(0.5 * saved_projected_context_joint_anchor + 0.5 * raw_gpu608)` in float64, preserving the intermediate anchor projection. The GPU component uses600trees,depth6,learningrate.05,seed4173,the exact40% training-hour sample and608features. All14 final checkpoints and the independent array/CSV reconstruction replay exactly. Full data-to-submission recipe, input/source/sample hashes and inference/export command: `docs/cuda_xgb608_final.md`.

Best file: `artifacts/submission-cuda-xgb608-context-joint-blend-001/submission.csv`, SHA256 `729ce3e406560b30ab867ad6e0f448650a304168e56293d3d681c5fd7d638c76`. Final checkpoint archive: `artifacts/cuda-xgb608-final-003-download.zip`, SHA256 `83a0d528cfcf3817e35c060f7446532f017ec2a92a0138db920f58fa9ed0bf3b`; fit332.17s. Proof: `artifacts/cuda-xgb608-submissions-independent-audit-001/manifest.json`. Actual submission observations are saved separately as `kaggle_observation.json` in each export directory and `docs/kaggle_scores.json`; frozen export manifests are unchanged.

## Accepted validation comparisons — 2026-11-02

Both candidates use `project_vinje(0.75 * saved_current_GPU608_blend + 0.25 * new_raw_model)` with the nested anchor projection preserved. The weight was locked on2021 before2022; both independently beat the frozen anchor and same-weight raw608 control on2022. Both final fits and independent exports completed; SHOP is the measured public best above. Capacity scored0.99644, improving the older0.99640 anchor but below SHOP0.99651; it is not promoted.

| Candidate | Purged2021 AUC | Purged2022 AUC | Independent2022 proof |
| --- | --- | --- | --- |
| SHOP664 continuous-state/startup features | 0.9951018232145258 | 0.9938546734526688 | `artifacts/cuda-xgb664-select-2022-independent-audit-001/manifest.json` |
| Capacity608,1200trees/depth8/lr.03 | 0.9950519105980167 | 0.9937720670635095 | `artifacts/cuda-xgb608-capacity-2022-independent-audit-001/manifest.json` |

Paired28-day whole-case bootstrap,200replicates/seed4173, gives descriptive95% gain intervals for SHOP of [0.00002414,0.00010614] / [0.00003161,0.00019342], and capacity of [-0.00000881,0.00005452] / [-0.00002830,0.00006105]. These condition on fixed predictions and do not correct reused-year/adaptive selection, training variability or shift. Full evidence: `artifacts/cuda-shop664-uncertainty-{2021,2022}-001/manifest.json` and `artifacts/cuda-capacity608-uncertainty-{2021,2022}-001/manifest.json`. No rule or blend was changed using these descriptive checks. Both remain below the transfer gates.

## Previous accepted candidate — recovery 2026-11-02

**Public AUC 0.99631; then-observed rank23 independently reconfirmed in authenticated Kaggle UI during the evening continuation.** First place0.99820, third0.99805. Validation remains reused development evidence: **2021 0.994937718669 / 2022 0.993691487954**. Both project gates remain unmet. The accepted GPU608 final completed as private saved notebook version4, scriptVersionId364965805, run cuda-xgb608-final-003; it now supersedes this result as described above. Prior interactive recovery attempts and the separately rejected latent-shadow splice are recorded in docs/continuation_20261102_evening.md and docs/experiments.md.

Frozen weights [base, context, joint, combined608] = [0.140625, 0.421875, 0.1875, 0.25], then one Vinje projection. All four saved checkpoints replay exactly; independent float64 reconstruction, 725 IDs ×2353 CSV columns, [0,1] and readback passed. Submission: `artifacts/submission-lp-context-joint-blend-001/submission.csv`, SHA256 `f7907cce9d7348a1c77959e64d5e6099303123846c0648b37c816bb495411fcf`. Proof: `artifacts/verify-lp-context-joint-export-001/manifest.json`. Full recipe and commands: `docs/combined_export_recovery.md`; observation separately saved as `kaggle_observation.json`. The combined full fit completed in992.53s and replay in19.40s. Comparison gain is small; descriptive2022 bootstrap interval crosses zero.

The earlier joint result below is superseded as public/local best and preserved for history.

## Recent candidate outcomes

The 395-feature GPU XGBoost blend passed its frozen comparison against the earlier joint anchor: **2021 0.994935592985 / 2022 0.993725913572**, with weight 0.25. Its final fit completed in 208.49 seconds, with all 14 hosted checkpoint replays exact. The independently verified submission scored **0.99626**, below the then-current public best of 0.99631. This candidate is not promoted. Its small development gain did not establish a leaderboard improvement. Recipe: `docs/cuda_xgb_final.md`. CSV SHA256: `895b957f2c2fb19d1228b4cc6b960a24477ff977a06d2d790e68dd5b3d50cf92`. The separate `artifacts/submission-cuda-xgb-joint-blend-001/kaggle_observation.json` records the actual result; verified manifests remain immutable.

The 608-feature GPU candidate selected a 0.5 weight against the projected current 608-feature anchor. Its locked scores are **2021 0.9950359172801482 / 2022 0.9937529477072502**, gains of 0.000098198611 and 0.000061459754. These remain small improvements on reused development years. Lock SHA256: `5ab61a227270502d0f3a1387ea856a4cd0c10e2b50fec6b3864bad12629ee11f`. The separate final protocol `a5939edb8ec7f46dbdb4c6edd0c06909ea09ed0ac029709b06945a5c6434683b` and private dataset archive `56ae4c302ad93fd6b312112e831e78d4b37e157c1d89ddcf0c51b863044aa4b8` passed independent review. The earlier Computer Use transport interruption was resolved; final003 and both declared submissions are complete, with results above. Reproduction: `docs/cuda_xgb608_final.md`.

The water-value correction branch passed a bounded pre-2021 reachability diagnostic and then a locked 2019–2020 predictive screen. The latter's threshold-error gains are concentrated in Lio; inferred correction-distance diagnostics worsen, so this does not identify true water values. It has not yet been integrated with the strong 14-unit model. The continuous-state LP diagnostic improves a raw physical proxy on 36 pre-2021 cases, with several per-unit regressions retained. Neither diagnostic is 2021/2022 acceptance or leaderboard evidence. The nonlinear-PQ specialist remains rejected on 2022. No new result satisfies the 0.998 / 0.998 transfer gate.

## Earlier joint candidate

The fixed joint-model comparison selected **18.75% LP base +56.25% LP temporal-context +25% joint-plant LP**, followed by one Vinje projection: **2021 0.994888491751 / 2022 0.993641964015**. The global joint weight was selected from [0, .25, .5, .75, 1] on 2021 and locked before the 2022 prediction artifact existed. The unchanged blend improves the prior recipe by 0.000107379846 / 0.000171718097. Every lead-day group improves in both years; Haukeli unit AUC declines, so this is a pooled-objective improvement with a documented counterexample. No post-hoc generator weights are used.

Selection and exact array/provenance verification: `artifacts/lp-joint-comparison-001/analysis.json`, `selection_2021.json`, and `artifacts/verify-lp-joint-comparison-001/manifest.json`. Methods and diagnostics: `docs/lp_joint_comparison.md`. Both years are reused development folds. A descriptive paired 28-day circular case bootstrap, 200 replicates with seed4173, gives gain intervals [0.000067736, 0.000157452] / [0.000089625, 0.000280690]; these exclude adaptive-selection, training and distribution-shift uncertainty.

The new component replaces 78 LP-v1 inputs with 87 joint-plant inputs, retaining the same model and sample settings (404 features total). The joint cache SHA256 is `5fd1de4d08bf5db95892dc66a870431f38c3571e1bcc65ec62cbbe8882eb5db6`. All 3,647 cases solved, with no soft slacks. Independent physical checks include objective reconstruction, water conservation, unit allocations, source/input hashes and invariance to poisoned post-start historical storage. See `docs/lp_joint_plant.md` for the finite-grid, fixed-head and relaxed-commitment limitations.

Full fit `lgb-normalized-lp-joint-final-001` completed in629.43s; independent replay took13.30s and reproduced the complete raw test array and raw CSV SHA256 `2a0de19bb3dad535eda858611f6b62b63dd54011577a2ddd7e1480f877623fa6` exactly. All three final components and the frozen blend passed `verify-lp-joint-export-002`. The first verifier failed because a historical context export manifest had subsequently gained Kaggle metadata. Restoring only its four-null Kaggle object in memory reproduced the original exact hash; all model/data/source/weight checks remained strict, and the prior composition was independently reconstructed. The failed run is preserved. This joint candidate now has public score0.99625, rank24. Reproduction commands (use fresh run IDs when rerunning):

```sh
.venv-ml/bin/python scripts/train.py --run-id lgb-normalized-lp-joint-final-001 --years --feature-set normalized-lp-joint --trees 350 --sample-fraction 0.4 --threads 3 --final
.venv-ml/bin/python scripts/train.py --run-id replay-lp-joint-final-001 --years --feature-set normalized-lp-joint --inference-from artifacts/lgb-normalized-lp-joint-final-001 --threads 3
.venv-ml/bin/python scripts/export_selected_blend.py --run-id submission-lp-joint-blend-001 --selection-run lp-joint-comparison-001 --final-runs lgb-normalized-lp-final-001 lgb-normalized-lp-context-final-001 lgb-normalized-lp-joint-final-001
```

## Previous accepted local candidate

The fixed eight-candidate comparison selected **75% LP temporal-context +25% LP base**, followed by Vinje projection: **2021 0.994781111905 / 2022 0.993470245918**. The 2022 gain over the prior specialist/count candidate is only 0.000012049091, so this is a fragile improvement on repeatedly reused development years. A different grid item had a higher 2022 score but was not selected; weights remain those chosen on 2021. Provenance: `artifacts/lp-candidate-comparison-001/manifest.json`, its `selection_2021.json`, and `docs/lp_candidate_comparison.md`.

The temporal-context component adds 126 neighboring-discharge summaries to the 395 LP features, retaining the same350-tree configuration and sample rule. `lgb-normalized-lp-context-final-001` completed its full-data fit and exact checkpoint replay. The LP base full fit already exists. Replay the context component and export the frozen recipe using fresh run IDs:

```sh
.venv-ml/bin/python scripts/train.py --run-id replay-lp-context-final-001 --years --feature-set normalized-lp-context --inference-from artifacts/lgb-normalized-lp-context-final-001 --threads 2
.venv-ml/bin/python scripts/export_selected_blend.py --run-id submission-lp-context-blend-001 --selection-run lp-candidate-comparison-001 --final-runs lgb-normalized-lp-final-001 lgb-normalized-lp-context-final-001
```

This recipe now has public Kaggle score 0.99604, detailed below. The standalone temporal-context model is rejected because its 2022 score declined; the selected blend improves both years. LP dual-switching features hurt 2021 despite helping 2022 and are also rejected as a replacement. A locality-only restriction worsened all three specialist units; its small25% blend trails the accepted candidate, so no full locality fit is planned.

## LP component validation and reproduction

`lgb-normalized-lp-001`, with the Vinje probability projection selected on 2021 in `postprocess-normalized-lp-001`:

| Validation | Raw model | Projected probabilities |
| --- | ---: | ---: |
| 2021 | 0.994370012291 | 0.994372219552 |
| 2022 | 0.993348542164 | 0.993351463186 |

The matched model without the 78 LP features scored 0.989849818 / 0.989469422. These are repeatedly reused development holdouts, not untouched tests. All validation cases retain every hour. The last training starts are December 25 of the preceding year, so the last training target timestamp precedes January 1; six daily starts are purged.

The input-only LP solves each case independently from initial storage, known seven-day prices/inflows/constraints, terminal water values at start plus seven days, and supplied topology. It uses nine storage pools and fixed-head generation curves, with documented approximations. No commitments, solver runtime, or future realized reservoir volumes enter its predictors. All 3,647 cases solved successfully; see `docs/lp_features.md` and the cache manifest for numerical diagnostics and limitations.

The 395 float32 inputs combine base features, physical proxies, price context, LP dispatch/storage/dual features, and normalization of monetary features by mean absolute weekly price plus one. Fourteen separate binary LightGBM models use 350 trees, 31 leaves, learning rate 0.06, minimum leaf size 100, L2=1, feature fraction 0.9, 40% deterministic training-hour sample, seed 4173 and no early stopping. Complete expanded parameters, data/source hashes, sampled row IDs, splits, environment and checkpoints are in `artifacts/lgb-normalized-lp-001/manifest.json`. Postprocessing uses no smoothing and projects Vinje probabilities onto G1 >= G2 >= G3; its frozen 2021 selection is saved before 2022 evaluation.

Reproduction from this workspace, using the uv environment:

```sh
uv venv --python 3.11 .venv-ml
uv pip install --python .venv-ml/bin/python -r pyproject.toml
.venv-ml/bin/python scripts/build_lp_features.py --run-id lp-full-001 --all
.venv-ml/bin/python scripts/repair_lp_cache_schema.py artifacts/lp-full-001
.venv-ml/bin/python scripts/train.py --run-id lp-reproduce --years 2021 2022 --feature-set normalized-lp --trees 350 --sample-fraction 0.4 --threads 4
.venv-ml/bin/python scripts/analyze_predictions.py lp-reproduce --run-id lp-postprocess-reproduce
```

The existing `lp-full-001` cache should be verified and reused, not overwritten. The frozen v1 builder needs the documented metadata serialization repair under pandas 3. Its numeric feature values do not change. Feature archive SHA256: `0ebba7d10ffb29294e36e5e8742c73165e202ac1d96e6d15fbef7ea9d7cca7b9`. Existing validation checkpoints: `artifacts/lgb-normalized-lp-001/fold{2021,2022}_{unit}.txt`.

Full-data fit `lgb-normalized-lp-final-001` completed. Independent replay `replay-lp-final-001` reproduced every raw test probability bitwise and raw CSV SHA256 `83b759ec4cd29f74aa527bbc016f3d0f40e3fe74743479b2f4299afa0da5178c`. Its data-to-model recipe is:

```sh
.venv-ml/bin/python scripts/train.py --run-id lgb-normalized-lp-final-001 --years --feature-set normalized-lp --trees 350 --sample-fraction 0.4 --threads 3 --final
.venv-ml/bin/python scripts/train.py --run-id replay-lp-final-001 --years --feature-set normalized-lp --inference-from artifacts/lgb-normalized-lp-final-001 --threads 3
.venv-ml/bin/python scripts/export_submission.py --model-run lgb-normalized-lp-final-001 --selection-run postprocess-normalized-lp-001 --validation-model lgb-normalized-lp-001 --run-id submission-lp-001
```

The selected LP blend has now been submitted; see the current Kaggle result below. Later LP-context, directional-storage, specialist and plant-count experiments remain separate candidates until measured.

## Previous joint Kaggle result

**Public AUC 0.99625, team rank24**, observed2026-11-02T13:12:22.956714Z. Artifact: `artifacts/submission-lp-joint-blend-001/submission.csv`; SHA256 `911ec76d34b7bc2c1134286537a6bf068662effb80f93b200ddb6ec1fc1b110c`.

This is the accepted18.75%base/56.25%context/25%joint recipe. Every component checkpoint replay and rawCSV hash match exactly. Independent float64 weighted reconstruction followed by the fixed Vinje projection matches the saved test array byte-for-byte; schema/IDs/[0,1] and CSV readback pass. Proof: `artifacts/verify-lp-joint-export-002/manifest.json`. Observation is stored separately in `artifacts/submission-lp-joint-blend-001/kaggle_observation.json`, keeping the verified export manifest immutable. Numeric submission ID and private score remain unknown; no final-selection checkbox changed. Leading public scores remain.99820/.99808/.99805; both project gates remain unmet.

## Previous LP-context submission

**Public AUC 0.99604, team rank 28**, observed 2026-11-02T12:02:42Z. Artifact: `artifacts/submission-lp-context-blend-001/submission.csv`; SHA256 `36aa5a2f9378139916ee79aa710e9003875e3a81ce419f74f8fe1f75b50f3c98`.

This is the accepted75%LP-context+25%LP-base recipe above. Both raw model predictions replayed exactly from checkpoints. The context raw CSV SHA256 is `64fc8f5dbd5c00142a2961c6b039a396f047e05b8e13e1c4ecdafbec1c202232`. Independent float64 blend and Vinje projection reconstruction exactly matched the exported test matrix; CSV schema/IDs/probability bounds and readback passed. Proof: `artifacts/verify-lp-context-blend-002/manifest.json`. The first verifier incorrectly multiplied in float32; that failed check is preserved and the export was unchanged.

Private score and numeric submission ID remain unknown. Exact description and history: `docs/kaggle_scores.json`. No final-selection checkbox was changed. The0.998/both-validation-years transfer gate and0.999/top-three public goal remain unmet.

## Previous specialist/count submission

**Public AUC 0.99584, team rank 33**, observed 2026-11-02T11:41:14Z. Artifact: `artifacts/vinje-count-combined-final-001/submission.csv`; SHA256 `e769f2c77336ad958f14c6c38b18248707c55b248453d9012cf76fbc07ea4039`.

This uses the full normalized-LP model as its base, replaces Haukeli/Lio/Hogga with 50% base +50% XGBoost specialists, replaces the three Vinje probabilities with 50% base +50% active-count cumulative probabilities, then applies Vinje projection. Component weights were selected on 2021. Their fixed composition was proposed after observing both component-year results; its scores **0.994577404717 / 0.993458196827** are reused development evidence. All final count/XGB checkpoints and the LP base passed exact replay. The original files, complete training configurations, feature/sample indices and checksums are in the component manifests and `docs/vinje_count.md`.

```sh
.venv-ml/bin/python scripts/vinje_count.py --combined-final --run-id combined-inference-reproduce
```

The three leading public scores remained 0.99820, 0.99808 and 0.99805. Private score and numeric submission ID are unknown. Exact description, artifact checksum, rank and observation time are recorded in `docs/kaggle_scores.json`.

## Previous LP-blend submission

**Public AUC 0.99580, team rank 34**, observed 2026-11-02T11:27:19Z. Artifact: `artifacts/submission-lp-blend-001/submission.csv`; SHA256 `ff6c52f78c000d5235f460ca4ea814403f0fe5096249fb4b8f2a85a4e31b9e58`.

The selected mixture is **90% normalized-LP +7.5% normalized-context +2.5% physics**, followed by Vinje projection. Four mixtures were declared in `scripts/lp_ensemble_compare.py`; weights were selected on 2021 before reading 2022. Validation scores are **0.994467335276 / 0.993446371500**. Full provenance: `artifacts/lp-ensemble-001/analysis.json` and `docs/lp_ensemble_comparison.md`. All three final components passed exact saved-checkpoint replay. Export verifies full source/data/checkpoint hashes, matching validation/final recipes, exact schema and probability readback.

```sh
.venv-ml/bin/python scripts/export_selected_blend.py --run-id lp-blend-export-reproduce --selection-run lp-ensemble-001 --final-runs lgb-normalized-lp-final-001 lgb-normalized-context-final-001 lgb-physics-final-001
```

The observed private score is unknown. The UI did not expose a numeric submission ID; exact description and checksum identify it in `docs/kaggle_scores.json`. No final-selection checkbox was changed.

## Earlier submitted ensemble

**Public AUC 0.99254, team rank 82**, observed 2026-11-02T10:43:25Z. This is `submission-ensemble-001`, improving the previous team best 0.99172. At the same observation, the leading scores were 0.99820, 0.99808 and 0.99805. The private leaderboard remains unknown.

The submitted blend is 75% normalized-context and 25% physics, followed by Vinje projection. Weights were selected on 2021 only and then applied unchanged to 2022: **0.989966081512 / 0.989704028006**. Selection and all candidate weights are in `artifacts/ensemble-001/analysis.json` and `docs/ensemble_comparison.md`.

Final components are `artifacts/lgb-normalized-context-final-001/final_{unit}.txt` and `artifacts/lgb-physics-final-001/final_{unit}.txt`. Both use 350 trees and the matching recorded validation recipes. Replay runs `replay-context-final-001` and `replay-physics-final-001` reproduced their raw test arrays bitwise and their exact raw CSV checksums. `scripts/export_ensemble.py` verifies data, source, artifact, checkpoint, feature-order and model-recipe consistency before applying the frozen mixture. Export readback verifies all 725 IDs, 2,353 ordered columns, finite probabilities in [0,1], and serialization error at most 5e-10.

```sh
.venv-ml/bin/python scripts/export_ensemble.py --run-id ensemble-export-reproduce
```

Submitted artifact: `artifacts/submission-ensemble-001/submission.csv`.

SHA256: `cd972841fa1e8da50a0fbfd07d30dece5b695ea893d867e68b83df97283600d6`.

Kaggle observation and exact description: `docs/kaggle_scores.json`. The inspected UI did not expose a numeric submission ID; it is recorded as unknown rather than invented. No final-selection checkboxes were changed.

## Preserved references and negative experiments

- First new physics-only submission: public 0.99116; SHA256 `2bf904ad8337c1f5884be8385ababb9aa761ec9fc098de0761dbd6c04ae40590`. Its raw replay was exact.
- Earlier team reference CSV was recovered from Kaggle: `artifacts/legacy-reference/submission_round5_lgb_generators.csv`, public 0.99172, SHA256 `5b736f51f14028eaee5382124314fd4c368b72f546116d0a6ac053fafa086a29`. Its original checkpoints and validation arrays are unavailable. No validation score is assigned to blends involving it.
- Larger unnormalized physics model: 0.98431056 / 0.98575100, worse than the 350-tree recipe. It is rejected despite continuing to finish after the sandbox denied cancellation.
- Case calibration, economic logistic/spline blends and per-unit tree-prefix tuning improved 2021 but harmed 2022; rejected. Prefix scores were 0.99034352 / 0.98926910.
- Complete experiment and failure provenance is retained in `docs/experiments.md` and ignored artifact directories. The continuation regression suite passed 108 tests in9.687s (2026-11-02). Credentials, environments, raw/generated artifacts and caches stay out of commits.
