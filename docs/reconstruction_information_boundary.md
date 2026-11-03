# Reconstruction information boundary

Status: documentation and plan only. Reconstruction execution is deferred. No new restricted replay, guard, native interposer, model fit, model selection or numerical experiment was run for this inventory. Existing experiment records remain unchanged.

The defensible description is **audited supplied numerical-input ancestry with prior external research and evaluation exposure**. It is not a clean-room reconstruction, a claim that development was blind to test-related information, or proof that every modeling choice originated in the supplied files.

The machine-readable inventory is [`reconstruction_source_inventory.json`](reconstruction_source_inventory.json), SHA256 `2f4d8704d8b9b20e82e4e31fa74be6dbdcca77d8d531314c3e5a1bad492b0e19`. It copies the existing lineage audit's 528 file hashes, 253 source snapshots, cache/model/bundle/composition records and ten supplied numerical/schema leaves. It also identifies external exposure, empirical choices and the deferred execution checks. This inventory was assembled from explicitly named existing metadata and documents; it did not rescan giant artifact directories or open external archives. Its file membership is not an automatically approved runtime allowlist.

## Numerical inputs and inherited artifacts

| Category | What is included | What the evidence establishes |
|---|---|---|
| Supplied inputs | Kernel day-ahead prices, inflows, historical volumes, synthetic water values and training commitments; extended minimum-flow/minimum-volume constraints and topology; prediction mapping and sample submission | Ten exact path/hash pairs are in the inventory. These are the declared numerical leaves of the audited original ensemble. |
| Local physical calculations | `lp-full-001`, `lp-joint-full-001`, `lp-shop-full-001` | Existing source/data/checksum closure attributes these cached calculations to supplied numerical inputs. Their physical equations and approximations were informed by general external documentation. |
| Original learned ensemble, A0 | Four earlier LightGBM variants, nested blends, GPU XGBoost608/664 and the final LightGBM664 addition | The original source/model/checkpoint/bundle chain is preserved by `provided-only-lineage-audit-002`. A0 is an ensemble, not a single model. “SHOP664” denotes locally engineered physics features, not a downloaded solver-output table. |
| Canonical115 anchor | Pooled unit-relative features and its accepted0.25 blend with A0 | Exact inherited composition and hashes were audited. The anchor is a material input to later recipes; calling only the final classifier “supplied-only” would omit this dependency. |
| Proposed startup142 replay | Canonical115 prefix plus27 supplied-topology economics fields, saved1400-tree model and fixed0.25 blend with the accepted115 anchor | Existing unrestricted inference rebuilt the27 additions and replayed all1,705,200 final scalar predictions. It did not regenerate115 features, inherited models or original LP caches. |

The lineage audit is [`../artifacts/provided-only-lineage-audit-002/manifest.json`](../reports/reconstruction_20261103/manifests/provided-only-lineage-audit-002.json), SHA256 `6c017919db2e6879e718398bb0aec7fc59f79b73b68a439a73a9750885b61f92`. It records no external numerical input files in that ancestry. This is a saved-file provenance and composition audit, not a fresh rerun of every model/solver, a formal semantic taint proof, or proof that hard-coded formulas, physical constants and hyperparameters were uninfluenced by outside knowledge.

## Research, empirical choices and previous exposure

External dataset discovery emerged during the autonomous research and Kaggle submission sequence. The public SINTEF SHOP17.15.1 and17.17.0 archives supplied simulator outputs for the same3647 cases, including the725 competition test cases. The0.99795 submission directly used17.17 envC NoMIP relaxed commitment fractions and an earlier17.15.1 component. The0.99952 submission added17.17 envC AllMIP commitment states. Those entries therefore use external numerical information at inference. They are separate from the supplied-input ancestry above. Source identities, hashes, separate score receipts and the complete nested formulas remain in [`best_result.md`](best_result.md) and the inventory; no new public score is inferred here.

Later historical mechanism analysis was restricted to original cases1–2186 before2021. That restriction is meaningful for that individual analysis, but it cannot undo earlier exposure to external simulator outputs for validation and test cases. No access to official hidden test labels is documented; nevertheless, closely related test-case simulator predictions and public leaderboard feedback had already been observed. Other external hourly hydrology, public solver modes and source-generation metadata were investigated in separate branches. They are not numerical dependencies of the chosen startup142 replay. This inventory does not claim exhaustive reconstruction of every browser view or discarded hypothesis.

External publications, SHOP/SINTEF documentation and model consultations influenced optimizer approximations, feature design and experimental priorities. The existing [`web_research.md`](../reports/reconstruction_20261103/research/web_research.md), [`consultation.md`](../reports/reconstruction_20261103/research/consultation.md) and [`historical_solver_mechanism_research.md`](../reports/reconstruction_20261103/research/historical_solver_mechanism_research.md) distinguish recorded recommendations, tested decisions and unsuccessful hypotheses. Exact model identity was not always exposed in consultation UI; preserve those limitations. A numerical-file boundary does not erase this methodological influence.

Several decisions also depend on the supplied labels and repeated evaluation:

- Vinje's `G1 >= G2 >= G3` probability projection is empirical, with observed exceptions. Initial descriptive statistics examined all2922 supplied training cases, including2021 and2022. A subsequent grid selected projection with no smoothing on2021 and confirmed it on2022. It is not a supplied physical law or globally untouched confirmation.
- Unit asymmetries, weak-unit priorities, fixed diagnostic cases, capacity choices and ensemble families were informed by training-label statistics and previous experiments. Full Tokke ordering and exchangeability were not assumed. Some physical hypotheses were motivated by external historical comparisons.
- The chronological fits use specified target cutoffs and a six-start-date purge so168-hour horizons do not cross validation boundaries. However,2021 and2022 have been reused extensively for development. Freezing a particular run's choice before its2022 scoring does not make that year globally fresh.
- Public scores/ranks and supplied test covariates were visible during development. Report measured public scores separately from reused development AUC. Private leaderboard and final competition outcomes remain unknown.
- “Provided inputs” does not imply causal forecasting. The historical0.99653 candidate explicitly uses supplied retrospective future storage/water-value trajectories. Those971 additions are not in the selected startup142/A0 ancestry; they must not be silently substituted into it.

These are information-boundary disclosures, not a claim that an unmeasured shortcut was used. Existing evidence supports the specified numerical lineage; it does not certify independence from all prior knowledge.

## Existing replay versus deferred restricted replay

The proposed first target is the complete existing startup142 final recipe:

- Final manifest: [`../artifacts/provided-startup142-final-001/manifest.json`](../reports/reconstruction_20261103/manifests/provided-startup142-final-001.json).
- Fixed recipe: [`provided_startup142_recipe.json`](../reports/reconstruction_20261103/research/provided_startup142_recipe.json), SHA256 `63b79ceaf39181b81aff1e775408e1fb0a511554a2e8ad90945228be2c962893`.
- Saved native checkpoint: `artifacts/provided-startup142-final-001/startup142.txt`.
- Expected CSV SHA256: `498c2f7b9acd3a3974b855b816f94da4ba52ea6ce1de86153cd294cafb5b1953`.
- Existing full inference receipt: [`../artifacts/physics-provided-startup142-final-inference-001/manifest.json`](../reports/reconstruction_20261103/manifests/physics-provided-startup142-final-inference-001.json), SHA256 `3a24af261959f550cde839c991f9aa36874ba0235abeb249b41c515967d2e5e1`.

That existing receipt establishes exact27-field reconstruction, full native prediction replay and identical selected array/CSV from the pinned caches and checkpoint. It was not executed under the proposed new access boundary. The verifier also hashes declared inputs, including the supplied training-label file; “no target decoding” in its inference code must not be represented as an OS-level inability to read that file.

One availability probe attempted `/usr/bin/sandbox-exec -p '(version 1) (allow default) (deny network*)' /usr/bin/true`. It exited71 with `sandbox_apply: Operation not permitted`. Docker, Podman, bwrap and nsjail were not installed. This was not a successful isolation test. No escalation or bypass was attempted. Execution was then stopped at the documentation-only boundary: no new guard/interposer was implemented, no canary was tested and no restricted numerical replay ran.

## Reconstruction plan, not executed

1. **Freeze the target and scope.** Preserve the supplied-input target, exact source snapshots, data/schema hashes, original splits/samples, all nested anchor projections and selected weights. Keep external-output submissions and research records clearly separate. Reproduction does not authorize new model selection or use of external outputs.
2. **Create a finite staging closure.** Copy only the approved supplied files, exact code snapshots, audited clean checkpoints/caches and necessary manifests into a new staging root. Enumerate paths and hashes explicitly. Do not allow the entire workspace, artifact tree or user home; adjacent directories contain external solver outputs. Pin the Python/native software separately and disclose trust in that software.
3. **Verify the access boundary before inference.** Prefer kernel/container isolation with file and network denial if available. If only Python audit hooks plus a native interposer are feasible, describe it as process-level interception: direct syscalls, malicious/native-extension bypasses and trusted-library contents remain limitations. Test denied reads through both Python and native code, including an external-path canary, and denied network connections. Record failures honestly; do not infer OS isolation from Python-only tests.
4. **Replay the saved142 checkpoint and compose the final CSV.** Rebuild the27 additions from supplied topology and pinned115 rows; replay every test prediction; preserve the exact0.25 mixture and Vinje projection; check all original IDs,14-unit×168-hour mapping, finite `[0,1]` probabilities, selected bytes and CSV checksum. This phase reproduces cached-artifact inference only.
5. **Rebuild the feature and anchor ancestry separately.** Under the same declared boundary, regenerate original provided-input physical caches and their formulas/feasibility receipts, construct the664 and canonical115 matrices with the recorded columns and row order, and replay every inherited checkpoint/blend stage. Match frozen artifacts where bitwise equality is expected and state any platform-dependent tolerance prospectively. The lineage inventory identifies the sources and intermediate manifests; the existing protocols retain detailed commands and settings.
6. **Keep training reproduction a separate phase.** A full retraining run must use the exact recorded target cutoffs, six-case purge, original40% samples, seed4173, architecture/settings and frozen selection. It is more than a no-fit inference replay and may have hardware/library nondeterminism. Do not claim raw-data-to-model reproduction merely because a saved checkpoint loads or a cached prediction matches.

The immediate deliverable is this inventory and plan. Restricted inference, canary checks, feature-cache regeneration and retraining remain unexecuted. They require a later explicit restart; no current score or lineage record is replaced by the plan.
