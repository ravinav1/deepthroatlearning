# Supplied-data-only attempt — running log (started 2026-11-03)

Information boundary: inputs limited to the supplied competition files and audited supplied-only caches/checkpoints in the original workspace. External solver directories are refused by `scripts/supplied_only/common.py`. Every script writes an input ledger with SHA256 hashes under `reports/supplied_only/`.

## D1: overlap label agreement (negative)

Overlapping cases disagree far more than assumed. Agreement on the same calendar hour between cases one day apart is 0.8777 (positive-class Jaccard 0.788); at six days 0.742. The 97% gate for the cross-case consensus levers (L1, L2) is not met, so both are dropped without fitting. Evidence: `reports/supplied_only/d1_overlap_agreement.json`.

## Component verification

`scripts/supplied_only/components.py` reproduced every recorded fold score (anchor 0.9951042082 / 0.9938717711; canonical blend 0.9952596524 / 0.9940509265). Component arrays saved in `models/supplied_only/components_{2021,2022}.npz`; inputs in `reports/supplied_only/components_inputs.json`.

## L6 Tokke partial-order projection (accepted, negligible)

Projecting Tokke onto G1 ≥ G3 ≥ G4 and G3 ≥ G2 changes AUC by +0.00000057 (2021) and +0.00000005 (2022). Selected mix 1.0 on 2021; 2022 positive. Kept because it is monotone-safe, but it contributes nothing measurable. Evidence: `l6_tokke_projection.json`.

## L3 per-plant weights (rejected)

Coordinate-wise per-plant weights over canonical115 / startup142 / busbar144 raw predictions: 2021 +0.000229, 2022 −0.000113. Rejected; same transfer failure as Round9 in the old lineage. Evidence: `l3_plant_weights_selection_2021.json`, `l3_plant_weights_evaluation_2022.json`.

## L3b global three-component weights (accepted)

Full grid over one global weight per component. Selected on 2021: canonical115 0, startup142 0.25, busbar144 0.10, applied to the anchor A0 with Vinje projection. 2021 0.9953761449 (+0.000116 over the canonical blend), 2022 0.9941452767 (+0.000094). This is the current supplied-only candidate recipe. Evidence: `l3b_global_weights_selection_2021.json`, `l3b_global_weights_evaluation_2022.json`.

## L4 canonical115 seed bag (2021 only; stopped)

Two extra seeds (row bagging 0.8) on the saved canonical115 matrices: projected standalone AUC 0.994280 (seed 4173), 0.994330, 0.994336; three-seed mean 0.994458. On the anchor at 0.25 the bag beats the single seed by +0.000021, but on top of the accepted L3b recipe adding the bag only hurts (0.995358 at 0.1 vs 0.995376). The 2022 and final feature matrices lived only in the original workspace, which disappeared during this session, so the lever cannot be completed. Evidence: `l4_seed_bag_2021.json`.

## L5 legacy-lineage member (accepted)

Retrained the Round10-style separate-generator LightGBM (113 features + two price ranks, economic rescaling, fixed per-generator tree counts) from the supplied raw files in this clone (`configs/round10_legacy_so_seed{299,673}_{2021,2022,final}.json`). Seed 299 standalone: 0.988673 (2021), 0.987770 (2022). Blended at a 2021-selected weight of 0.05:
- on top of L3b: 2021 0.9954430 (+0.000067), 2022 0.9941885 (+0.000043), accepted;
- directly on the anchor (exportable fallback): 2021 0.9952124 (+0.000108), 2022 0.9939351 (+0.000063), accepted.
Evidence: `l5_legacy_selection_2021*.json`, `l5_legacy_evaluation_2022*.json`.

## Blocker: original workspace removed

`/Users/heap/programming/tdt4173` (raw data, LP caches, startup142/busbar144 code and test arrays, pinned environment) vanished mid-session. The clone's GitHub release restored the 0.99652 anchor's 98 checkpoints and test arrays (export reproduces the exact CSV), and the raw data had already been copied. The accepted L3b recipe cannot be exported without the startup142 and busbar144 final test predictions or their feature code.

## Two-seed legacy member and fallback export

With seeds 299 and 673 averaged (standalone 0.988897 / 0.987973), weight 0.05 remains the 2021 choice:
- on L3b: 2021 0.9954456 (+0.000069), 2022 0.9942126 (+0.000067) → best development candidate, **not exportable** without the workspace;
- on the anchor: 2021 0.9952137 (+0.000110), 2022 0.9939539 (+0.000082) → exported.

Fallback export `models/supplied_only/exports/so-fallback-anchor-legacy-001/submission.csv`, recipe `configs/supplied_only/fallback_anchor_legacy.json`: `project_tokke(project_vinje(0.95·A0 + 0.05·mean(legacy seed299, seed673)))`. 725 × 2,353, template order, nine significant digits, readback ≤ 5e-10, independent reconstruction exact. Inputs: the restored release anchor array and the two clone-trained legacy prediction files; raw data only from the supplied files.

Development summary against the 0.99652 anchor (dev 0.9951042 / 0.9938718): fallback +0.00011 / +0.00008; blocked L3b+legacy +0.00034 / +0.00034. Public gain expected to be of the same order, so neither reaches 0.99820.

## Kaggle result: fallback submission

`so-fallback-anchor-legacy-001` (zip of the verified CSV, SHA256 8b289e47…) submitted 2026-11-03 ~13:30 UTC via the Kaggle UI. **Public 0.99649**, below the 0.99652 anchor. The +0.00011 / +0.00008 development gains did not transfer; the public set (2023) is a different regime and this margin is within its noise. Not a new best; the 0.99652 anchor stays the reference supplied-only entry.

## Rebuilt feature table and startup-style component (in progress)

`scripts/supplied_only/build_664.py` rebuilt the frozen 664-column table from the research modules and the release-restored LP caches (`lp-full-001`, `lp-joint-full-001`, `lp-shop-full-001` features.npz); column order verified against `docs/lgb_shop664_protocol.json`. The original 2021/2022 sample indices were lost, so those folds use a fresh 40% hourly sample (rng seed 4173); the final fit reuses the exact original `train_sample_final.npy`.

`scripts/supplied_only/startup_like.py` (tag `startup16`): canonical115 rows plus 16 fields from the supplied topology (p_min, p_max, start/stop cost) combined with existing normalized margins and price windows. This is a re-implementation inspired by the lost startup142 recipe, not a replay of it. Pooled LightGBM, 1,400 trees, 31 leaves, lr 0.06, seed 4173. Grid [0, .1, .25, .5] on the anchor, 2021 select, 2022 locked.

## startup16 result (accepted)

| Fold | Anchor | startup16 standalone (projected) | Anchor + 0.25 startup16 | Gain |
|---|---:|---:|---:|---:|
| 2021 | 0.9951042082 | 0.9944211532 | 0.9952950864 | +0.000191 |
| 2022 | 0.9938717711 | 0.9931725678 | 0.9941764124 | +0.000305 |

Weight 0.25 selected on 2021 from [0, .1, .25, .5]; the 2022 run scored the whole grid in one pass but the reported 2022 figure is the 2021-locked 0.25 entry, and no weight was changed after seeing 2022. Fits took 467 s / 539 s with four threads. Final fit (all 2,922 cases, original final sample) and one extra seed per fold launched afterwards. Evidence: `models/supplied_only/startup16_{2021,2022}/manifest.json`.

## Kaggle results: startup16 candidates

| Export | Recipe | Dev 2021 / 2022 | Public |
|---|---|---|---|
| so-primary-anchor-startup16-001 (SHA256 a45f8001…) | P_tokke(P_vinje(0.75·A0 + 0.25·startup16)) | 0.9952951 / 0.9941764 | **0.99656** (new supplied-only best, +0.00004 over the anchor) |
| so-secondary-anchor-startup16-legacy-001 (SHA256 38cf96b1…) | primary, then + 0.05 legacy mean | 0.9953725 / 0.9942156 | 0.99654 |

Both uploaded via the Kaggle UI at about 14:00 UTC. The legacy member again fails to transfer; it is dropped from further recipes.

## startup16 + cross-case LP states (rejected)

Appending the neighbouring cases' SHOP relaxed state for the same calendar hour (offsets ±1..±3 days, mean, count; 8 columns) to startup16: anchor blend at 0.25 scored 0.9953166 on 2021 (+0.00002 vs startup16) but 0.9941041 on 2022 (−0.00007). Rejected. Evidence: `models/supplied_only/startup16cc_{2021,2022}/manifest.json`.

## startup16 seed bag

Seed 1 (row bagging 0.8): standalone 0.9944047 / 0.9930665; anchor + 0.25·mean(seed 4173, seed 1): 2021 0.9953148, 2022 0.9941981 (vs single seed 0.9952951 / 0.9941764). Exported as `so-bag2-anchor-startup16-001` (SHA256 f7b0b52f…). Seeds 2 and 3 launched for all folds.

## Kaggle result: bag2

`so-bag2-anchor-startup16-001` (SHA256 f7b0b52f…): **public 0.99657**, best supplied-only entry so far (anchor 0.99652 → startup16 0.99656 → bag2 0.99657).

## Seed bag saturation

Three seeds (4173, 1, 2): anchor + 0.25·mean → 2021 0.9953157, 2022 0.9941790, i.e. 2022 below the two-seed bag (0.9941981). Bagging stops at two seeds. Evidence: `compose_eval_seeds4173_1_2.json`.

## Volatility shift (diagnostic)

Median within-week relative price std (`norm_price_week_std`, first hour of each case): ≤2020 0.092, 2021 0.182, 2022 0.219, test 2023–24 0.321 (q90 0.625). Motivates the `--volatility-weight` option: training-hour weight 1 + 3·std, normalized to mean one.

## startup16 with volatility weighting (rejected)

Weight 1 + 3·norm_price_week_std per training hour: anchor blend 2021 0.9953084 (+0.00001 vs startup16), 2022 0.9941251 (−0.00005). On the high-volatility half of cases: 2021 +0.00004, 2022 −0.00003. Rejected. High-volatility weeks are already the easier half (anchor 0.9961 vs 0.9940 on 2021). Evidence: `subset_eval_startup16_startup16vw.json`.

## Optimizer-only baseline, no machine learning (2026-11-04)

Question answered: how well does the pure optimization surrogate rank unit-hours on its own, with no fitted model? Script `scripts/supplied_only/optimizer_only.py`; export `models/supplied_only/exports/optimizer-only-001/` (CSV SHA256 `e305b51397ea84bf4717ffb225bea8ceb1a3b92119819ddcd8987a48624b2472`). Inputs: the label-free caches `lp-shop-full-001` (fractional commitment u with supplied start/stop costs) and `lp-full-001` (q/Qmin), both verified against their manifests; nothing trained, so no purge. Twelve predeclared monotone candidates; selection by 2021 micro AUC, 2022 reported unchanged. The `qproxy/none` row reproduces `lp-diagnostic-002` (0.946964869 / 0.952251082) to 1e-6, confirming label alignment.

| Score | none | Vinje | Vinje+Tokke |
|---|---:|---:|---:|
| state u | 0.955747 / 0.956800 | 0.956440 / 0.957296 | 0.957324 / 0.957759 |
| qproxy | 0.946964 / 0.952251 | 0.947375 / 0.952732 | 0.947793 / 0.953164 |
| u + 0.01 qproxy | 0.956777 / 0.957835 | 0.957410 / 0.958307 | 0.958173 / 0.958729 |
| 0.5 u + 0.5 qproxy | 0.961358 / 0.960287 | 0.961728 / 0.960596 | **0.962333 / 0.960947** (selected) |

Verdict: the optimizer alone reaches about 0.96 dev AUC versus 0.9953 / 0.9942 for the ML pipeline that consumes the same LP outputs as features. The gap is the unreproduced SHOP behaviour already catalogued in `docs/reconstruction_mechanism_evidence.md`. Public score observed 2026-11-04 (uploaded as a zipped CSV): **0.96516**, versus 0.99657 for the best ML entry; recorded in `reports/kaggle_scores.json`.
