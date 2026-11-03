# Modeling review — 2026-11-02

This review uses the supplied labels/topology, `docs/data_audit.md`, and the ongoing `lgb-context-001` validation log. It makes no leaderboard-score claim. The completed 2021 context fold achieved micro AUC 0.98065317; the 2022 fold was still running when this review was prepared. An AUC above 0.99 is a research target, not an established expectation.

## Measured target structure

Statistics below use all 2,922 supplied training cases, 14 generators, and 168 hours. They describe labels; no new model was trained. Within-case switch rate is `mean(y[:, unit, 1:] != y[:, unit, :-1])`. Constant-case rate is `mean(sum_h(y) in {0,168})`.

| Unit | Hourly switch rate | Constant-case fraction |
|---|---:|---:|
| Hogga G1 | 1.03% | 57.05% |
| Tokke G1 | 1.81% | 48.77% |
| Tokke G2 | 1.17% | 59.17% |
| Tokke G3 | 1.51% | 53.76% |
| Tokke G4 | 1.27% | 57.87% |
| Lio G1 | 0.70% | 69.82% |
| Byrte G1 | 0.59% | 73.79% |
| Vinje G1 | 1.81% | 44.56% |
| Vinje G2 | 1.99% | 42.68% |
| Vinje G3 | 2.16% | 46.27% |
| Haukeli G1 | 0.30% | 81.28% |
| Songa G1 | 1.21% | 59.31% |
| Kjela G1 | 1.40% | 54.59% |
| Vesle Kjela G1 | 0.45% | 74.95% |

Vinje labels obey G1 >= G2 >= G3 on **99.9112%** of case-hours overall, **99.9641%** in 2021 and **99.9364%** in 2022. At exactly one operating unit, G1 is selected 98.68% overall; at exactly two, G1 and G2 are each selected approximately 99.99% and 100.00%. This is a strong empirical ordering, with exceptions; it does not justify replacing probabilities by hard binary states.

Tokke units are not exchangeable. At exactly one operating unit, G1 is selected 99.06%; at exactly two, G1 is selected 99.93% and G3 94.42%; at exactly three, G1 and G3 are always selected in the measured data, while G4 is selected 66.36% and G2 33.64%. The preferred order G1,G3,G4,G2 holds for 97.9680% of case-hours in 2021 and 98.4817% in 2022. Enforcing a complete Tokke order would discard a real G2/G4 selection mechanism. Numerical generator order is especially unsuitable.

The topology explains why generator identities matter: Tokke G1/G2 share penstock 1 and G3/G4 penstock 2; their flow-loss coupling may influence the third-unit choice. Vinje units have separate penstocks. Use identities and plant state, not an equality constraint among same-plant units.

## Highest-priority ablations

1. **Within-case probability smoothing.** Test centered edge-normalized windows 3,7,13,25 blended with original probabilities at weights 0,.25,.5,1. The low switch frequencies support this hypothesis. Select the recipe using only 2021; freeze it before evaluating 2022. Never smooth across case boundaries or across different cases predicting the same timestamp. A sustained incorrect case-level prediction will not be repaired by smoothing, so this complements physical features.
2. **Optional Vinje probability projection.** L2 projection onto G1 >= G2 >= G3 preserves the summed unit probabilities and produces a coherent empirical ordering. Evaluate jointly with smoothing. The documented exceptions mean the projection must earn its place in validation.
3. **Topology-aware water balance and opportunity cost.** The weakest completed 2021 unit scores are Haukeli 0.912147, Lio 0.959224, and Hogga 0.969480; these are the best initial targets for feature/error analysis. The current features include direct local inflow and unadjusted storage headroom but do not represent upstream discharge or mandatory bypass flows. Include upstream catchment flow totals, reservoir capacity, initial storage, minimum-volume constraints and expected inflow in dimensionally consistent supply/headroom ratios. Distinguish directly connected storage from water that would first need upstream generation. Root's physical feature work should be evaluated against the same context model configuration before simultaneously expanding tree budgets.
4. **Increase tree budget and sampling separately.** Existing training samples only part of the case-hour rows. Compare a deeper/longer model on the same sampled rows before varying the sample fraction; then use full data for selected configurations if runtime allows. This distinguishes feature gains from capacity or sampling gains. Keep folds identical and retain per-unit scores, as micro gains can hide a major weak-unit regression.

## Next model families if the preceding gains plateau

- **Vinje plant-count model:** train a four-class classifier for the number of active units (0..3), then derive G1/G2/G3 probabilities from P(count >=1/2/3). Compare with the existing per-unit binary models and blend only if 2021 improves. This shares information across units and guarantees calibrated count coherence, but cannot represent the rare non-nested label states. A per-unit residual model could recover exceptions later.
- **Case-level state plus hourly allocation:** Haukeli has 81.28% constant cases, Lio 69.82%, and Vesle Kjela 74.95%. Predict the seven-day on-fraction or the all-off/mixed/all-on case state from initial conditions and week summaries. Combine these only through chronologically out-of-fold predictions for training the hourly model. Do not train an hourly second stage on in-sample case predictions and evaluate using out-of-sample ones.
- **Approximate startup-cost optimization features:** use a physically derived per-hour marginal profit and known startup/stop costs to compute a two-state dynamic-programming schedule or soft state scores. Supply this approximation to the learned model as features. Start with a single unit and evaluate an ablation; a full recreation of SHOP's hydraulic optimization is substantially more work and exact solver label reproduction is unproven.
- **Chronological stacking of connected-unit predictions:** predicted upstream discharge/activity can explain downstream state. Generate these predictions on purged rolling folds; never use observed validation commitments, including earlier validation cases. Simply appending true upstream-unit labels during training creates an inference mismatch and potentially severe leakage.
- **Cross-unit calibration:** pooled micro AUC compares scores between units. Fit any intercept/temperature adjustment exclusively on earlier out-of-fold predictions, then evaluate the later year. Per-unit AUC cannot establish whether such calibration helps pooled AUC.

## Topology connections worth representing

The provided YAML contains these cascade links: Vesle Kjela -> Foersvatn -> Kjela -> Hyljelihyl -> Venemo -> Totak -> Vaamarvatn -> Vinje -> Vinjevatn -> Tokke -> Bandak -> Hogga. Some links are rivers, tunnels or bypasses rather than generation, so graph reachability alone does not determine available discharge. Songa feeds Totak; Haukeli feeds Vinjevatn through Vatjern's catchment; Byrte feeds Byrtevatn -> Lio -> r_Vest_Vassdraget -> Bandak. Tributary inflow to r_Vest_Vassdraget therefore matters to Hogga even though it is not one of the 17 reservoir inflows.

The input documentation defines water values in EUR/MWh, not EUR/m3. Do not subtract upstream and downstream water values as though they were unconverted water-volume costs. Any head/energy-equivalent conversion must state its approximation and be tested. Current and terminal reservoir water values can be redundant across connected groups; a reduction may help efficiency, but is less urgent than physically meaningful derived features.

## Validation and provenance

Keep the six-case purge and score all 168 hours ×14 units. Known future price/inflow inside the case horizon are legitimate optimizer inputs. Future historical reservoir volumes, post-solve calculation time, validation labels from neighboring cases, and hidden labels remain excluded. Random splitting of case-hour rows is unsuitable.

Source label SHA-256: `6d51f8cc6869cb82b44708168808c305c1e007cf79878ca1c1d316d07aae3ab3`. Topology SHA-256: `29cfa233cec7fc4b5401d0d571a566e6e17b7bb7b46a175abcd37a53fe175497`. Statistics were computed with `.venv-ml/bin/python`, NumPy 2.4.6 and pandas 3.0.6. Label array construction is `pd.read_csv(label_path).iloc[:,3:].to_numpy().reshape(2922,14,168)` in the supplied generator order. The postprocessing experiment logs its exact code snapshots, full grid, saved-prediction checksums, folds and results separately.

## Completed postprocessing result

`postprocess-context-001` evaluated the planned 32-configuration grid. Selecting solely on 2021 chose **no smoothing, Vinje ordering projection enabled**. The 2021 micro AUC changed from 0.98065317 to 0.98069869; the frozen configuration changed 2022 from 0.98435803 to 0.98438025. Every tested nonzero smoothing configuration scored below the 2021 baseline, despite low target switching rates. Do not infer that persistence alone warrants smoothing. It can erase correctly timed operational boundaries; dedicated sequence features or models remain separate hypotheses. Projection provided a small repeatable local improvement, with no measured leaderboard result.

Full report: `artifacts/postprocess-context-001/analysis.json`; selection was recorded in `selection_2021.json` before reading 2022 labels.
