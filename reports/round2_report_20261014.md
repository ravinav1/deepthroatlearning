# Second autonomous model investigation

**Completed 14 October 2026.** The selected models are fitted and one submission is exported and independently verified: [submission_round2_depth8_haukeli_weighted.csv](../submissions/submission_round2_depth8_haukeli_weighted.csv). The user subsequently reported a **public Kaggle score of 0.99053**, the new reported best. No upload was performed by the agent.

The investigation began at 13:06:15 UTC with a 2.5-hour allocation and an original wall-clock deadline of 15:36:15 UTC. The goal was paused before that deadline and resumed afterward; verification and delivery were completed following the explicit resume. Heavy training ran sequentially with four CPU threads. The [experiment ledger](autonomous_round2_20261014.json) records the schedule, 30 experiments and diagnostics, three final component fits, and deferred work.

The selected candidate improves historical micro ROC-AUC by **0.000647 in 2021 and 0.000605 in 2022** over the starting ensemble. The evidence supports better component selection and complementary models, while leaving transition timing and Lio unresolved. It does not establish a route to 0.995. The user-reported public score improved from **0.989** for `submission_rescaled_xgb_lgb_transformer.csv` to **0.99053** for the latest submission. The earlier score was supplied to only three decimal places; the new public result supports the combined recipe but does not isolate individual component effects.

## Historical result and selected recipe

All values below are pooled across the 14 generators and 168 hourly outputs per case. Each full year contains 365 cases. The Jan 7 onward check excludes six 2022 starts overlapping the last 2021 selection horizons.

| Ensemble | 2021 | 2022 | 2022 Jan 7 onward |
|---|---:|---:|---:|
| Starting winner | 0.988271938 | 0.987770675 | 0.987690089 |
| Confirmed fallback | 0.988523351 | 0.988148784 | 0.988068726 |
| Selected combination | **0.988918450** | **0.988375535** | **0.988277942** |

The [selected recipe](../configs/round2_submission_depth8_haukeli_weighted.json) uses depth-8 XGBoost outside Haukeli. Haukeli uses 25% original shared XGBoost, 25% Haukeli specialist and 50% original LightGBM. Every existing Transformer contribution becomes a 50/50 mean of the original and task-weighted Transformer. Total plant weights and LightGBM remain unchanged. Counts selected in 2021 are **748 trees, 121 specialist trees and 28 Transformer epochs**.

The fallback instead keeps the original shared tree model and averages the original Transformer with the new ensemble-monitored Transformer, retaining the Haukeli specialist mean. The selected recipe exceeds it by 0.000395/0.000227. Selection was recorded from two predefined 2021 combinations before reading the corresponding depth-8 and weighted-Transformer 2022 results. [Selection record](round2_depth8_combined_selection_2021.json), [complete comparison](round2_depth8_combined_both_comparison.json).

**The depth increase has mixed effects.** Holding the rest of the selected recipe fixed, depth 8 adds 0.000262 in 2021 but subtracts 0.000124 in 2022. Other changes compensate. The combined result therefore does not demonstrate that deeper trees generalize better. This later ablation describes attribution; it does not silently replace the 2021-selected recipe using 2022 results. [Depth attribution](round2_depth8_attribution.json).

## What was tested

These component experiments preserve the original full ensemble except for the named change. “Mean” means averaging original and new predictions within that component's existing share.

| Change | 2021 ensemble AUC | 2022 ensemble AUC |
|---|---:|---:|
| Ensemble-monitored Transformer mean | 0.988513384 | 0.987964578 |
| Task-weighted Transformer mean | 0.988648944 | 0.988311182 |
| Weekly Mixer mean | 0.988359760 | 0.988135175 |
| Haukeli specialist mean | 0.988276888 | 0.987958553 |
| Depth-8 XGBoost replacement | 0.988561865 | 0.987760194 |
| Lio specialist replacement | 0.988955732 | 0.987743382 |

The [monitor control](round2_transformer_monitor_budget_both_comparison.json) selects checkpoints using the frozen final ensemble, rather than standalone neural AUC. An initial short-patience run stopped too early; the controlled comparison used the original 39-epoch training budget, selecting epoch 37. The fitting budget and architecture were held fixed; the chosen checkpoint changes the number of training epochs represented by the exported model.

The [task-weighted control](round2_transformer_task_weights_both_comparison.json) weights generator BCE by frozen plant neural shares, normalized to mean one. It directs learning toward outputs used by the ensemble. Four plants with zero neural contribution receive zero direct loss in this control, so these predictions must remain ensemble components; tree models still cover those plants. Averaging helps more reliably than replacing the old Transformer; this suggests useful diversity, without proving negative transfer caused the original errors.

The [Mixer](round2_mixer_both_comparison.json) processes the same full week with alternating temporal and channel MLPs. Replacement is worse in both years, but its mean helps. It also has more parameters than the Transformer, so architecture and capacity are not isolated. The [TSMixer paper](https://arxiv.org/abs/2303.06053) motivates this alternative; its forecasting results are not a benchmark for this task. [GradNorm](https://proceedings.mlr.press/v80/chen18a.html) motivates considering task balance, but the implemented fixed weights are not GradNorm.

Tree controls covered convergence, depth, specialist fitting and stronger regularization, following [XGBoost's tuning documentation](https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html). Unbounded LightGBM continuation helped 2022 but regressed 2021. One large late leaf update affected 1,021 Lio outputs; source trees and feature mappings were intact. A separate capped-update follow-up also failed 2021 ensemble selection, so it was not fitted for 2022. Global depth 4 and specialist depth/penalty controls produced no stronger consistent replacement. [Tree research](round2_tree_research.json), [update diagnosis](round2_lgb_2021_drop_diagnosis.json), [capped follow-up](round2_lgb_capped_2021_comparison.json).

Replacing the XGBoost Lio contribution with the [Lio specialist](round2_xgb_lio_both_comparison.json) slightly improved within-Lio ranking of the full ensemble in 2022 while worsening pooled cross-generator ranking; the specialist itself had worse standalone Lio AUC. A 2021-only Platt fit improved BCE but reduced ensemble AUC; identity calibration was retained and the calibrated variant was not evaluated on 2022. Naive temporal smoothing also selected zero smoothing. These are useful rejections, not evidence that calibration or temporal learning can never help. [Calibration selection](round2_xgb_lio_platt_selection_2021.json), [smoothing probe](round2_temporal_probe.json).

A separate scalar-temperature calibration fitted on 2021 added only 0.00001288/0.00001096 to ensemble AUC in 2021/2022. It was not included in the final recipe. [Temperature comparison](round2_temperature_comparison.json).

[RDrop](https://proceedings.neurips.cc/paper/2021/file/5a66b9200f29ac3fa0ae244cc2a51b39-Paper.pdf) completed 39 epochs, selecting epoch 29. Its original/new mean scored **0.988436752** in 2021: above the starting winner but below the predeclared task-weighted threshold of 0.988648944. Stochastic tree sampling's best combined recipe scored **0.988658626**, below 0.988918450 for the selected combination. Neither advanced to 2022; these are failed advancement screens, not measured 2022 failures. [RDrop selection](round2_transformer_rdrop_selection_2021.json), [sampling selection](round2_sampling_combined_selection_2021.json).

The signed transition-loss control completed 39 epochs, selecting epoch 37. Its best 2021 ensemble variant scored 0.988534430, below the selected weighted-Transformer mean at 0.988648944; it did not advance to 2022. [Transition selection](round2_transformer_transition_selection_2021.json).

The final screen used BCE plus a sampled pairwise logistic loss, inspired by [RankNet](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf). It compares ON/OFF examples across the whole weekly batch using a differentiable ranking surrogate. Its epoch-31 prediction mean scored 0.988507871 on 2021, below the weighted-Transformer threshold, so it did not advance to 2022. [RankNet selection](round2_transformer_ranknet_selection_2021.json).

## What still fails

The selected candidate reduces pooled misordered positive/negative pairs by about 5.5%/4.9%. Improvements occur in all 2021 months and 11 of 12 in 2022. Generalization is therefore not failing uniformly.

Hours within two hours of true ON/OFF transitions constitute **8.05%/11.87% of outputs**, yet **50.78%/67.80% of remaining wrong pairs** involve them. Relative to available pairs, these are 3.27×/2.77× concentrations. Transition-region AUC remains only 0.8671/0.8724. These label-derived masks are diagnostics, not prediction inputs; pair shares overlap and cannot be added. Real one-hour runs rule out imposing an unsupported hard minimum duration.

Lio remains unstable: its 2022 AUC falls from **0.964884 to 0.961617**, despite the overall gain. Haukeli improves to 0.958094 but remains difficult. Both depth 8 and the weighted neural mean separately hurt Lio in 2022. [Selected-candidate diagnosis](round2_selected_candidate_diagnosis.json).

Economic rescaling already removed most raw-price extrapolation. Price magnitude alone is not a stable explanation after controlling for case composition. A frozen ensemble trained before 2021 scores 0.985722 on 2022, below the later reference, but training history, data quantity, selected counts and distribution all differ; this is not a clean overfitting measurement. [Generalization analysis](round2_generalization_review.json), [frozen-model comparison](round2_frozen_ensemble_2022_comparison.json).

The task, data descriptions and topology review found no demonstrated time/output alignment bug. Supplied files do not establish hidden initial commitment or minimum-duration rules. Configurable [SHOP generator features](https://docs.shop.sintef.energy/objects/generator/generator.html) are not proof those options generated these labels. [Problem-formulation review](round2_problem_formulation_review.json).

## Validation strength and limits

Training excludes six overlapping starts before validation. Promising new controls freeze 2021-selected counts for 2022, saving the final checkpoint without 2022 early stopping. **Both years are heavily reused development data.** Older reference components already used their own validation results. Each new control used one seed; repeated-seed uncertainty was not measured. The Jan 7 check removes one overlap source, not that reuse. No competition labels are used. [Protocol review](round2_validation_protocol_review.json).

An independent reconstruction reproduced both selected historical scores exactly using the unchanged supplied `kaggle_metric.py`. It verified all 365 IDs and 2,352 output columns, their order, binary labels and finite probabilities before scoring. This checks historical implementation; the final exported submission has separate verification described below. [Official-metric verification](round2_official_metric_verification.json).

Whole-case moving-block bootstrap uses 7/14/28-day blocks, keeping all 2352 outputs together. The 14-day 95% delta intervals against the original are [0.000311, 0.001130] and [0.000317, 0.000984]. Evidence is weaker against the improved fallback: some intervals cross zero. These conditional intervals do not account for repeated model selection or guarantee a leaderboard gain. [Uncertainty report](round2_depth8_combined_uncertainty.json).

## Deferred experiments

Prepared but unrun primary controls are longer depth-6 training, larger child weight, stronger global tree penalties, input-boundary reweighting, lower neural learning rate, plateau scheduling and stronger weight decay. Mixup and label smoothing were not fitted. Unused 2022 config files are templates, not frozen confirmations; their counts and stopping rules must be fixed from a prior selection before future use. Additional historical folds and repeated-seed checks were not completed. Earlier seed ensembles, EMA/SWA, dropout and scheduler experiments already exist and were reviewed rather than presented as new tests. [Neural research](round2_neural_research.json), [live status](autonomous_round2_20261014.json).

The next scientific priority is to test whether transition and Lio improvements persist across additional predeclared chronological windows, with model selection restricted to earlier periods. Lower learning rate and stronger weight decay are still untested controls in this round. The simple transition-loss and ranking-loss controls already failed their advancement screens; repeating those unchanged is not the recommendation.

## Final artifact verification

The new depth-8 XGBoost, Haukeli specialist and task-weighted Transformer were fitted on all **2,922 supplied labeled cases**, using the frozen counts of **748 trees, 121 trees and 28 epochs**. Training IDs do not overlap the 725 competition IDs. Neural scaling was fitted on training inputs. The task-weighted model is used only as an ensemble component; heads without direct supervised loss have zero contribution.

Saved-checkpoint predictions were independently replayed on 17 fixed competition cases per new model. All three passed, with maximum absolute differences below 2.981 × 10⁻⁸. This is a sampled model replay, not a claim that every case was independently rerun. [Depth-8 replay](round2_depth8_checkpoint_replay.json), [Haukeli replay](round2_haukeli_checkpoint_replay.json), [Transformer replay](round2_weighted_transformer_checkpoint_replay.json).

The final CSV contains **725 rows × 2,352 probabilities**, plus `Run No`, in the exact sample-submission order. All **1,705,200 probabilities** are finite and within [0, 1]. A separate verifier recomputed the entire six-source blend column by column and obtained **zero difference**. It checked selected configs, counts, feature order, checkpoint/prediction hashes and the original sources. All **185 protected originals**, including raw data, `kaggle_metric.py` and the previous best artifacts, remain unchanged. The neural exporter did not persist a training-ID hash during fitting; verification instead records the independently reconstructed loader cohort and checks its count/date bounds and unchanged source code. [Final verification](round2_final_verification.json), [export provenance](submission_round2_depth8_haukeli_weighted_run.json).

Submission SHA-256: `1e977991cff10a4a75a1555f42417701f55c6a4b7c37ab68560314ca7dd6888b`.

To reproduce the blend from the saved component predictions, use a new output filename:

```bash
.venv/bin/python src/make_goal_submission.py \
  --config configs/round2_submission_depth8_haukeli_weighted.json \
  --output submissions/submission_round2_reproduced.csv
```

The exporter refuses to overwrite an existing artifact. The public score above is user-reported; the private Kaggle score remains unknown. Historical selection and all model artifacts were fixed before this public result was supplied.
