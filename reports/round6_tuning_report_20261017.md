# LightGBM capacity regularization and training length

Controlled tuning found **no robust improvement over Round5**. The selected change removes L2 regularization from the generator-specific LightGBM component. Native validation gains are tiny, their bootstrap intervals include zero, and the older frozen models transfer worse. **Keep the existing Round5 submission, user-reported public AUC 0.99172, as the recommended reference.** One [verified experimental CSV](../submissions/submission_round6_lgb_l2zero_experimental.csv) is available; its Kaggle score is unknown.

## Historical confirmation

The recipe stays at 50% Round3 plus 50% generator-specific LightGBM, uniformly averaging seeds 299 and 673. The comparison baseline is the actual Round5 ensemble, not Round3 alone.

| Evaluation | Round5 AUC | L2 zero candidate AUC | Gain |
|---|---:|---:|---:|
| 2021, trained before 2021 | 0.990374846 | 0.990385555 | +0.000010709 |
| 2022, trained before 2022 | 0.989497911 | 0.989532759 | +0.000034848 |
| Same 2022 fit, Jan 7 onward | 0.989411166 | 0.989445826 | +0.000034660 |
| 2022, pre-2021 training and no refit, Jan 7 onward | 0.987045304 | 0.986964767 | -0.000080537 |

The native fits pass the declared positive-gain export gate. The frozen result was a diagnostic, not an export gate, and makes this candidate exploratory rather than a recommended replacement. Refitted 2022 improves in two of four quarters; frozen 2022 regresses in all four. Native and frozen evaluations share 2022 labels.

Paired circular 14-day whole-case bootstrap intervals, using 200 draws, are approximately [-0.000054, +0.000088] for 2021, [-0.000075, +0.000147] for native 2022 Jan 7 onward, and [-0.000187, +0.000006] for frozen 2022 Jan 7 onward. All include zero. These intervals condition on the selected predictions and do not correct selection optimism. Both years are reused development data.

## Controlled comparisons

All rows below use seed 299 and the same 113 input features. The last column compares the fixed single-seed ensemble, 50% Round3 plus 50% candidate, against the matched Round5 seed-299 recipe. Its baseline AUC is 0.990274871. This screen is distinct from the actual two-seed confirmation above.

| Change | Training AUC, all hours | Standalone validation AUC | Ensemble AUC change |
|---|---:|---:|---:|
| Reference | 0.999306 | 0.988261 | +0.000000000 |
| 31 leaves | 0.998496 | 0.988496 | +0.000005580 |
| 127 leaves | 0.999621 | 0.987897 | -0.000148347 |
| L2 = 0 | 0.999429 | 0.988259 | +0.000023906 |
| L2 = 16 | 0.999079 | 0.987987 | -0.000171748 |
| Minimum leaf 300 | 0.999217 | 0.988442 | -0.000022231 |
| Half the rounds | 0.998323 | 0.986949 | -0.000247889 |
| Double the rounds | 0.999690 | 0.988470 | -0.000006011 |
| 255 histogram bins | 0.999374 | 0.988308 | -0.000084502 |
| 31 leaves and L2 = 0 (adaptive) | 0.998681 | 0.988434 | +0.000011684 |

Increasing capacity to 127 leaves fits training better but lowers validation performance. Halving the original per-generator tree counts is clearly worse; doubling them produces little ensemble change. Several changes improve standalone LightGBM AUC while worsening the blend. Removing L2 has almost no standalone benefit, so its small ensemble gain cannot be described as a uniformly stronger classifier.

## What the diagnosis shows

The reference LightGBM component scores 0.999306 on all training hours, 0.999692 on fitted sampled hours, and 0.988261 on 2021 validation. Its largest within-generator gaps are Haukeli (0.999160 to 0.951978), Lio (0.999941 to 0.965676), and Hogga (0.997484 to 0.974280). Training scores are optimistic because they are in-sample and cases overlap heavily.

The tested changes do not identify a substantial validation or ensemble improvement. They support investigating transfer across years and generator-specific weaknesses, without establishing that overfitting is the sole cause. The frozen result is relevant to the longer gap before private 2024, but is not a prediction of its score: the training cohorts and operating conditions differ.

## Selection and reproducibility

Eight one-factor variants covered leaf count, L2, minimum leaf size, boosting length and histogram resolution. The histogram check and one combined setting were added adaptively after partial 2021 results, documented before any candidate-2022 evaluation. The combination joined the two individually positive settings, 31 leaves and L2 zero; it did not beat L2 zero on the declared screening metric. No further search extensions followed.

Feature construction, relative monetary scaling, learning rate 0.04, rotating stride-4 sampling and all unspecified parameters stayed fixed. Weekly features use all 168 hours before training-row sampling. The original tree schedule came from the earlier 2019/2020 fits; half/double variants apply their multiplier afterward, allowing 65 to 2,500 trees per generator.

The winner was locked on 2021 before fitting the second seed and before candidate-2022 evaluation. Its uniform seed mean had to improve actual Round5 on 2021; both native-2022 subsets also had to improve. There was no fallback after seeing 2022. Splits remove training cases whose target week overlaps validation; Jan 7 onward also removes six starts overlapping the late-2021 selection horizons. Features and blend weights were not tuned in this run.

[Locked selection](round6_selection_2021.json), [2021 paired results](round6_l2zero_paired_2021.json), [native 2022](round6_l2zero_paired_native_2022.json), [frozen 2022](round6_l2zero_paired_frozen_2022.json), [combination protocol](round6_combination_protocol_20261017.json), [histogram protocol](round6_histogram_protocol_20261017.json).

## Delivery and verification

The experiment finished its fitting and verification in approximately **70.7 minutes** on 17 October 2026, within the two-hour budget. The only new submission is [submission_round6_lgb_l2zero_experimental.csv](../submissions/submission_round6_lgb_l2zero_experimental.csv). It is a probability average with the same fixed recipe; no model settings or blend weights were changed after the 2022 checks.

Both final seeds fit all **2,922 supplied labelled cases**, with zero competition-ID overlap. The file contains **725 rows and 2,352 probabilities plus Run No**, in template order. All **1,705,200** blend probabilities and every seed-mean probability reconstruct exactly. Each of the **28 saved final models** replayed exactly on five fixed test cases; this is sampled model replay, not an independent replay of every test case. All **1,190 protected original files** remain unchanged, including raw data, earlier models/submissions and kaggle_metric.py. Heavy fitting has stopped. The automated run performed no Kaggle upload or git push.

[Final verification](round6_final_verification.json), [export recipe](../configs/round6_submission_lgb_tuned.json), [export provenance](submission_round6_lgb_l2zero_experimental_run.json), [run ledger](autonomous_round6_20261017.json).

The implementation is isolated in new [training](../src/round6_lightgbm.py), [evaluation](../src/round6_evaluate.py) and [replay verification](../src/round6_verify.py) scripts. For a later rerun, copy a configuration to a new filename and update or remove deadline_utc; existing run artifacts are deliberately preserved.
