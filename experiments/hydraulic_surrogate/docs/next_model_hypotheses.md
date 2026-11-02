# Next model hypotheses from preserved errors

Research owner: `xgb_large`, 2026-11-02. No model or calibrator was fitted for this review. The only new execution is a read-only pair-count diagnostic of existing frozen probability arrays. The proposals below are hypotheses, not accepted experiments or leaderboard evidence. Root owns browser consultation, hosted queues and submissions; this document makes no model-version claim about that consultation.

The diagnostic anchor is explicitly the four-source recipe in `artifacts/lp-context-joint-comparison-001/`, weights `[.140625, .421875, .1875, .25]` in base/context/joint/combined order. It is not silently replaced with any GPU candidate. A future experiment must freeze its exact anchor before fitting or selecting corrections.

The small **latent water-shadow reachability diagnostic** has now passed its predeclared screen on pre-2021 training cases. The next work is a separately reviewed learnability protocol; no predictive mapping has yet been fitted. Two other distinct mechanisms are bounded asymmetric probability curvature and schedule-level soft marginals.

## What the error decomposition establishes

`artifacts/pair-calibration-diagnostic-001/manifest.json` independently counts every directed positive-unit/negative-unit error pair. Its 14×14 matrices sum to `1 − micro AUC` within `1e-13`; diagonal cells also reproduce every individual unit AUC. Runtime was 1.113 seconds, without feature rebuilding, model inference, optimization or fitting.

| Diagnostic | 2021 | 2022 |
| --- | ---: | ---: |
| Frozen anchor micro AUC | .994937718669 | .993691487954 |
| Cross-unit share of all erroneous pairs | 90.25% | 91.23% |
| Cross-unit share of all compared pairs | 93.37% | 93.30% |
| Same-unit conditional pair error rate | .00744175 | .00825738 |
| Cross-unit conditional pair error rate | .00489323 | .00616856 |
| Loose ceiling if every cross-unit error vanished | .999506358706 | .999446750017 |

The large cross-unit error share does **not** establish that most error is caused by calibration: nearly all pairs are cross-unit in the first place. The ceiling is only a necessary bound for strictly monotone unit maps, and assumes simultaneous elimination of every cross-unit error; it is not an attainable calibrated result. Improving to .999 would require removing approximately 89%/92% of the present cross-unit errors while preserving within-unit ranking. Generic calibration has no demonstrated ability to do that.

Mean `p − y` for Haukeli changes sign, from `+.0211753` in 2021 to `−.0153459` in 2022. Lio remains positive (`+.0353145` / `+.0432136`); Hogga is `+.0175021` / `+.0128384`. The five largest directed error cells in 2022 all involve Lio negatives except the Lio/Lio diagonal. These are descriptive observations, not permission to select a Lio-only correction after seeing 2022.

The earlier exact decomposition in `artifacts/current-blend-residuals-001/` assigns Haukeli/Lio/Hogga 40.53% / 39.91% of total pooled error. Their case-mean Brier shares are 36.55%/32.12%/18.33% in 2021 and 34.72%/44.44%/18.58% in 2022. Hourly allocation therefore remains material, even where mean bias is visible.

## Prior failures that constrain the design

| Previous mechanism | Preserved evidence | Consequence |
| --- | --- | --- |
| Weekly fraction prediction followed by a case-wise logit shift | `docs/case_calibration.md`: selected weak-unit .5 blend gained .001749 in 2021, lost .000738 in 2022 | Do not repeat a generic fraction regressor or choose Haukeli alone retrospectively. |
| Linear/spline economic threshold models | `docs/economic_probe.md`: selected .1 Haukeli logistic blend gained .000293 in 2021, lost .000164 in 2022 | A small normalized margin model is already tested; improved log loss did not imply improved pooled ranking. |
| Shared logit slope plus 14 generator offsets | Reference `reports/round9_group_calibration_results.json`: H1-2021 fit, six-day purge, H2 loss .000483283508; confirmation not evaluated | A fresh implementation of the same affine calibrator is a repeat. An older weak-model calibration gained .000227/.000322 on 2022, so the evidence is context-dependent, not universally negative. |
| Lio Platt scaling | Reference `reports/round2_xgb_lio_platt_selection_2021.json`: identity .988955732003 versus Platt .988865848351 | Positive-slope preservation of unit AUC does not protect full-matrix AUC. |
| Plain moving-window probability smoothing | `docs/modeling_recommendations.md`: every nonzero tested window/weight lost the 2021 screen | Low switch frequency alone is insufficient justification for smoothing. |
| Fixed-dual nonlinear operating profit and max-margin DP features | `docs/nonlinear_pq_results.md`: 2021 .994977829721; unchanged .5 splice 2022 .993683153450 below .993691487954 anchor | Actual nonlinear P(q) alone did not solve the problem. Its water dual stayed frozen; its DP retained only max-marginal summaries. Preserve the failed lock and never retune that branch. |
| Monotonic margin/flow constraints, including exact aliases | `docs/monotonic_specialists.md`, `docs/monotonic_alias_specialists.md`: both lose 2022 | Additional positive-shape constraints are not a distinct correction for missing economics. |

## Mechanism 1: learn an economic water-shadow correction, after reachability

For a single-unit plant, let `nu_t = lambda_up,t − lambda_down,t` from the frozen input-only LP, in EUR/Mm3. Let `s = 1 + mean(abs(price))` in EUR/MWh and `E` be the frozen initial-head standalone best-energy factor in MWh/Mm3. Test a dimensionless case-level additive displacement `d`:

```
nu_t(d) = nu_t + d * s * E
r_t(d) = max over physical q>=Qmin [price_t * P(q) - .0036*q*nu_t(d)]
J_d(u) = sum_t u_t*r_t(d) - start_cost*N01(u) - stop_cost*N10(u)
```

Use the existing original unhulled P(q), feasible sampled operating domain and supplied asymmetric switching costs. Initial/terminal states stay free. Negative on-profit and negative local shadow differences are allowed; no sign flip, clipping-to-zero or duplicate `.0036` conversion is introduced. The additive shift works at zero/negative duals, whereas a multiplicative `exp(theta*z)` correction cannot move a zero dual and preserves its sign.

This corrects the operating opportunity cost before temporal scheduling, so it can change the ordering and position of within-week blocks. The rejected fraction correction only moved existing hourly logits; the rejected nonlinear-PQ feature used `d=0`. Neither tests this mechanism. However, the scalar shift cannot repair missing directional hydraulics, ramps, teacher boundary conditions or all time-varying constraints.

SHOP supports several water-value descriptions; supplied EUR/MWh values and missing case-generation commands do not establish which hidden teacher mapping is correct. A learned `d` would be a surrogate discrepancy parameter, not an identified physical water value. Inverse optimization uses observed decisions to infer objective parameters, but identifiability and model mismatch remain concerns. [SHOP water values](https://docs.shop.sintef.energy/examples/water_values/individual_water_values.html), [inverse-optimization review](https://arxiv.org/abs/2109.03920).

**Precise first diagnostic, executed unchanged as `latent-shadow-reachability-001`.** Choose 36 cases solely by dates: 12 uniformly spaced starts from 2015–2018, 12 from 2019, 12 from 2020 ending no later than December 25. Within each date block use rounded `linspace(0, n-1, 12)` positions. Inspect Haukeli, Lio and Hogga, with no error-based case selection. Parse label values only for those selected rows and three units. Keep all 2021/2022 labels outside this diagnostic.

Evaluate the fixed `d = [-1, -.875, ..., .875, 1]` grid. Decode the maximum-profit two-state path using deterministic off-first ties; save all 17 paths, objectives, normalized observed-path regrets, per-hour raw rewards and sample IDs. At `d=0`, recomputed rewards and max-margin features must match the immutable nonlinear-PQ cache. Compare three quantities: unchanged `d=0`, the best one fixed displacement across all 36 cases, and a label-informed best displacement separately for each case/unit. The last two are diagnostic oracles, never deployable predictions or validation scores. The best finite-grid path is not an upper bound for every possible continuous or time-dependent correction.

Primary rejection gate: require at least 25% lower total hour mismatch for the case/unit oracle than `d=0`, at least 10% reduction for each of the three units, and at least 10% reduction in each date block. Require less than 20% of case/unit optima at the grid endpoints. These conservative screening thresholds are proposed in advance to prevent expanding a tiny gain into an expensive model. Report regret, on-fraction error, switch-count error and boundary timing separately; do not select a favorable metric after the result. If the single constant displacement recovers essentially the same benefit, reject the more complex case-latent explanation and report that simpler finding.

Negative controls before any learned mapping: (a) adding the same hourly scalar to both on/off rewards must leave every path unchanged; (b) seasonal within-unit permutation of case-optimal displacements must remove most of the case-specific advantage; (c) physically zero discharge must incur zero added water cost. Check synthetic negative-price/negative-on-profit cases and compare short-horizon DP with exhaustive enumeration, including asymmetric costs and free boundaries. Use the current observed data only; no original optimizer, teacher dispatch amounts or re-solved labels are assumed available.

The completed screen reduced hour mismatches from 2,543 to 910 (64.22%): Haukeli 757→45, Lio 1,142→363, Hogga 644→502. Each date block improved by 28.54% / 77.94% / 61.64%, and no case/unit optimum reached a grid endpoint. The best constant displacement separately per unit leaves 2,034 errors, while the case-specific oracle leaves 910; seasonal within-unit permutation leaves 1,663. The permutation thus removes 66.99% of the oracle's advantage over the best constant-per-unit control. Most improvement lies away from true switch boundaries (2,366→746 errors); near-boundary errors only move 177→164. This supports a case-dependent economic discrepancy as a testable mechanism, without establishing its predictability.

All declared numerical gates passed. Recomputed `d=0` rewards and six nonlinear-PQ features matched the frozen cache exactly before selected labels were decoded. Eight synthetic/legal-input checks passed in `artifacts/latent-shadow-preflight-002/`. Real diagnostic runtime was 2.196 seconds and measured peak RSS 183,369,728 bytes (174.88 MiB). Full rewards, paths, all finite-grid errors/regrets, targets, permutation indices and source/data/cache provenance are retained in `artifacts/latent-shadow-reachability-001/manifest.json`, SHA256 `de72537abca5f019e031c3404dbe73b6e1b2b961684e28ac5e44e9a5a7591279`. Original pre-result research prose is archived in `artifacts/latent-shadow-research-summary-001/` because this document is being updated; the experiment manifest remains immutable.

If reachability passes, a separate reviewed protocol can infer a regularized interval/posterior over `d` from training labels, then learn it from at most eight prespecified legal case inputs: initial fill/headroom, natural inflow relative to capacity, minimum-release burden, terminal value/price scale, and LP storage/dual summaries. Use 2015–2018 to fit and 2019–2020 to check learnability before 2021 selection. Expand chronologically with the usual six-day purge; 2021 labels may enter the 2022 training partition, but not the 2021 fit or any post-lock weight tuning. Permuted-target and constant-displacement controls must pass the same splits. A case-level discrepancy must not be computed using that case's labels at inference.

## Mechanism 2: bounded curvature-only cross-unit calibration

A narrowly distinct cheap calibration test is a one-parameter asymmetric shape per generator:

```
g_u(p) = sigmoid(logit(p) + k_u * log(4*p*(1-p)))
|k_u| <= .25
```

This fixes `g(.5)=.5`, returns the exact original bytes when `k=0`, and is strictly increasing for interior probabilities. It is a restricted beta family with `a=1+k`, `b=1-k`, `c=k*log(4)`. It adds curvature unavailable to the failed affine-logit model without adding another freely fitted intercept or temperature. Its rationale is asymmetric confidence error, not a constant unit prevalence correction. Beta calibration supplies a general asymmetric family; it does not establish a gain here. [Kull et al., 2017](https://proceedings.mlr.press/v54/kull17a.html).

Fit only the 14 curvature coefficients with a fixed identity-centered ridge penalty on case-balanced log loss, using a purged H1-2021 calibration partition and H2-2021 screen if earlier predictions of the exact anchor are unavailable. Freeze coefficients after that screen, with no refit on H2 or 2022. Such a screen cannot be called a full-year out-of-fold 2021 result; producing that would require genuine earlier chronological base predictions. Prefer an all-unit test over selecting Lio from the 2022 bias.

Falsification: require a positive held-out whole-matrix AUC gain, exact unchanged within-unit ranking, no new clipped ties, and improvement not confined to one month. Identity is the only initial selection control; do not search extra offsets, free temperatures, price-regime bases or separate target scopes. A diagnostic common strictly increasing map must leave pooled AUC unchanged. The same-year full-fit AUC is not acceptance evidence. Because fixed affine corrections have already failed and biases change sign, this is lower priority than the reachability check.

## Mechanism 3: schedule-level soft marginals from the strong anchor

Use the existing case/hour probabilities as unary log potentials, rather than replacing them with a weak physical-only score. For each of the three prespecified single-unit plants, define a distribution over all 168-hour binary schedules:

```
P(u | inputs) proportional to exp(sum_t u_t*logit(p_t)
   - k*(start_cost*N01(u) + stop_cost*N10(u))/(s*Pmax))
```

Compute `P(u_t=1)` by log-space forward/backward recursion with free boundary states. A fixed global grid `k in [0, .05, .2, .8]` is sufficient for an initial screen; `k=0` must return the original probabilities exactly. Other generators remain unchanged. This uses asymmetric costs and a globally normalized distribution over complete schedules, so it differs from moving averages, a hard MAP schedule, the reference's local learned residual, and the rejected maximum-objective-difference feature. Such globally normalized sequence distributions have a primary-method basis; this particular hydraulic approximation is our proposal. [Ziebart et al., maximum-entropy inverse reinforcement learning](https://www.cs.cmu.edu/~bziebart/publications/maxentirl-bziebart.pdf).

The strongest risk is double-counting persistence already learned from temporal/LP features. Reject if it fails the unchanged anchor or if gains disappear after excluding the first/last 12 hours. Require a transition-free identity control, brute-force short-sequence marginal verification, exact isolation between cases, and reporting both correctly timed short blocks and long constant weeks. Keep negative-price cases and original `[0,1]` probabilities. Select the single global `k` only on 2021, lock it, then test the one unchanged 2022 result. No blend or per-unit weight search is needed initially. A favorable schedule likelihood or physically plausible path cannot substitute for pooled AUC.

All proposals require a fresh reviewed protocol, exact source/anchor/data hashes, retained failures and the existing chronological gate. No current result reaches .998 in both validation years or .999 with a verified top-three Kaggle rank. No remote transfer is justified.
