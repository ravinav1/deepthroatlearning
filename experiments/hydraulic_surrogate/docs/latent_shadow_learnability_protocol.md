# Latent water-shadow learnability protocol

Prepared before target expansion or a learned fit, on 2026-11-02. This protocol follows the accepted finite-grid reachability screen `latent-shadow-reachability-001`; its oracle is not predictive evidence. Root approved the concept, splits and gates. The reviewed source closure and successful tests are frozen in the companion JSON before target generation; later results belong in a separate results document.

The initial question is whether a small model can predict useful distributions over the already fixed water-shadow displacements from legal case inputs. It does not tune the physical feasible domain, displaced-water formula, switching convention, 17-value grid or target plants. It does not require the original SHOP optimizer or optimized discharge labels.

## Exact data partitions and execution order

Training cases start 2015-01-01 through 2018-12-25 UTC. The six starts December 26–31, 2018 are excluded. The last retained training horizon ends before the first screen horizon begins. Screen cases start 2019-01-01 through 2020-12-25, reported separately for 2019 and 2020. All 2021/2022 target values are excluded from this experiment. The screen years have appeared in the 36-case diagnostic, so they are reused development evidence, not untouched holdouts.

1. Freeze source/data/cache hashes and this configuration after review. Generate only the training partition's eight input features and all 17 hypothetical schedules. Decode labels only for those case IDs and Haukeli/Lio/Hogga.
2. Fit the fixed real-input mapping, a constant-distribution control, and a seasonal target-permutation control using only the training cache. Save coefficients, normalizers and training diagnostics; lock their byte hashes before generating screen targets.
3. A separate screen-target command requires the completed training lock and its explicit SHA256. It builds the same input features and grid schedules, then parses only the 2019–2020 screen labels.
4. Evaluate the unchanged locked models once. Preserve all outcomes. No coefficient, grid, prior, penalty, scope or output mixture is changed after the screen. A failed screen stops this branch before 2021.

No saved 2019/2020 validation arrays for the exact four-source anchor were found. Predictions from its models trained through 2020 would be training predictions and must not be used as held-out evidence. This stage reports three-target pooled AUC and DP errors explicitly. It cannot establish whole-14-unit anchor improvement or either .998 gate. If this stage passes, a separate later protocol must train only before 2021, lock a whole-matrix blend on 2021, and test unchanged 2022.

## Frozen physical candidate family

Use `GRID = linspace(-1, 1, 17)`; targets are Haukeli_G1, Lio_G1 and Hogga_G1. Let `s = 1 + mean(abs(price))` and `E` be the frozen v1 standalone initial-head best-energy factor. For each displacement `d`, use `nu_t(d)=nu_t+d*s*E`, in EUR/Mm3. Compute hourly on-profit on the original sampled feasible P(q) domain, retaining negative values and applying `.0036` exactly once. Use supplied start/stop costs and the same free-boundary, off-first-tie two-state path decoder as the completed screen. No LP solve, head update, directional-pool rewrite or guessed teacher command enters this experiment.

Every case's `d=0` rewards and nonlinear-PQ features must replay the immutable cache exactly. Cache IDs/dates, raw-input hashes, case LP archives and the frozen source closure are checked. Save the legal feature vector, full17-grid rewards, paths, errors and target distribution per case. Each case file is independent, keeping target-generation RAM bounded; generation is resumable only with exact matching source/config/data hashes and retained completed case files.

## Eight input features per case/unit

| Field | Formula |
| --- | --- |
| Direct-intake initial fill | Initial volume of the first source reservoir / its capacity. |
| Frozen connected-pool initial fill | Sum initial volumes in the existing v1 pool / sum capacities. This deliberately preserves the known Haukeli pool approximation. |
| Direct natural weekly inflow | `.0036*sum(168 hourly direct-reservoir inflow)/direct capacity`. |
| Relative terminal local water value | Supplied source-reservoir EUR/MWh at start+7 days divided by `s`; not a water-volume subtraction. |
| Mean normalized local LP shadow | `mean(nu)/(s*E)`. |
| Std normalized local LP shadow | `std(nu)/(s*E)`, population standard deviation. |
| LP zero-discharge fraction | Fraction of own unit hours whose frozen LP discharge is exactly zero. |
| Inverse monetary scale | `1/s`. |

The direct and pool fill columns can coincide for single-reservoir pools; retain the declared schema and identity shrinkage. Constant training columns map to zero. Only case-start storage is read; later observed storage, solver timing, actual schedule fractions and validation labels are excluded from inputs. Known full-horizon natural inflow and supplied terminal water values are legal inputs. Feature means/stds are fitted on the training partition only, with zero variance replaced by one; standardized values are clipped to `[-5,5]` using those frozen statistics.

## Ambiguous inverse targets and small predictive model

For each training case/unit, `e_k` is the number of hourly mismatches between its observed labels and the MAP schedule for displacement `d_k`. Preserve the entire vector and all minimum-error sets. A set can be disconnected; never turn its convex hull into a claim that every intermediate displacement is equally good.

Use the fixed regularized soft target

```
log prior_k = -d_k^2/(2*.25^2)
q_k proportional to exp(-(e_k-min(e))/2 + log prior_k)
```

The scale of 2 is in mismatch-hours. It and the zero-centered prior are fixed modeling choices, not inferred physical uncertainty. Preserve the minimum-error intervals, the indices within one hour of optimum, and the central 90% grid interval of `q`; report their widths. This uses the inverse uncertainty rather than an arbitrarily unique case optimum, while the prior pulls unidentifiable cases toward the unchanged shadow.

Fit three multinomial linear maps, one per target unit:

```
pi_k(x) = softmax(log prior_k + W_k @ [1, standardized_8_features])
loss = mean_case cross_entropy(q, pi) + .01 * sum(W^2)
```

Each map has 17×9 coefficients. Penalize intercepts too, start all coefficients at zero, use deterministic L-BFGS-B with analytic gradients, maximum500 iterations, `ftol=1e-12`, `gtol=1e-8`, one BLAS thread. Require optimizer success and finite values; no early stopping or selection among restarts. The fixed objective is convex in W; the ridge penalty removes the softmax shift ambiguity.

At inference the input-only `pi` weights the same17 physical MAP schedules: `p_t = sum_k pi_k * u_t(d_k)`. This remains in `[0,1]`, uses no case labels, and can change timing as well as on-fraction. No calibration temperature, learned case-fraction target or further blend is fitted. Record the inferred expected displacement and central90% interval as diagnostics; neither is an identified true water value.

Controls use the same training targets, regularization and inference operation. The constant control fits only the17 intercepts, with all eight feature coefficients zero. The permutation control uses the full eight features but permutes training soft-target vectors within each calendar-year/season/unit group using seed4173. Also retain the unshifted `d=0` path and the fixed zero-parameter prior mixture; these isolate physical reachability and smoothing across latent displacements from learned case information.

## Predictive screen and falsification gates

Report separately for 2019 and 2020: soft-target cross entropy; expected mismatch under predicted `pi`; mismatch after thresholding final marginal probabilities at `.5` with off-first ties; mean distance of expected displacement to the observed acceptable grid set; central90% interval coverage of that set; three-target pooled AUC; per-unit AUC; on-fraction and switch-count error; errors within ±3hours of true switch boundaries and elsewhere. Any target-set/coverage quantity uses screen labels only for scoring.

The fixed promotion screen requires, in **each** year:

- At least10% fewer thresholded hour errors than both the unshifted path and the fitted constant-distribution control.
- Lower expected hour mismatch and soft-target cross entropy than the constant and permutation controls.
- Higher three-target pooled AUC than the constant and permutation controls; preserve all unit scores and require no unit's thresholded hour error to exceed its constant control by more than5% (if the control has zero errors, permit zero only).
- No optimizer/data/source/inference failure and no use of 2021/2022 labels.

These gates assess predictive economics within a weak physical schedule family. They do not authorize a submission, final-data fit, migration or claim about the strong whole-matrix anchor. If they pass, 2021/2022 integration remains a new experiment with an exact new anchor, unchanged code and a matched no-new-feature/control splice. A gain over input-only DP alone is insufficient.

Required tests before target generation: exact grid and source closure; unit conversion once; negative operating profit; d0 cache replay; selected-row-only label decoder with a poison later-year fixture; feature invariance to future storage/label modifications; soft-target symmetry/identity shrinkage/disconnected minima; analytic gradient finite differences; synthetic real-feature versus permuted-feature learnability; constant-control parameter restriction; case-isolated inference and `[0,1]` probabilities. Preserve every failure with a new run ID.
