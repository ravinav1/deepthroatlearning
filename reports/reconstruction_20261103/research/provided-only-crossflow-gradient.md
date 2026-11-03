# Shared-conduit marginal externality

The conditional PQ update includes the effect of a unit's flow on its own head, but omits its effect on every other unit sharing a conduit. This is a different mechanism from the previously rejected reservoir-storage head feedback: initial gross heads, terminal water values and reservoir topology stay fixed here.

The method research source is SINTEF's **13–14 March 2019** user-meeting presentation, [Getting Energy Conversion Right](https://www.sintef.no/contentassets/6b622776fc0f4bab9b5168d69ba93da1/getting-energy-conversion-right-pq-curves-jiehong.pdf), printed slides 11–14. The slides distinguish previous-flow PQ updates, proportional-flow PQ updates, and an aggregate busbar formulation. They explicitly describe oscillation of the previous-flow method near the terminal water value. They describe the busbar method using a convex piecewise-linear approximation of aggregate cubic conduit losses. The cover date, not the website upload/crawl date, determines the citation year. This is evidence about available methods, not proof of the competition's exact SHOP17.17 invocation. No numerical values from those slides or any public solver outputs enter this diagnostic.

## Derivation

For a unit i, let Q be plant flow and Qg its penstock-group flow, in m³/s. Its effective head is

`h_i = H_i - k_main*Q² - k_group*Qg²`.

At constant turbine and generator efficiency, define `C_i = 0.00981*eta_t_i*eta_g_i`, so `P_i = C_i*q_i*h_i` in MW. The derivative of total plant power with respect to q_i includes its own conditional-PQ derivative plus

`E_i = -2*k_main*Q*sum(j != i, C_j*q_j)`

`      -2*k_group*Qg*sum(j != i and j in same group, C_j*q_j)`.

This externality is normally negative. A price-weighted conditional objective must therefore include `price_t*E_i*(q_i-q_i_ref)`. The reference constant can be omitted when optimizing, but must not be confused with physical power or actual profit. With LP flows measured as Mm³ per hour, the gradient coefficient is divided by 0.0036. Negative prices reverse the economic sign; the formula must not clip the correction to a nonnegative penalty.

For variable efficiencies write `P=A(q,h)*eta_g(P)`, with `A=.00981*q*h*eta_t(q,h)`. Away from interpolation knots,

`dP/dh = A_h*eta_g(P)/(1-A*eta_g'(P))`,

`A_h = .00981*q*(eta_t + h*deta_t/dh)`.

Replace `C_j*q_j` in E_i with `dP_j/dh_j`. The implementation uses a prespecified ±0.001m central head difference through the existing supplied-data power evaluator, including its eight generator-efficiency iterations. This avoids claiming an exact implicit fixed point that the current evaluator does not solve. At a knot this is a symmetric finite-difference estimate rather than a unique derivative.

Two analytical tests verify the constant-efficiency formula and that the own conditional derivative plus externality matches a direct finite difference of simultaneous total power, including two distinct penstock groups.

## Bounded diagnostic, frozen before solving

`docs/provided_crossflow_gradient12_recipe.json` freezes twelve existing cases `original36[::3]`, spanning 2015–2020. Source, original raw-data and local-cache hashes are pinned. No model is trained and no validation-year targets are opened.

The four reported arms are original linked-state LP, previously sealed conditional-Jacobi iteration5, bounded conditional updates without the gradient, and identical bounded updates with the gradient. Both bounded arms start at the original LP, then make exactly four proposals using the unchanged hourly PQ/network solver. Each proposal is evaluated at steps `[0.5,0.25,0.125,0]`; accept the first whose coupled-power merit does not decrease by more than 1e-6. The largest step is bounded at one half. The merit uses simultaneous nonlinear power for Tokke/Vinje, unchanged other-unit surrogate revenue, supplied terminal storage value and recomputed startup/shutdown, spill, bypass and slack costs. Step choices use supplied inputs only. All physical results are sealed before decoding the twelve original label rows.

This is a controlled sequential-approximation diagnostic, **not a convergence proof**. An exact first-order SCA would also require the own-power approximation to touch the same physical surface and enforce consistently approximated coupling constraints. Our four-segment concave origin-inclusive hull differs from raw nonlinear power, and changing reference flows changes unit bounds. Convex interpolation preserves the original linear water constraints, but does not prove updated nonlinear generator feasibility. The run reports the original state-link violation under raw physical power, hull-to-physical gap, merit progression, accepted steps and maximum flow changes. It never reports interpolated duals as valid shadow prices. A zero step can mean rejection of the direction, not stationarity. Fixed iterations can still yield reproducible features; that requires a separately approved cache/model experiment.

Run command:

```sh
.venv-ml/bin/python scripts/provided_crossflow_diagnostic.py --recipe docs/provided_crossflow_gradient12_recipe.json --run-id provided-crossflow-gradient12-001
```

Full source and numerical results are recorded in the run manifest. Root separately owns the aggregate busbar-loss formulation; this branch does not implement or approximate that alternative.

## Twelve-case result and prospective extension

The frozen12 run completed in 35.97 seconds. Manifest SHA256: `b0d1485d5e1bbe79cc4230d4a114446d2495f0f60d731cb47e666d0ba167a75a`.

| Arm | Flow-ratio AUC | State AUC | State cutoff errors |
| --- | ---: | ---: | ---: |
| Original linked LP | 0.949926896 | 0.932563326 | 2668 |
| Conditional Jacobi iteration5 | 0.958535674 | 0.949293890 | 2438 |
| Bounded, no gradient | 0.955407694 | 0.949216867 | 2425 |
| Bounded, cross-unit gradient | 0.967994105 | 0.959134631 | 2297 |

The gain is uneven: Vinje and Haukeli improve while Lio's within-unit flow-ratio AUC falls from 0.91184 in the matched bounded control to 0.88531. Maximum original linear-network residual is 3.7e-13. Both arms reach a raw-physical state-link violation of 0.02146 and an intermediate hull power gap of 7.65MW. The gradient arm's final maximum flow change is 14.71m³/s; convergence is not established. Aggregate merit gains are 21,645 for the gradient and 37,480 for the matched control, in the unchanged model's monetary units. Thus the label improvement is not evidence that the gradient arm found a better nonlinear optimum. Both merit paths are nondecreasing individually; 19 of48 gradient proposals take the zero step.

After observing this result, root authorized an unchanged extension to all existing36 cases. `docs/provided_crossflow_gradient36_recipe.json` has SHA256 `bdb149bd851d205685d257a2acdedfd99798a967beaf3c06268aaa136331bd07`. It freezes the same equations, steps, iterations, controls and scoring. The reused12 and additional24 are reported separately. The additional24 are prospective for this gradient method; their labels were already used in earlier physical diagnostics and are not described as a globally untouched holdout. No full cache is launched by the diagnostic itself.

The36 extension completed in114.23s, manifest SHA256 `070b19d56853e1093d5922ecfbbe43706a928aa311f8884f5aa20ecdebf8322f`. Its additional24 results support the AUC improvement:

| Additional24 arm | Flow-ratio AUC | State AUC | State cutoff errors |
| --- | ---: | ---: | ---: |
| Original linked LP | 0.942608774 | 0.945861674 | 5393 |
| Conditional Jacobi iteration5 | 0.941190553 | 0.944043408 | 5432 |
| Bounded, no gradient | 0.945004823 | 0.948342239 | 5320 |
| Bounded, cross-unit gradient | 0.950371153 | 0.953235033 | 5323 |

The gradient makes three more cutoff errors than the matched bounded control on the additional24, while improving AUC and beating the original baseline's cutoff errors. Across all36, its flow-ratio AUC is0.956101619 against0.948180827 for the matched control, and cutoff errors are7620 against7745. The reused12 reproduce the original results exactly. All36 retain zero soft slack, maximum linear-network residual3.7e-13 and the same explicitly unproven nonlinear feasibility. Independent saved-array/rank/original-row audit passed at `artifacts/metric-trick-gradient36-audit-001/manifest.json`, SHA256 `3cc5ce491068e798c87d07eefc6923dabd240506cb0ecbe4cf940ee9d43a5ab3`; this audit does not claim new LP replay.

Root then authorized a separate four-worker input-only3647-case cache. `scripts/provided_crossflow_cache.py` and `docs/provided_crossflow_cache_recipe.json` retain the exact five-iteration bounded method, save both final proposal and accepted schedule, and replay each. Failures discard the entire partial path and preserve the original local LP schedule, availability0. The15 fields use the same own-unit/state/fill layout as the conditional cache, but column9 is the dimensionless final cross-unit derivative divided by the final-proposal average conversion slope. No mixed-schedule dual is used. The exact field names and source/data/parent/diagnostic hashes are pinned in the recipe. A full cache is not a validation, training or submission result.

The complete cache finished in1532.78 seconds with all3647 cases and zero fallbacks. Its manifest is `artifacts/provided-crossflow-cache-001/manifest.json`, SHA256 `14fe1ed5ee1d271e1f083938d3d764d3f3390fc341300d0548e26e5f4bfd8230`. The independent full-cache audit passed at `artifacts/provided-crossflow-full-cache-audit-001/manifest.json`, SHA256 `42ee6f95cbeb1629c46178d25115af98a2ac79f4e6d058ea720515e722ba600a`. It reconstructed every15-field feature tensor and aggregate byte, checked all case/date/unit/source/input mappings, replayed the saved final mixture arithmetic and checked original linear water balances. It relies on the producer's full LP replay and reviewed physical evaluator; it does not certify nonlinear feasibility or repeat all LP solves independently.

The first pooled LightGBM144 experiment retained the complete canonical115 prefix, then appended the15 own-unit gradient fields and14 simultaneous unit states. The original purged40% training sample,1400 trees,31 leaves and other settings were unchanged. In `artifacts/provided-crossflow144-2021-001`, the fixed2021 grid selected weight0.25 into the clean canonical115 anchor: micro AUC0.9953035361075371 versus0.9952596524075921. Independent saved-array/rank/scoped-label audit passed at `artifacts/provided-crossflow144-2021-result-audit-001/manifest.json`, SHA256 `a067d39d5e40282c61712c41f5aaf8b517951f2c9bbcb13f249cbd36e1332005`. This modest2021 improvement did not survive unchanged2022: the fixed0.25 blend scored0.9940156938979107 against anchor0.9940509264719491 and was rejected. The independent2022 audit passed at `artifacts/provided-crossflow144-2022-result-audit-001/manifest.json`, SHA256 `c1086854eebac16571b76717fd2043eb762a44f033ab0f2f7a8f5d3dee3e83d9`. No final model or submission follows from this failed LightGBM blend. The separately proposed scalar CatBoost experiment chose gradient features from2021 before this2022 result and retains its independent learner/control gate.

A later fixed12 diagnostic tested physical-merit-maximizing steps `[1,.5,.25,.125,.0625,0]` and exactly20 iterations, reporting both iteration5 and20. It retained constant heads, all physics and original12 cases; no labels affected step choice or stopping. Results were negative: flow-ratio AUC0.9657790235854111 at5 and0.9657772648417243 at20 versus the original first-acceptable-step5 result0.9679941046911619; cutoff errors increased2297→2308. Of228 updates,210 chose zero. Independent audit `artifacts/provided-crossflow-convergence12-audit-002/manifest.json`, SHA256 `39aa950511f878833b3b025c1c9c1fcc06f1b70c98b94fd3148d403134ff7385`, reproduced every recorded fixed-grid choice, all24 saved checkpoint proxies and physical merits, scoped labels and rank scores. It did not repeat LP solves or recompute unsaved intermediate trials. Audit001 is preserved as a checker failure: a reused probability-only AUC guard incorrectly rejected valid unbounded flow/power ratios; v2 uses unrestricted finite-score ranks without clipping or changing any candidate. No extension or cache was authorized for this negative convergence experiment.

The full cache completed all3647 cases with zero fallback in1532.78s. Its manifest is `artifacts/provided-crossflow-cache-001/manifest.json`, SHA256 `14fe1ed5ee1d271e1f083938d3d764d3f3390fc341300d0548e26e5f4bfd8230`; aggregate features SHA256 `b3717e0510cff40bf23ae0fdff0c015c622b5703e2feb30325660e025ca63ddd`. Independent full-cache audit passed in23.42s at `artifacts/provided-crossflow-full-cache-audit-001/manifest.json`, SHA256 `42ee6f95cbeb1629c46178d25115af98a2ac79f4e6d058ea720515e722ba600a`. It checked every field/aggregate byte, input/source/artifact closure, exact IDs/dates/unit order, final accepted mixtures and original water balances. It relies on the producer's complete per-case LP replay and does not claim nonlinear feasibility. Root authorized the separately locked canonical144 chronological model experiment after this receipt; results belong to that model's own manifest.
