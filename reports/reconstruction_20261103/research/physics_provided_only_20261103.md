# Provided-data-only physics and pooled-model track

The user now excludes external data and asks for a model that retains first place. This track does not use downloaded solver outputs, external pseudo-labels, teacher models, public-solver feature selection or additional constraints. The earlier external-data result is a separate historical result and does not satisfy this constraint.

## Clean starting point

The canonical115 representation selects and aligns the original 664 supplied-input features into a common generator-relative schema. Its physics caches (`lp-full-001`, `lp-joint-full-001`, `lp-shop-full-001`) were computed locally from the supplied kernel/extended files and topology. Their `data` maps exactly match the ten provided file hashes in `docs/lgb_shop664_protocol.json`. The independent closure receipt is `artifacts/provided-only-lineage-audit-002/manifest.json`.

The accepted canonical115 continuation selected a .25 blend with the earlier clean LGB664 ensemble. Validation AUC is .9952596524075921 in 2021 and .9940509264719491 at the unchanged weight in 2022. Its final artifact is `artifacts/physics-trick-canonical115-final-001/submission.csv`, SHA256 `b422d6ca990b8d18e7d397060254af3a95fb3e5a0e1cdc191a88baa10c6f5783`. Root/metric own its clean export and submission; this agent does not submit.

The separately prepared canonical144 branch is excluded: it used downloaded hourly storage and an additional Staavatn maximum constraint. Its name and local availability do not make it provided-data-only.

## Supplied daily trajectories: rejected

`sintef_project/provided_canonical_trajectory_features.py` preserves the canonical115 prefix and adds 45 fields, giving 160 features. These comprise own-pool filling, daily storage change and inferred release, seven-day supplied storage/water-value context, direct normalized price-minus-water margins, downstream context, supplied minimum-volume slack and global pool filling/terminal water values. They use only supplied CSV/YAML files. Daily post-start historical storage is explicitly a retrospective competition covariate, not an operational forecast or reconstructed solver output.

Two focused tests pass. They guard input paths, exclude label reads from feature construction, verify case permutation and unit/plant row mapping, preserve prefix bytes, and independently check storage interpolation, water-balance units and terminal water-value margins. Structural source review found no blocker. Preparation: `artifacts/physics-provided-only-preparation-001`; recipe SHA256 `58adb3991d5e15d566e93b613bae564fadf00f7f16b1113c9523680e3fe962b6`.

The model retained 1400 trees, 31 leaves, learning rate .06, minimum child size 100, L2 1, feature fraction .9, original 40% sampled hours, seed 4173, four threads and no early stopping. Training used cases strictly before 2021 with the original six-day purge. Its full native checkpoint replay passed.

| Weight on canonical160 | 2021 micro AUC |
| --- | ---: |
| 0 | .9952596524075921 |
| .1 | .9952533265376748 |
| .25 | .9952113321419919 |
| .5 | .9950583402578539 |
| 1 | .9942691733162403 |

The declared grid selected identity. No rescue weight, 2022 fit, final fit or export follows. Runtime was 624.70 seconds. Checkpoint SHA256: `be5f3ddd0c84a41678e937df02213b16869fea8773a192c0318954c2d30d8db7`. All predictions, source/data hashes, sample IDs and configuration remain in `artifacts/physics-provided-only-canonical160-2021-001`.

## Endogenous head-value diagnostic: no expansion

The earlier head refresh updated PQ curves but omitted the marginal value of retaining storage to preserve hydraulic head. A distinct diagnostic adds the first-order storage term

`price × dP/dH × dH/dV`, in EUR per Mm³,

to the LP objective. It uses the frozen local LP's discharge/storage as the reference, supplied unhulled unit-alone PQ physics, and the supplied volume-to-head curves. Beginning/end-hour averaging contributes half of each hourly head derivative to each neighboring storage boundary. The fixed initial boundary has no decision variable. Bandak's contribution includes Hogga's positive upstream-head effect and Tokke's negative tailwater effect. Only the exact single-reservoir Bandak/Vinjevatn heads are changed; pooled-reservoir allocation is not invented.

The fixed sample takes every third case from the preexisting 36-case metadata selection, yielding 12 cases strictly before 2021. One updated-curve solve without the storage term is the matched control. One updated-curve-plus-gradient solve is the candidate. No damping, repeated iteration, threshold search or model training is performed. Original target rows are decoded only for these 12 cases after all predictions and replay checks are sealed.

The first diagnostic exposed an endpoint derivative error: a zero-volume reservoir received a zero slope, while the feasible direction has a positive one-sided slope. A separately preserved v2 corrects that boundary handling and uses one-sided finite-difference checks there. The cases, iteration count, .01m power derivative step, score function and all other settings remain unchanged. The first run is not overwritten.

| Fixed LP score | Baseline | Curves only | Curves + gradient, corrected |
| --- | ---: | ---: | ---: |
| Micro AUC, 12 pre-2021 cases | .9366445550 | .9312378195 | .9242649999 |
| Maximum head inconsistency | 0m reference | 1.358m | 3.439m |
| Maximum uncapped physical rating excess | .977MW | .135MW | .523MW |

Zero-gradient control replay and repeated candidate replay are bitwise exact. New solves have zero soft slack and aggregate water-balance error no larger than 7.73e-12 Mm³. The original cached baseline differs from a fresh float64 physical mass sum by up to 4.79e-5 Mm³, reflecting the baseline input/RHS precision; this is retained rather than called exact conservation. The maximum corrected derivative-check discrepancy is 1.17%, with the fixed head perturbation crossing piecewise efficiency knots. This is an approximate one-step linearization, not a fully converged nonlinear model.

The gradient variant worsens diagnostic AUC and nonlinear consistency. It does not justify a full cache or supervised extension. Records: `artifacts/physics-provided-only-head-gradient12-001` and `artifacts/physics-provided-only-head-gradient12-002`. The corrected run took 11.11 seconds. Its recipe SHA256 is `e17c7953082aaaa4a12d293e4d50ad49011929ec6dd7fa46966897b4e9d62a14`.

Other clean tracks are owned by root/metric: larger pooled capacity, explicit startup economics, and the supplied-input integer-physics features. No result from them is claimed here. The first-place target remains active under the provided-only constraint.


## Restore omitted original inputs: small 2021 gain, continuation deferred

The next fixed ablation restores all 201 original 664 columns absent from the union of canonical common/global/all own sources. These existing columns are broadcast to every unit at each case-hour, yielding 316 features while preserving every 115 prefix byte. No raw input or formula is added, and the failed retrospective 45 fields are excluded. This does not claim that every other unit’s own-source columns are present in each row. The restored fields include original reservoir volumes/fills, inflow summaries, environmental constraints, normalized water values/margins and joint unit-state/count outputs from the local physics cache.

A focused test proves the 463/201 source partition, original column order, row broadcasting and unchanged prefix. Structural review found no blocker. The runner rebuilds the frozen 664 features with the appropriate training-year label cutoff, checks all sampled training labels against the saved canonical cache, and verifies 17 deterministic train/validation prefix probes before fitting. The 2021 grid, 1400 trees, 31 leaves, original 40% sample, seed 4173 and four threads remain fixed against the clean 115 anchor; any accepted weight is locked for 2022.

Run `physics-provided-only-canonical316-2021-001` completed in 1,534.01 seconds with 6.707 GB peak RSS. Recipe `docs/physics_provided_only_canonical316_recipe.json` has SHA256 `9be49e83d26c45a47d5d1cd3e05688efef51fe28cf61e4a8bc4c9b5e983ac1f1`. The coordinator explicitly authorized a temporary third four-thread job after checking memory headroom; the runtime guards require at least 20% free memory and at most 12 GiB own peak RSS. The first unrun recipe is archived, and the only execution revision was a stricter memory floor. The fixed grid selected weight .1, yielding 2021 AUC .9952670135943384 versus .9952596524075921 for the clean 115 anchor (+.00000736). Weights .25, .5 and 1 scored .9952505624393033, .9951542469912296 and .9946339535311148. Full checkpoint replay passed. Independent saved-array/rank/source/input audit passed in `artifacts/provided-only-canonical316-2021-audit-001`. Parent explicitly deferred 2022 continuation behind physical diagnosis because this gain is small; no 2022 or final claim follows.


## Bounded relax-and-fix: no expansion

Run `physics-provided-only-relax-fix36-001` used the fixed 36 pre-2021 cases, a 2-second MILP budget and 1 second for each of two fixed-state LP replays, with one solver thread. At endpoint tolerance 1e-6, fractional states and their same-unit immediate neighbors remained free binary variables. Failed cases would preserve the original local LP exactly.

All 36 cases produced feasible incumbents with zero soft slack and exact fixed-state LP replay. Seventeen reached proven restricted optimality; nineteen retained feasible time-limited incumbents. Restricted optimality does not establish unrestricted MILP optimality. All 36 cases also had the existing three-binary reference, so full-population and matched-coverage results coincide.

| Diagnostic | Continuous LP | Existing three-binary | Relax-and-fix |
| --- | ---: | ---: | ---: |
| Discharge-ratio AUC | .9446522833 | .9463268150 | .9409557667 |
| State AUC | .9415567670 | .9337099712 | .9056622857 |
| State errors at .5 | 8,061 | 7,227 | 7,530 |

The binary state naturally loses ranking resolution; the continuous discharge score also worsens. This does not justify a full cache or supervised extension. Decisions and predictions were sealed before one target read. Full scores, objective/feasibility/failure receipts, hashes and commands are in `artifacts/physics-provided-only-relax-fix36-001/manifest.json`. Runtime was 69.73 seconds. Recipe SHA256: `f01d82eb3d8777ea81f5a5d558e4c19b793a89059e800bdb4aacd76fb35d6e62`. No submission follows.


## User-authorized external mechanism research

The user subsequently authorized examining public solver outputs for patterns that can inform models whose training and inference inputs remain exclusively supplied data. This changes the research boundary, not the runtime input boundary. Any resulting change is labeled externally informed; no public output is used as a candidate feature, fitted teacher, numerical lookup, or target.

`physics-external-mechanism-research-002` compares continuous public No_MIP/All_MIP outputs with the unchanged local LP on the preexisting 36 pre-2021 cases. No target is read. Local versus public No_MIP threshold decisions differ most at Vinje G3 (1,273 of 6,048), Tokke G2 (789), Haukeli (774) and Hogga (734). Lio and Byrte differ only 27 each. This argues against a universally missing environmental constraint: local water allocation and shared physical losses are more plausible explanations for the broad gap. External All_MIP versus No_MIP is a different integrality comparison. These descriptive public comparisons are not validation scores.

The first research loader stopped on an overly strict horizon-end assertion; the last fixed case ends exactly at the exclusive 2021-01-01 boundary. Run 002 changes that assertion to allow equality before any numerical comparison. Both records are preserved.

The public case 1 full JSON is saved after five full-mode iterations, five incremental iterations and SHOPSIM, whereas the CSV solver columns were extracted after full mode. Its final-stage physical fields therefore cannot be equated with CSV predictions. `physics-external-case1-hydraulics-001` checks only physical head/loss semantics against the supplied YAML. Tokke shared main/penstock loss is 23.7568 m at 32 m3/s per unit, while the local unit-alone curve uses 2.2528 m. Vinje's corresponding first-hour values are 6.142640625 m and 2.195015625 m. The known shared-loss expression reproduces exported values to numerical precision. Earlier shared-loss iteration failures remain relevant; this is confirmation, not a claim that the failed implementation has become successful.

Songa has an additional, distinct inlet-network loss. Its two source reservoirs connect through separate quadratic-loss tunnels. The junction pressure h solves `sum(sign(H_i-h)*sqrt(abs(H_i-h)/k_i)) = Q`. At zero total withdrawal, `h=(k2*H1+k1*H2)/(k1+k2)` and unequal reservoir heads permit exchange through the junction. The arithmetic mean is incorrect when resistances differ. Using exported final reference pressures only as a research check recovers all 168 junction pressures within 2.43e-11 m; additional feed loss reaches 9.093 m. The exported aggregate loss differs by at most .0133 m because final reference and physical fields are not fully synchronous. The implementation derives every candidate input from supplied initial volumes, head curves, tunnel losses, turbine/generator curves and terminal water values.

The separate `physics-provided-songa12-001` diagnostic tests original LP, tunnel-PQ-only and tunnel-PQ plus consistent terminal conversion on the same fixed 12 pre-2021 cases. The terminal factor uses the supplied turbine peak-efficiency discharge, not an external fitted factor. Candidate replay and conservation precede one original-label read. No full cache or model fit follows automatically.


## Songa tunnel diagnostic result: no aggregate promotion

`physics-provided-songa12-001` completed in 11.43 seconds. All three arms were feasible with zero soft slack; the final candidate replayed exactly. The consistent tunnel-PQ plus terminal correction improved Songa's own discharge-ratio AUC from .99689055 to .99773310. Applying the PQ change while retaining the old terminal factor instead reduced Songa to .98279548, demonstrating the importance of using consistent economics. However, aggregate AUC fell from .94992690 to .94788673 because downstream Tokke G1 dropped from .95898613 to .94476044; most other unit scores were unchanged. This is a negative aggregate result, with no full-cache expansion or fit. Recipe SHA256: `f50296fd74b9d79f0b2fa7931d28011e952a8f5547a2130165075f021fbd098d`.

## Conditional shared-flow curve diagnostic

A separate exact mechanism emerged from the public physical-state research. When unit j's discharge q varies while the other units' reference flows remain fixed, shared loss is `k_main*(q+Q_other_plant)^2 + k_penstock*(q+Q_other_penstock)^2`. The prior update froze the extra loss at the previous own discharge and therefore omitted the linear cross term `2*(k_main*Q_other_plant+k_penstock*Q_other_penstock)*(q-q_reference)`. At the reference point the two formulas agree, but their slopes differ. The new curve reproduces case1 Tokke exported minimum and maximum production within 2.12e-8 MW using supplied physical curves and public reference inputs only for research verification. The nonconverged Vinje final-stage outputs do not reproduce this identity; no fitted numerical adjustment is made. Evidence is in `artifacts/physics-external-case1-hydraulics-001/conditional_minimum_identity.json`.

The candidate runtime implementation reads only supplied inputs and their local LP cache. `physics-provided-conditional-shared36-001` retains all 36 fixed pre-2021 cases and compares the old stale-extra-loss rule against the conditional rule through five total solves per arm. Iteration1 is the exact original linked-state LP, followed by four undamped reference-flow updates. Initial heads, terminal values, nominal Pmin, constraints and other units remain fixed; the Songa correction is excluded. This isolates the cross term rather than claiming to reproduce SHOP's full head-optimization procedure. All iterations retain objective, feasibility, flow changes, threshold changes, reference-loss inconsistency and exact two-cycle diagnostics. Only the predeclared final iteration is scored after a physical seal. One actual-case smoke proves exact original-vs-hourly solver replay; the analytic cross-term test passes. Structural source review found no blocker. Recipe SHA256: `08d6be527ceef030c9eb8a3d27058b36563dd0c5e8ffa24c9403a3446331e66d`.


The first conditional-flow diagnostic stopped before target access after 59.12 seconds, at the stale-control fourth iteration of case 1727. The sum of reconstructed transition costs differed from the objective's transition variables by 4.85e-5 EUR, exceeding the preserved 1e-5 accounting tolerance. Individual solver row tolerances were 1e-7, so small transition residuals can accumulate when multiplied by costs and summed. The failed artifact and v1 sources remain unchanged. A separate v2 tightens primal/dual feasibility tolerances to 1e-9 for both updated arms; it does not weaken any check. The exact original first iteration remains at the baseline settings. Cases, physical formulas, five iterations, scoring and label scope are unchanged. V2 recipe SHA256: `3a14465e82973b205874659fbb05beb48dada8b10c5ddf5a7860c5bb2d947873`.


The tightened conditional-flow run `physics-provided-conditional-shared36-002` completed in 105.77 seconds. All updated solves had zero soft slack, water residual at most 1.20e-12 Mm3, and exact final checkpoint replay.

| Fixed 36-case metric | Original linked-state LP | Stale extra loss, iteration5 | Conditional shared loss, iteration5 |
| --- | ---: | ---: | ---: |
| Discharge-ratio AUC | .9446522833 | .9460774357 | .9465978682 |
| State AUC | .9415567670 | .9431098635 | .9458177370 |
| Power-ratio AUC | .9437815866 | .9451942964 | .9460665968 |
| State errors at .5 | 8,061 | 7,861 | 7,870 |

The conditional mechanism improves aggregate ranking against both controls. Vinje G3 improves from .88396851 to .89447630 and Haukeli from .87650898 to .90730443, while some other units lose. It is not converged: at iteration5 the maximum reference-loss inconsistency is 14.72 m (stale control18.96 m), 2,263 threshold states change from iteration4, and five cases have exact two-cycles. These limitations do not invalidate it as a supervised proxy feature; they do prevent claiming an exact physical solution. Root authorized preparation of a full proxy cache while a separate cross-generator objective-gradient diagnostic is investigated.

## Full conditional proxy cache: prepared, not launched

Runner `scripts/physics_provided_conditional_cache.py` and recipe `docs/physics_provided_conditional_cache_recipe.json` cover all 3,647 cases with four one-thread workers. Expected wall time is approximately 20–35 minutes from the measured 36-case cost; this is an estimate, not a run result. All 36 cached original predictions are byte-identical to the diagnostic's freshly solved first iteration. A one-case worker smoke completes in 1.14 seconds and reproduces the diagnostic's final flow, power, state, storage and dual arrays exactly. A focused test checks unit-axis mapping, normalized water-margin units and whole-case fallback semantics.

The cache retains final outputs/curves, five-iteration state/flow/power/storage/dual histories, physical residuals and oscillation records. Fifteen fixed own-unit fields expose final state and baseline/last-iteration changes, Q and P ratios, normalized final water margin, upstream/downstream fills, shared Tokke/Vinje state sums and availability. If any updated solve fails, all partial candidate output is discarded; that case retains exact baseline outputs with availability0. Every case remains in the population. No model is fitted and no target is decoded. The source is externally informed, but there is no external runtime dependency. Preparation recipe SHA256: `7a91d5dff2da6cd54faa02e95d1f366ba15f9d924fa78c207c3c450dbeae9a93`. Full execution is intentionally on hold for the coordinator's comparison with the objective-gradient diagnostic.


## Aggregate busbar-loss diagnostic and current execution

The 2019 SINTEF user-meeting presentation describes a busbar formulation that removes conduit loss from individual PQ curves and subtracts a convex piecewise-linear function of aggregate conduit flow. A later formulation report describes a previous-flow method, so exact version/configuration equivalence is uncertain. This is method evidence only: every candidate coefficient is derived from supplied curves and losses, and no public numerical output enters runtime.

`physics-provided-busbar12-001` completed in 13.50 seconds. Aggregate discharge-ratio AUC improved from .94992690 to .95980546; state AUC improved from .93256333 to .94932090 and threshold errors fell from 2,668 to 2,425. A matched gross-curve-without-loss control scored .93799050 discharge AUC. Tokke/Vinje and Haukeli improve, while Lio falls from .93694975 to .89921892. All cases had zero soft slack, maximum water residual 5.12e-13 Mm3, zero observed net fractional-minimum deficit, and exact final replay. The method retains conservative gross-unit ratings, fixed supplied-derived mean efficiencies, eight loss intervals, and capacity-average loss slopes at negative prices. It is an approximate LP, not an exact SHOP reconstruction. Recipe SHA256 `0ae53ba1e85f0d30a087be4dae80decbbb68db570d3bea7861cf2663fc09ce33`; manifest SHA256 `1b9b743f73af1a05d94bfebaed5c880d2859068cd5d26b96c2b6668289b835b0`.

Root authorized the unchanged extension to all 36 pre-2021 cases, reporting the reused 12 and additional 24 separately. Runner `scripts/physics_provided_busbar_diagnostic36.py`, recipe `docs/physics_provided_busbar36_recipe.json` SHA256 `2624f3ad5b7f11b5010829a36ba227c348aaf8edc93f10fe054ad910908eba6c`; run `physics-provided-busbar36-001`, session20046. No third full feature cache is authorized.

The full conditional shared-flow cache was subsequently authorized and launched as `physics-provided-conditional-cache-001`, session50994. At the latest check it had completed 1,275/3,647 cases in 404 seconds with zero fallbacks. Its recipe and source remain unchanged. Structural independently reviewed the diagnostic and is now building a separate bounded cross-flow-gradient cache. Its full wrapper passed source review: accepted-mixture replay is distinct from proposal replay, and no invalid mixed-schedule dual is emitted.


## Busbar unchanged 36-case extension

`physics-provided-busbar36-001` completed in 47.47 seconds. On the additional 24 cases, the unchanged candidate improves discharge-ratio AUC to .95177183 versus original .94260877 and gross-zero-loss control .94004645. State AUC is .95124000 versus original .94586167, with 5,159 versus 5,393 errors. All 36 cases score .95424476 discharge AUC, .95063486 state AUC and 7,584 state errors, against original .94465228/.94155677/8,061. The reused12 results match the first diagnostic exactly. No third full cache was launched. The bounded cross-flow-gradient arm remains stronger on all36 (.95610162) but busbar is slightly stronger on the additional24 (.95177183 versus .95037115); this supports diversity, not a claim of chronological learned improvement.

A separate controlled12 beginning-head run `physics-provided-beginning-head12-001` is now active (session63194), recipe SHA256 `796e1c0915cb50c44a7c99cc81123dc3e46e130157523c054d0605a9c52bc99c`. It retains initial terminal economics and applies conditional shared-flow curves in both arms. Both use the same Hogga/Tokke curve constructor; only the candidate uses heads from supplied initial storage at hour0 and previous local end-storage thereafter. Exact single-reservoir Bandak/Vinjevatn pools support this mapping. Vinje keeps initial head, all other physical inputs remain fixed, and no storage-objective gradient is added. Boundary timing and pool/tailwater mapping tests passed. This is distinct from the earlier failed midpoint-head/unit-alone experiment.


## Beginning-hour head result and busbar cache preparation

`physics-provided-beginning-head12-001` completed in 86.31 seconds. Beginning-hour heads slightly reduced discharge-ratio AUC to .95815865 from the matched fixed-head conditional control .95853567. State AUC improved slightly (.94966283 versus .94929389), and state errors fell from 2,438 to 2,426. The fixed-head arm exactly reproduces the earlier conditional12 score. This mixed result does not justify a full head-update cache; the timing implementation is retained and independently reviewed. Both arms keep original terminal values, nominal Pmin and all environmental constraints. No numerical external input is involved.

Root authorized a two-worker busbar cache to begin after the conditional cache ends. `scripts/physics_provided_busbar_cache.py` with `sintef_project/provided_busbar_features.py` emits 15 genuine fields: state and baseline-state difference, two Q ratios, net/gross P ratios, allocated conduit loss, baseline flow change, normalized water margin, upstream/downstream fills, Tokke/Vinje state sums and availability. No iteration-history fields are fabricated. The full population and input hashes are pinned, with exact original whole-case fallback on candidate solve/replay failure. Recipe `docs/physics_provided_busbar_cache_recipe.json` SHA256 `1f202848bd62297d4c4a305e8daf81c63172779cd560c089655dfcf96fb44de0`. A targeted formula test passes; smoke `physics-provided-busbar-cache-smoke-001` proves case1 parity with the36-case diagnostic (0.91 seconds) and exact original fallback after a deliberate case134 solver failure. Full cache has not yet launched. Conditional cache last checked at2,100/3,647, zero fallbacks,717 seconds.


## Bounded all14 binary busbar diagnostic: negative result

Root authorized `physics-provided-busbar-integer12-001`, which changes only commitment integrality in the corrected busbar LP. It retains the origin-inclusive unit PQ hull, fixed group efficiencies, conservative gross rating and original terminal economics. Each of the fixed12 cases receives a five-second one-thread MIP limit, followed by two one-second fixed-incumbent LP solves. The fixed binary schedule and conditional LP duals replay exactly; no claim of global MIP duals or exact nonlinear binary physics is made. Whole-case continuous-busbar fallback is declared for any missing or invalid incumbent. Source review found no blocker. Recipe SHA256 `1e13d5993a288e4e5c169b40eb731997ca915a436e935532120cfd59b328de62`.

The run completed in63.01 seconds. All12 had feasible incumbents and zero soft slack; five proved surrogate optimality, while seven reached the limit, with maximum relative gap .00211053. No case needed fallback. Aggregate discharge-ratio AUC fell to .94657772 from continuous busbar .95980546, and state errors increased to2,756 from2,425. A descriptive decomposition on the same original labels shows Lio errors212→375 and Haukeli228→336; Tokke unit errors748→809 but plant-count absolute error654→641, indicating some allocation ambiguity alongside broader scheduling error. Vinje count error677→674 is nearly unchanged. This negative result does not authorize a binary full cache. `error_decomposition.json` is a post-diagnostic explanation, not a tuning or selection step.

The continuous busbar36 saved-array/rank audit independently passed in `provided-busbar36-saved-audit-001`, manifest SHA256 `0c4a513451fd1943b9c51afca86aa6d0ca9bdaf201673e7d8cfed970def953e1`. Its two-worker full-cache source and honest15 feature schema also passed review. The separate auditor `scripts/audit_provided_busbar_cache.py --kind busbar` will validate the different no-history format without modifying the currently frozen conditional/gradient validator.


## Conditional cache complete; continuous busbar cache launched

`physics-provided-conditional-cache-001` completed all3,647 cases in1,220.04 seconds with no fallbacks, no target reads, and per-case exact final replays. Feature shape is[3647,168,14,15]. Manifest SHA256 `daca717e605f5c20ee82643ac64d4d914890fb7795e5022d17f04f215de5f681`; feature-array SHA256 `64348e51422a4cc4c47b5267dc2f1c90a879a0b120be967547065db3f2fe6d0d`. The process exited0. Structural is performing the independent full formula/cache audit before metric's frozen chronological learner proceeds; this cache completion is not a validation or leaderboard improvement.

As explicitly authorized, the separate continuous busbar cache started only after conditional exited: session17749, run `physics-provided-busbar-cache-001`, log `artifacts/physics-provided-busbar-cache-001.log`, command `.venv-ml/bin/python scripts/physics_provided_busbar_cache.py --recipe docs/physics_provided_busbar_cache_recipe.json --run-id physics-provided-busbar-cache-001`. It uses two one-thread workers and the unchanged independently reviewed recipe SHA256 `1f202848bd62297d4c4a305e8daf81c63172779cd560c089655dfcf96fb44de0`. It does not include the rejected binary or beginning-head variants. Structural's gradient cache remains a distinct branch.


## Four-plant turbine-peak terminal conversion

`physics-provided-peak-terminal12-001` completed in12.30 seconds. All dispatch PQ curves and other inputs remain byte-identical; only Lio/Byrte/Haukeli/Vesle Kjela conversion factors change from max P/Q to evaluation at supplied turbine peak-efficiency discharge, followed by cumulative downstream terminal-value reconstruction using provided horizon-end local water values. This is an externally informed convention, with no external numerical coefficient or research-module import. The peak discharge is interpolated at gross head, then power evaluated at net head using supplied losses; it is not claimed to be a globally optimal P/Q point. Physical-units and downstream-propagation tests pass; independent source review found no blocker. Recipe SHA256 `56c647513ba1ea967bff2fdc2b8a94503c390795d7a53e19926c9e2a433a53a5`.

The12-case qratio AUC improves .94992690→.95153212, state AUC .93256333→.93412973, errors2,668→2,610. Haukeli contributes most of the gain (.82306889→.84611627); Lio slightly worsens. The unchanged36-case extension `physics-provided-peak-terminal36-001` completes in30.02 seconds. Its additional24 shows a smaller qratio gain .94260877→.94281710 and state AUC .94586167→.94618479, with five more threshold errors (5,393→5,398). All36 qratio .94465228→.94534169 and errors8,061→8,008. Reused12 results match exactly. Recipe SHA256 `b7283e3edbd3c948a529af6866de0869d8694eaf915cc338f5e198af8275c034`.

Before any extra full cache, root authorized `physics-provided-peak-interactions12-001` (session76219). It compares unchanged busbar and bounded-gradient methods against each method with only these terminal factors changed. Original controls must reproduce their sealed36-case outputs bitwise; gradient retains the exact five iterations, fixed .5/.25/.125/0 physical-merit steps and proposal/mixture replay. Current full caches and learner sources remain frozen. Interaction recipe SHA256 `9549fb17a7a722612ce185a919130f0877d8a819aa461faeca9ef9c2a0445931`.


## Terminal conversion interactions: modest gains, no new cache

`physics-provided-peak-interactions12-001` completed in69.50 seconds. Both original busbar and original bounded-gradient controls reproduced their earlier sealed36-case schedules bitwise. Adding peak terminal factors to busbar was nearly neutral: qratio .95980546→.95985911, state AUC .94932090→.94896325, errors2,425→2,424. Adding them to the bounded-gradient method improved qratio .96799410→.96812146, state AUC .95913463→.95946658 and errors2,297→2,239. These are reused12 exploratory diagnostics, not new chronological model validation. Independent source review found no blocker. No extra full cache follows; the completed conditional, running gradient/busbar caches and frozen chronological learners remain the priority.


## Independent cache verification and interaction extension

The complete conditional cache independently passed all3,647 case/formula/aggregate-byte checks in22.78 seconds: `provided-conditional-full-cache-audit-001/manifest.json`, SHA256 `374720f3ab21a80117b8057d5471946082da90ee5817fb6e675b467823e45aba`. Dates, units, original supplied-source closure and water balances were verified; zero fallback. The result is available to metric's frozen learner.

Root closed the near-neutral busbar+peak interaction and authorized only an unchanged gradient+peak extension to36. `physics-provided-peak-gradient36-001` runs in session58437 with recipe SHA256 `e0fe323612dcdecb90c135864ddc045bc2d619d8e00311f6a9d851bc971cfb88`. Original gradient controls must match the sealed36 schedules bitwise; reused12 and additional24 are separately reported. No fourth full cache is authorized. Busbar cache session17749 remained healthy at1,700/3,647 with zero fallbacks after599 seconds.


## Gradient+peak unchanged36: mixed additional24 result

`physics-provided-peak-gradient36-001` completed in168.50 seconds. Additional24 qratio improves .95037115→.95069003 and state AUC .95323503→.95347129, but state errors worsen5,323→5,333. All36 qratio .95610162→.95636510 and errors7,620→7,572; the error reduction is driven by the reused12 subset. Original controls and candidate proposal/mixture replays remain exact. This does not meet the coordinator's stated requirement for additional24 support in both ranking and errors, so no fourth full cache or combined variant is prepared. The result is retained as exploratory evidence; current conditional/gradient/busbar caches and first learners stay unchanged.


Coordinator clarification: gradient+peak is **deferred with promising AUC support**, not definitively rejected. The competition's primary ranking metric improves on both additional24 and all36. The threshold-error requirement was a prospective internal heuristic and was not met. No fourth cache is started because validating the current three learned candidates has higher priority. The modest coupled terminal correction can be reconsidered after those results; all sources and artifacts remain preserved.


## Busbar cache complete and verified

`physics-provided-busbar-cache-001` finished all3,647 cases in1,203.66 seconds with zero fallbacks and exact per-case replay; process17749 exited0. Manifest SHA256 `9c266f71bad082e8a69986f4eac7a05e68be260266444f7c352a596afbc68fde`; features SHA256 `12a68b42d601c64752aa12c40a05158edded8dd134871291fac7b57c32e63298`. Independent full audit passed every formula, hash, mapping, aggregate byte, conduit-loss allocation and original water-balance check: `provided-busbar-full-cache-audit-001/manifest.json`, SHA256 `1f8a253ac2cb092792cd39e757c9d1f8239ab97c42f7795f0e82174d8911c005`. Metric owns its frozen2021 learner. No process remains active under physics ownership at this checkpoint.

## Case1561 research and directional Haukeli split

The research-only artifact `physics-external-case1561-research-001` compares already authorized public full-mode Haukeli outputs and later-stage physical JSON against local provided-input schedules. No public number becomes a candidate feature, teacher, label or coefficient. Full CSV Haukeli runs168h at707.42/708.49MWh (NoMIP/AllMIP), versus177.7MWh continuous busbar and104.1MWh binary busbar. The physical minimum remains about.3MW and terminal cumulative water value is close to local, so no large minimum-output or terminal-conversion discrepancy explains it. The original research report/source are retained; `research_corrected.json` corrects only a metadata key that mislabeled all-pool ordering as Haukeli membership.

A full-stage follow-up `full_stage_comparison.json` is essential: full-mode Tokke production stays about96.2–97.7MW, near capacity. The later JSON's24–32m3/s Tokke variation reflects redispatch and is not evidence about full-mode allocation. The CSV has no solver_discharge columns. Any cross-stage interpretation remains tentative.

Root identified a real topology simplification: Vatjern(.3Mm3) and Langeidvatn(32.1Mm3) were pooled, while the supplied network only permits downhill Langeid-to-Vatjern flow. However case1561 Vatjern starts.15Mm3 and receives only.014688Mm3 natural inflow, so its capacity does not by itself force the missing Haukeli operation. The provided river cost is a convex piecewise-constant marginal curve, per the [official river documentation](https://docs.shop.sintef.energy/objects/river/river.html#flow-cost-curve); its last cost is continued without inventing a hard maximum.

`physics-provided-haukeli-split12-001` completed in18.53 seconds after waiting for a training slot to free. It compares the exact original9-pool control with10-pool directional arms at zero added river cost and the supplied river cost. All14 constant-head PQ curves/limits and terminal conversion remain unchanged. Individual storage/inflow totals reconcile to original float32 aggregates, with residuals recorded; each arm replays exactly and has zero slack. Supplied-cost split qratio AUC .94961384 is below original .94992690, while errors fall2,668→2,644. Zero-cost split is worse (.94877544,2,753 errors). Haukeli's own qratio AUC declines .82306889→.81319032 with the supplied cost. Recipe SHA256 `4c7e28f3009c6246418918a79ed70a9dced6fc09a0a1d9e5eaece071a5c80d5a`. Root closed expansion: no36 or full cache.

## Next authorized preparation

Root requested a new bounded12 endogenous-head/shared-flow comparison, with no frozen-module changes. It must include Tokke's Vinjevatn upstream benefit, Vinje's Vinjevatn tailwater penalty, Hogga's Bandak benefit and Tokke's Bandak tailwater penalty, respecting outlet clamps. Vaamarvatn upstream head remains fixed explicitly. End-storage[t] affects generation at[t+1] only; final storage has zero generation gradient. The derivative must match finite-difference physical revenue at fixed flows. Both matched arms use the same shared-flow gradient and physical-merit line search; only one adds the storage-objective derivative. Prepare recipe/source review before solving. Current two fits use8threads; no new large cache or heavy fit is authorized.

## Endogenous-head comparison running

The two finite-difference tests in `tests/test_provided_endogenous_head_shared.py` pass locally and in root's independent rerun. They cover both reservoir dependencies, outlet clamping, positive and negative prices, the hour shift, final-hour zero derivative, and nonmutation of head arrays. Root reviewed the new module and runner without finding a blocker, then authorized the fixed 12-case run. Recipe SHA256 is `31de7512fd2056c345bed954292c861f64d0aa438089322afb53737802b98343`; exact source snapshots are in `artifacts/physics-provided-endogenous-head12-snapshot-001`. Run `physics-provided-endogenous-head12-001` is active under session39101 with one solver thread. The recipe retains unchanged terminal values, nominal Pmin and the original origin-inclusive four-segment hull; it makes no exact nonlinear feasibility or convergence claim. No 36-case extension or full cache follows automatically.

The separate pooled CatBoost144 v2 common helper, bundle builder, worker and generated private notebook were source-reviewed while the diagnostic ran. No blocking issue was found in the sample/label ordering, chronological cutoff, exact numeric columns, one-hot identities, selection lock, prefix handling or fresh-process replay. The embedded worker/helper/protocol bytes match the frozen sources. Stage2021 protocol SHA256 is `58678bb12af9c05014998ddabf93678dc54f11267aeacf385f55f73f4f637f33`. This was a source review, not a claim that the hosted model has trained or passed its return audit.


## Endogenous-head12 completed; no extension

`physics-provided-endogenous-head12-001` completed in 134.22 seconds with exact proposal/mixture replays, zero soft slack and maximum network residual below 2e-13. Manifest SHA256 `8525afd8019ec1c335c1459610dc098b3fc4acff297151b2c41a781ed0476441`. Fixed-head cross-gradient qratio/state AUC are .96799410/.95913463 with 2,297 threshold errors; dynamic heads alone .96637625/.95974459 with 2,151; dynamic heads plus the storage derivative .96721261/.95886214 with 2,296. The derivative improves its matched dynamic qratio control but both lose to the fixed-head ranking reference. Rank-formula reproduction, unit/case errors, physical checks and descriptive leave-one-out scores are in `physics-provided-endogenous-head12-analysis-001`. Both formulations retain physical hull/state-link discrepancies; maximum reported state-link violation .04599 is not claimed feasible nonlinear dispatch.

The dynamic-only error reduction is concentrated in case1561: Haukeli saves 168 errors while Hogga adds 23. Across the other 11 cases the net reduction is only one error; dynamic qratio remains below fixed-head after every one-case exclusion. Haukeli's relaxed state is .511775 throughout with dynamic heads alone versus .409420 with the storage derivative. These are threshold crossings, not recovery of full physical dispatch: total surrogate output is only 128.04 versus 105.14MWh. Dynamic-only accepts steps .5 then .25 and stops; the derivative accepts .5 then three zero steps. Root closed both the derivative branch and any 36-case extension, retaining all artifacts. No full cache or learner uses these new outputs.

Root launched the separate supplied-economics startup142 2021 learner as session68849. Physics owns `scripts/audit_provided_startup142.py`: streamed independent reconstruction of all 27 appended fields from supplied YAML, exact115 prefix and sampled original training-label checks, chronological checks, then result-mode checkpoint/rank/projection/lock/CSV verification. Preflight runs as session7884 at `physics-provided-startup142-preflight-001`; no new model fit is owned here. Structural owns further descriptive Haukeli mechanism research. GPU Cat144 source review passed as recorded above.


Startup142 preflight completed successfully in 3.13 seconds: `physics-provided-startup142-preflight-001/manifest.json`, SHA256 `69fb5047c3fb951d837aa631299ac3b32cb3dd742babccd9dad27cc0b50b53c4`. All 2,056,586 training rows and 858,480 prediction rows preserve 115 prefix columns and reproduce the 27 appended formulas exactly. The original 2,186 training cases, 40% sample RNG sequence and six-day purge were checked; no 2021 target values were decoded. Metric reviewed the auditor without a blocking defect. `audit_provided_startup142_v2.py` preserves the original version and adds a strict requirement that reported score keys equal the complete frozen grid (or identity plus locked weight on 2022). It will audit the completed result.

The four local pooled Cat144 verification/selection/audit files were also reviewed without a blocker. Stage protocols, supplied-data parent/bundle hashes, scoped original targets, complete model replay, one-hot JSON probes, fixed grid/2022 lock, matched raw-LGB comparator, exact unavailable-row fallback and independent rank/PAV reconstruction are consistent. These reviews do not assert a hosted training or validation outcome.


## Startup142 first chronological result and Vinje tunnel preparation

`provided-startup142-2021-001` passed at the frozen grid's 25% blend: AUC .9953329489195831 versus clean115 .9952596524075921, a gain of .0000732965. Independent result audit `physics-provided-startup142-2021-audit-001` passed in 4.19 seconds; receipt SHA256 `aa9865c210b1c9a29bbab3490aae945cd6dc5cd1855d69b1a2d6f5ca48e6b3a7` binds fit manifest `dd3c8842cfa2e63c9515c1fb5a3369016cca2f8ff0e6446fca6e8901b3213a07`. All appended formulas, original training/fold labels, full score grid, rank AUC, 512 checkpoint rows and exact selected bytes were checked. Root launched unchanged 2022 weight .25 in session43310; physics owns its forthcoming audit. No duplicate fit.

Structural delegated the module/tests for the new twelve-pool Vinje tunnel diagnostic. `provided_vinje_tunnel_split.py` keeps the original Vinje index as Vaamarvatn, adds Totak/Venemo/Hyljelihyl, restores named turbine/bypass/spill routes and the supplied Hylje marginal river costs, and disables nonexistent Hylje spill. Four zero-cost directional tunnel arcs are unbounded in the matched free-transfer control; the candidate only imposes frozen initial-head capacities derived from supplied opening heights and quadratic loss factors. These are approximate transfer envelopes, not exact hydraulic equalities or defensible weekly head bounds. Existing Kjela pooled bypass/spill routing remains explicitly approximate. Original 14 PQ curves and terminal conversion are unchanged; each split reservoir gets the old global Vinje terminal value. Four focused tests pass in 1.46 seconds, including captured-LP proof that only the four tunnel upper bounds differ. No actual solve or diagnostic label read occurred in this preparation. Structural owns driver/protocol and source review.


## Startup142 locked2022 passed; Vinje pressure diagnostic audited

`provided-startup142-2022-001` passed the unchanged 25% blend gate: AUC .9940802862111509 versus clean115 .9940509264719491 (+.0000293597). Independent result audit `physics-provided-startup142-2022-audit-001` passed in 4.56 seconds; SHA256 `8df321ce721a645602d30f4fe69ddd51babc7e9bf16b7368da784a0369a979dc` binds fit manifest `2e8f8efa3baa7811c48cb8aa6eca956ad5b834dd6385d30e64fd39f9ef5fa401`. The exact recipe/source, complete27-field formulas, original train/fold labels, sample/purge, locked weight, 512 checkpoint rows, rank scores and selected bytes all passed. Root owns the authorized final-training schedule; physics will audit the final artifact. No leaderboard score is inferred.

`provided-vinje-tunnel-split12-001` completed all12 cases/replays with zero slack in18.14 seconds. The matched free-transfer split has qratio AUC .94988629 versus original .94992690; the fixed initial-pressure arm declines to .94389698 while threshold errors fall2,668→2,565. Case1561 is a substantive physical result: Haukeli production rises from zero under the free split to577.326MWh under pressure bounds, Tokke shadow rises23530.7→27029.812 EUR/Mm3, and the Totak–Vaamar cap binds all168 hours (mean capacity shadow3683.39). These frozen capacities are too crude as weekly envelopes, so no full cache is promoted from this arm. Unlike the earlier relaxed-state threshold crossing, this changes actual generation, but aggregate ranking still decides model promotion.

Independent audit `physics-provided-vinje-split12-audit-001` passed in1.54 seconds, SHA256 `8e9589bfae1474d4e554d47312047585529e19d2797b02c200f8014b8bcaf3e5`. All saved predictions, rank/unit/case scores and original12 labels match; each named reservoir/hour balance and soft storage/bypass/Vest constraint was reconstructed independently, with pressure caps, zero Hylje spill and restricted-objective ordering verified. No additional solve or fit. Candidate module/test/source snapshots remain frozen. Structural owns method follow-up and KKT interpretation.


## Startup142 final artifact verified

Final fit completed701.27s, independent final audit passed6.43s, and complete standalone inference replay passed47.65s. All1,705,200 scalar outputs and actualCSV bytes reproduce exactly. CSV `artifacts/provided-startup142-final-001/submission.csv`, SHA256 `498c2f7b9acd3a3974b855b816f94da4ba52ea6ce1de86153cd294cafb5b1953`. Full provided-only recipe, fold scores, checkpoint/hash and no-fit command are in `docs/provided_startup142_result.md`. No submission or leaderboard score was made by physics. All owned training/inference processes have exited normally.

Coordinator was reported errored with a model-capacity message by metric. Children continue only already authorized work and preserve handoff. Structural owns the newly approved paired12 signed-tunnel equality diagnostic; physics supplied its separate module/tests and reviewed the driver, with all old frozen modules intact. Metric is preparing a separately declared cheap equal-raw startup+busbar blend; no equivalent prior experiment was known. No new heavy fit or UI action is started here.
