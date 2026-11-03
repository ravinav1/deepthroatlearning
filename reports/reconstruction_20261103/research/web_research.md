# Primary-source research: mechanistic improvements for commitment prediction

Research date: 2026-11-02. Only public SINTEF/SHOP documentation, SINTEF publications and official SciPy documentation are used for technical claims. No other team's private code or hidden competition labels were accessed. The objective is to find useful mechanisms, not to claim an unmeasured path to 0.999 leaderboard AUC.

## Strongest actionable finding

SINTEF's June 26, 2026 InterOpt article describes the proposed workflow as a fast relaxed optimization followed by ML prediction of accurate unit commitments. This directly supports an LP-feature-plus-classifier experiment rather than treating raw schedules as unrelated labels. It does not establish the solver settings used for this competition. [SINTEF InterOpt article](https://www.sintef.no/en/latest-news/2026/how-ai-could-squeeze-more-value-from-norwegian-hydropower/)

**Our proposed experiment:** solve a bounded approximation of the supplied watercourse for each case, then give the supervised model continuous discharge, storage and scarcity signals. These are computed from legal inputs independently of labels. The classifier can learn where the approximation differs from the original optimizer. The first implementation is `sintef_project/lp_features.py`; its approximation and benchmark are recorded in `docs/lp_features.md`.

## Commitment settings are not recoverable from static topology alone

SHOP's documented default relaxes generator commitment variables. In relaxed hours, the full-to-incremental transition uses a default cut-off at half the generator's minimum discharge. Integer commitment can instead be enabled with plant MIP flags. Initial generator state has a separate input attribute, with an unset state available. The supplied YAML contains no MIP flags or initial-state settings, but unseen case-generation commands could override defaults. Therefore LP discharge divided by physical minimum discharge is a useful feature; it is not established ground truth. [SHOP generator documentation](https://docs.shop.sintef.energy/objects/generator/generator.html)

The global `universal_mip` command can override individual plant flags; without that command, the flags govern usage. Absence of a flag in the static YAML cannot prove absence of a later override. [Universal MIP command](https://docs.shop.sintef.energy/commands/set_universal_mip.html)

In non-MIP hours, start/stop costs require the linear-startup option; without it, SHOP can omit those costs. Thus our earlier binary switching dynamic program is one scheduling prior, not a verified reconstruction. A no-startup LP and a linear-switching-cost LP would be defensible separate ablations. [Linear startup command](https://docs.shop.sintef.energy/commands/set_linear_startup.html)

## Generation curves, head and marginal water costs

SHOP generates head-dependent production/discharge curves, makes their segment slopes non-increasing, and in relaxed full iterations connects the origin to the best-efficiency point. Physical discharge limits come from turbine curves as well as power constraints. Negative prices and forced water movement can make incremental-segment loading physically incorrect. This motivates using actual unit-specific curves and a documented linear conversion at negative-price hours in the inexpensive LP. [SHOP PQ curves](https://docs.shop.sintef.energy/examples/pq_curves/pq_curves.html)

The plant's production factor for energy-value conversion defaults to best-point operation at initial head. Main-tunnel and penstock head losses are quadratic in flow; units may share a penstock. Our first LP uses initial reservoir levels and unit-alone losses, so shared-unit effects and changing head remain approximation errors. A second iteration using the first LP's storage and plant flows is a concrete follow-up if the first features improve validation. [SHOP plant documentation](https://docs.shop.sintef.energy/objects/plant/plant.html)

The provided EUR/MWh water values are consistent with local `energy_value_input`. Global terminal value in EUR/Mm3 is constructed from local energy value times a local conversion factor plus the downstream global value. Directly subtracting upstream and downstream EUR/MWh inputs is incorrect. For a network LP, converting to global terminal values prevents both omitting downstream production opportunity and double-charging the same water. The supplied generation script is absent, so the attribute mapping remains an explicit inference consistent with the dataset document. [SHOP water-value descriptions](https://docs.shop.sintef.energy/examples/water_values/water_values.html)

SHOP exposes water scarcity through the duals of reservoir balance equations. This is the mechanism missing from a purely terminal-price threshold when storage, inflow or downstream constraints bind. The model can use hourly price minus dual-derived local water opportunity cost, alongside normalized discharge. [Individual water-value example](https://docs.shop.sintef.energy/examples/water_values/individual_water_values.html)

SciPy's HiGHS marginal values describe derivatives of its minimization objective with respect to constraint right-hand sides. For our balance orientation, the positive economic water value is the negative equality marginal, multiplied back by the objective's numerical scale. This sign is independently checked by perturbing initial storage in a small test. [SciPy HiGHS documentation](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html)

## Hydraulic routes matter more than generic storage totals

SHOP rivers can connect multiple upstream objects but have one downstream outlet; natural inflow enters separately. Minimum flow, maximum flow and flow costs are independent constraints. Free rivers need not obey reservoir-elevation restrictions unless the relevant physical model is enabled. Our LP keeps water balances exact and records approximations to flow curves. [SHOP river documentation](https://docs.shop.sintef.energy/objects/river/river.html)

**Direct findings from supplied YAML:** Botnedalsvatn bypass/spill reaches the Vest river and Bandak, bypassing both Byrte and Lio. Totak bypass/spill can reach Bandak without using Vinje or Tokke. Kjelavatn bypass reaches Hyljelihyl rather than following the Vesle Kjela turbine route through Foersvatn. The Vest minimum flow must include its own natural inflow, Lio discharge and designated bypass/spill contributions. Pool aggregation must preserve these distinct outlets.

Tunnel flow depends on opening elevations and pressure loss proportional to squared flow. Combining connected reservoirs into a single freely accessible pool loses these limits and may overstate water available to a plant. Keeping 17 reservoirs and linearizing tunnel flow around the initial head, then around a first solution, is a higher-fidelity follow-up. [SHOP tunnel documentation](https://docs.shop.sintef.energy/objects/tunnel/tunnel.html)

## Why a static price threshold can fail

SHOP's best-profit methods consider marginal and average production costs, feasible unit combinations, reservoir levels and water values, with switching costs handled separately. A static threshold at local water value misses efficiency variation and combinations of units. The classifier should receive plant-total LP generation in addition to unit-specific output because similar units can have multiple almost-equivalent allocations. [SHOP best-profit tutorial](https://docs.shop.sintef.energy/examples/best_profit/best_profit.html)

**Inference for this dataset:** pairwise features for Tokke's shared penstocks and a learned allocation of plant-total generation could help resolve the systematic difference between G1/G3 and G2/G4. This is a hypothesis, not an observed validation improvement. Do not force probabilities of similar units to be identical or assume the LP solver's tie-breaking matches SHOP.

## Experiment order and acceptance criteria

1. Benchmark the nine-pool LP on 16 spread-out training-case inputs, preserving exact topology routes and checking water conservation. No label scoring is needed for this numerical check.
2. Cache all cases using one solver thread. Export dimensionless continuous features, including plant totals and soft-constraint slack. Never replace model probabilities with thresholded LP states.
3. Compare the existing model with and without LP features at identical training rows, hyperparameters and seed on purged 2021/2022 folds. Require measured improvement; retain per-unit/month diagnostics.
4. If beneficial, inspect error versus LP slack, negative-price hours and nearly tied unit allocations. Then test one specific fidelity improvement: shared-flow losses, a head update, separate reservoirs/tunnels, or explicit relaxed switching costs.
5. Preserve probabilistic ranking and the exact submission schema. Continue distinguishing local validation, public leaderboard scores and unknown private leaderboard performance.

## Source limitations and leads

The live SHOP documentation is mainly version 18.x, whereas the local dataset document says SHOP 17.17.0. Defaults and algorithms above are evidence for modeling choices; version-specific differences remain possible.

The public [September 2022 SINTEF presentation](https://www.sintef.no/contentassets/106baff79dc44ca5897fea4c51b1cfa7/p3_6-kong_sintef.pdf) discusses ML integration and command-setting selection. Its first slide is explicitly dated 12 September 2022 despite newer search timestamps; it must not be described as a 2026 dataset-method paper. The [May 2026 lecture record](https://www.sintef.no/en/publications/publication/01a0f384b24e-086bafd9-e06b-4f6f-9de9-14a043a483e7/) confirms a presentation on neural-network commitment prediction, but provides no verified architecture or experimental results in the accessible record. Two 2026 paper titles appear on SINTEF/NTNU author pages, but their full methods were not recovered; no results are inferred from their titles.

The supplied 2025 presentation URL redirected to an inaccessible host in the web tool. An older SINTEF paper PDF returned HTTP502. Neither failure is treated as evidence about model performance.


## Directed-reservoir follow-up, 2026-11-02

The river documentation confirms that `main_river` chooses the downstream route for energy-value conversion; it does not impose a release. An otherwise unconstrained river permits optimized downstream flow. The documented `flow_cost_curve` represents piecewise-constant marginal flow costs, with currency/hour/(m³/s) units. Consequently, Langeidvatn's `main_river=1` cannot by itself explain the dry all-on Haukeli cases. Adding its omitted positive river cost would discourage release, not directly cure those false negatives. This last implication is an inference. [SHOP river documentation](https://docs.shop.sintef.energy/objects/river/river.html).

The local topology also contains directed Staavatn→Kjelavatn and Langesae→Foersvatn rivers, and tiny Hyljelihyl→Venemo. These are additional potential pooling errors, but v2 deliberately changes only Haukeli. Tunnel-connected reservoirs require head/flow limits rather than indiscriminately imposing one-way flow. Their correction remains a separate, unimplemented hypothesis.

## Follow-up source access — 2026-11-02

The NTNU author page now exposes the neural-architecture chapter's DOI, [10.1109/EEM68581.2026.11589730](https://doi.org/10.1109/EEM68581.2026.11589730), via [Zhirong Yang's publication list](https://www.ntnu.no/ansatte/zhirong.yang). IEEE full text could not be retrieved by the web tool; no architecture or results are inferred from the title. The current [SINTEF meeting site](https://meet.sintef.energy/) preserves the May 2026 lecture programme but its presentation files redirect to authentication. No access boundary was bypassed and no recommendations are attributed to unread slides. The existing physically motivated LP-context experiments are supported by our measured residual analysis, independently of those inaccessible sources.
