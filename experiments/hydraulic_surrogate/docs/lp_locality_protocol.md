# Targeted LP locality ablation — proposal only

Prepared2026-11-02 after the fixed LP-candidate comparison. No fitting, newfeatures or frozen-source changes are authorized by this document. The exact ordered lists are in `docs/lp_locality_columns.json`; all names were checked against the completed normalized-LP v1 baseline manifest.

Hypothesis: Haukeli/Lio/Hogga may rely too heavily on calendar or distant raw reservoir proxies when transferring into the2022 price regime. Keep physically informative LP outputs, local reservoir inputs, known price context and local startup-cost summaries; remove direct calendar indicators and unrelated raw reservoir columns. This is unproven. Earlier main-repository removal of detailed system-wide features was harmful, so the proposal targets threeunits and retains LP-mediated cascade information rather than assuming all system context is dispensable.

## Fixed scope

- Baseline:`lgb-normalized-lp-001`, frozen LPv1; noLPv2 or new LP-temporalcontext features.
- Retain `horizon_hour` as optimization position. Omit hour-of-day,sine/cosineclock,weekday,month,day-of-year and annual sine/cosine. Omit `horizon_day` as a redundant encoding of horizonposition.
- Price summaries of the supplied week remain legitimate known optimizer inputs, including daily price aggregates; they are not direct calendar fields. Retain `monetary_scale_inverse` because fixed startup costs depend on absolute economic scale.
- Haukeli local pool:Vatjern andLangeidvatn. Lio local pool:Byrtevatn. Hogga local pool:Bandak.
- Retain own-unit LP q/Qmin,powerfraction and normalized dual-economicmargin, ownpool power/storage/slacks, immediately incoming turbine discharge/power summaries, and receivingpool LP storage/slacks. No unrelated raw reservoir initialvalues are retained. These few connected LP outputs preserve immediate cascade constraints.
- Lio andHogga retain r_Vest natural inflow and the relevant mandatoryflows; these are direct water-balance/release inputs. All other nonlocal raw inflows are omitted.
- Startup inputs are the existing local terminal-water-value DP margin atminimum/maximum operating power. Do not add the new LP-shadow switching method in this ablation; that is a separate experiment.

## Exact shared columns

```text
horizon_hour
price_week_zscore
price_week_rank
context_rank_remaining
norm_price
norm_price_offset_-24
norm_price_offset_-12
norm_price_offset_-6
norm_price_offset_-3
norm_price_offset_-1
norm_price_offset_+1
norm_price_offset_+3
norm_price_offset_+6
norm_price_offset_+12
norm_price_offset_+24
norm_price_week_mean
norm_price_week_std
norm_price_week_min
norm_price_week_max
norm_price_week_q10
norm_price_week_q25
norm_price_week_q50
norm_price_week_q75
norm_price_week_q90
norm_price_minus_week_mean
norm_price_day_mean
norm_price_day_min
norm_price_day_max
norm_price_day_std
norm_price_remaining_mean
norm_price_remaining_max
norm_price_previous_mean_6h
norm_price_minus_previous_mean_6h
norm_price_next_mean_6h
norm_price_minus_next_mean_6h
norm_price_previous_mean_12h
norm_price_minus_previous_mean_12h
norm_price_next_mean_12h
norm_price_minus_next_mean_12h
norm_price_previous_mean_24h
norm_price_minus_previous_mean_24h
norm_price_next_mean_24h
norm_price_minus_next_mean_24h
monetary_scale_inverse
```

Each unit list below adds to the shared list. The machine-readable JSON also contains the complete ordered union perunit in original baselinefeature order.

## Haukeli_G1: 81 total columns

```text
volume_Langeidvatn
fill_Langeidvatn
price_over_water_Langeidvatn
headroom_after_inflow_Langeidvatn
volume_Vatjern
fill_Vatjern
price_over_water_Vatjern
headroom_after_inflow_Vatjern
inflow_Langeidvatn
inflow_week_mean_Langeidvatn
inflow_elapsed_mm3_Langeidvatn
inflow_remaining_mm3_Langeidvatn
inflow_Vatjern
inflow_week_mean_Vatjern
inflow_elapsed_mm3_Vatjern
inflow_remaining_mm3_Vatjern
physics_head_Haukeli
physics_pool_fill_Haukeli
physics_supply_fraction_Haukeli
physics_dispatch_to_avoid_spill_Haukeli
physics_profitable_remaining_fraction_Haukeli
lp_qmin_ratio_Haukeli_G1
lp_power_fraction_Haukeli_G1
lp_economic_margin_Haukeli_G1
lp_storage_fill_Tokke
lp_storage_min_slack_Tokke
lp_storage_max_slack_Tokke
lp_plant_power_fraction_Haukeli
lp_storage_fill_Haukeli
lp_storage_min_slack_Haukeli
lp_storage_max_slack_Haukeli
norm_water_value_Langeidvatn
norm_price_minus_water_Langeidvatn
norm_water_value_Vatjern
norm_price_minus_water_Vatjern
norm_physics_switch_margin_minimum_Haukeli
norm_physics_switch_margin_maximum_Haukeli
```

## Lio_G1: 83 total columns

```text
volume_Byrtevatn
fill_Byrtevatn
price_over_water_Byrtevatn
headroom_after_inflow_Byrtevatn
inflow_Byrtevatn
inflow_week_mean_Byrtevatn
inflow_elapsed_mm3_Byrtevatn
inflow_remaining_mm3_Byrtevatn
inflow_r_Vest_Vassdraget
inflow_week_mean_r_Vest_Vassdraget
inflow_elapsed_mm3_r_Vest_Vassdraget
inflow_remaining_mm3_r_Vest_Vassdraget
min_volume_Byrtevatn
min_volume_week_max_Byrtevatn
min_flow_b_Byrtevatn
min_flow_week_max_b_Byrtevatn
min_flow_r_Vest_Vassdraget
min_flow_week_max_r_Vest_Vassdraget
physics_head_Lio
physics_pool_fill_Lio
physics_supply_fraction_Lio
physics_dispatch_to_avoid_spill_Lio
physics_profitable_remaining_fraction_Lio
lp_qmin_ratio_Lio_G1
lp_power_fraction_Lio_G1
lp_economic_margin_Lio_G1
lp_qmin_ratio_Byrte_G1
lp_power_fraction_Byrte_G1
lp_storage_fill_Hogga
lp_storage_min_slack_Hogga
lp_storage_max_slack_Hogga
lp_plant_power_fraction_Lio
lp_storage_fill_Lio
lp_storage_min_slack_Lio
lp_storage_max_slack_Lio
norm_water_value_Byrtevatn
norm_price_minus_water_Byrtevatn
norm_physics_switch_margin_minimum_Lio
norm_physics_switch_margin_maximum_Lio
```

## Hogga_G1: 86 total columns

```text
volume_Bandak
fill_Bandak
price_over_water_Bandak
headroom_after_inflow_Bandak
inflow_Bandak
inflow_week_mean_Bandak
inflow_elapsed_mm3_Bandak
inflow_remaining_mm3_Bandak
inflow_r_Vest_Vassdraget
inflow_week_mean_r_Vest_Vassdraget
inflow_elapsed_mm3_r_Vest_Vassdraget
inflow_remaining_mm3_r_Vest_Vassdraget
min_flow_b_Bandak
min_flow_week_max_b_Bandak
min_flow_r_Vest_Vassdraget
min_flow_week_max_r_Vest_Vassdraget
physics_head_Hogga
physics_pool_fill_Hogga
physics_supply_fraction_Hogga
physics_dispatch_to_avoid_spill_Hogga
physics_profitable_remaining_fraction_Hogga
lp_qmin_ratio_Hogga_G1
lp_power_fraction_Hogga_G1
lp_economic_margin_Hogga_G1
lp_qmin_ratio_Tokke_G1
lp_power_fraction_Tokke_G1
lp_qmin_ratio_Tokke_G2
lp_power_fraction_Tokke_G2
lp_qmin_ratio_Tokke_G3
lp_power_fraction_Tokke_G3
lp_qmin_ratio_Tokke_G4
lp_power_fraction_Tokke_G4
lp_qmin_ratio_Lio_G1
lp_power_fraction_Lio_G1
lp_plant_power_fraction_Hogga
lp_storage_fill_Hogga
lp_storage_min_slack_Hogga
lp_storage_max_slack_Hogga
norm_water_value_Bandak
norm_price_minus_water_Bandak
norm_physics_switch_margin_minimum_Hogga
norm_physics_switch_margin_maximum_Hogga
```

## Proposed comparison protocol

If approved for execution, train only the three selected LightGBM specialists using the exact350tree baseline configuration and eachfold's existing sampledrowindices. Preserve allhyperparameters, seeds, purgedcaseIDs and case-hourorder; only featurecolumns change. Compare perunitAUC and the full14-unit microAUC against the matched full-feature LPv1 baseline. A small jointreplacement grid [0,0.25,0.5,1] for the threeunits may be selected on2021 and frozen before2022, with no perunitweight search. Retain the previous acceptedcandidate if the change fails transfer; do not choose a different grid item using2022.

No fullfit should be launched before rootreviews the exact lists and authorizes the experiment. Main-repository transfer still requires at least0.998 on both chronologicalyears; these reusedyears do not constitute an untouchedholdout.

Baseline manifest SHA256:`bf7b863dc6094deb7bf8383c2068868b5dde936621a68bdd9373ae13fa5c25d8`. JSON SHA256:`8dce684894dac8e9132a40c5192f997a7b50257e10d9538c7f9d0194dc6f8161`.
