# Coupled active-domain PQ diagnostics

The supplied-only three-binary diagnostic completed and **failed its declared follow-up flags**. It corrects the origin-inclusive power approximation for Hogga, Lio and Haukeli inside the complete water network, preserving the original initial heads, routing, constraints and terminal-water conversion. It does not use external numerical values or fit a classifier.

For each corrected unit, `physical_on_curve` locates feasible on-domain boundaries from the supplied topology’s turbine/generator efficiencies and nominal operating limits. Four equally spaced knots approximate its active-domain concave hull, excluding the origin. The model uses `Q = Qmin*u + sum(dQ)` and `P = P(Qmin)*u + sum(slope*dQ)`. The Qmin offset enters every original water-balance/minimum-flow row. The binary u controls both discharge and power. Nonpositive-price hours add segment-order binaries; positive-price solutions are checked against the same ordered curve. Root precision avoids treating an interpolated Qmin with slightly subminimum physical power as exact. The concave hull and four segments remain an approximation; sampled error and piecewise residual are recorded separately, without claiming exact nonlinear physics.

New source is `sintef_project/active_domain_water_budget.py`; original frozen cores remain unchanged. Structural independently reviewed its equations, variable transform, ordering, off-state and fallback semantics with no blocking finding. Four synthetic tests passed: true minimum power and insufficient-water off, downstream offset routing, negative-price ordering, and coupled switching costs/free boundaries. A preparation-only relative-path logging error after writing an unrun protocol is preserved in `artifacts/metric-trick-active-domain-prep-failure-001`. The corrected immutable protocol is `docs/metric_trick_active_domain_protocol_v2.json`, SHA256 `1735255924fc1ab010481492e1b6024bee4efee0b6670e23837a1a6a3855837c`.

The original fixed12 pre2021 cases are1,532,930,1461,1462,1594,1694,1826,1827,1958,2055,2186. Each old three-binary control is previously proven and fully hash-pinned. Candidate caps are5s MILP and two1s fixed-state LP replays per case. All input-only physical results were sealed before original labels were decoded. No2021/2022 target row, external simulator value, case/threshold/weight search or learned target is used.

| Diagnostic | Old three-binary control | Active-domain three-binary |
|---|---:|---:|
|All14 P/Pmin>=0.5 errors|2643|2657|
|Hogga+Lio+Haukeli errors|701|771|
|Hogga errors|319|276|
|Lio errors|46|159|
|Haukeli errors|336|336|
|All14 raw Q/Qmin microAUC|0.9422005883|0.9356298935|

All12 candidates had finite feasible incumbents, zero slacks and bit-exact fixed-state replay; none needed the explicitly defined continuous-LP fallback. Only3/12 proved optimal within5s. The largest relative incumbent gap was0.00017808; largest objective gap was€1321.65. The candidates therefore fail the mathematical proof gate as well as the aggregate label-fidelity gates. Lio degradation concentrates in low-price2020 cases1958 and2055; their Qmax and Pmax equal the old values exactly and Qmin changes by at most0.0002m³/s, so a large bound change does not explain the effect. The run does not establish whether geometry or remaining incumbent suboptimality drives that deterioration.

The largest sampled four-segment errors versus original nonlinear power across these cases are0.16448MW for Hogga,0.21929MW for Lio and0.08572MW for Haukeli. Every solved candidate’s equality to its declared active piecewise curve is checked separately to1e−5MW. The original three-unit failure remains closed and unchanged.

Receipt: `artifacts/metric-trick-active-domain12-001/manifest.json`, SHA256 `d5815c97e9122af6286c1e4df828de57ff0c8d6df5da93f77230de7868f5ed1d`; runtime55.565s. It retains every input/curve/result, source snapshot, control checksum, solver status/bound/gap, proof, pre-label seal, original target/ID mapping and unit/block metric.

```sh
.venv-ml/bin/python scripts/metric_trick_active_domain.py --protocol docs/metric_trick_active_domain_protocol_v2.json --protocol-sha256 1735255924fc1ab010481492e1b6024bee4efee0b6670e23837a1a6a3855837c --run-id metric-trick-active-domain12-001
```

After seeing this failure, the parent explicitly authorized a **separate prospective all14 experiment**: three corrected binary units cannot assess joint Tokke/Vinje commitment. This changes integrality and active-domain geometry for all14 together, compared with the same old three-binary controls. It is not a pure geometry ablation and does not revive the failed three-unit recipe. Its new source is `scripts/metric_trick_active_domain_all14.py`; protocol `docs/metric_trick_active_domain_all14_protocol.json`, SHA256 `37c81df4c8b0e557bf0e4d483f8d0c03d95be3794a9e7c445946dc4324857fec`. The rationale and previous failure’s hash were recorded before any new solve. It uses the same12 cases,10s MILP caps and two1s LP replays, preserves feasible bounds/fallbacks, and authorizes no full cache or learned model automatically.

```sh
.venv-ml/bin/python scripts/metric_trick_active_domain_all14.py --protocol docs/metric_trick_active_domain_all14_protocol.json --protocol-sha256 37c81df4c8b0e557bf0e4d483f8d0c03d95be3794a9e7c445946dc4324857fec --run-id metric-trick-active-domain-all14-12-001
```


The separately authorized all14 test also failed its aggregate fidelity check. All14 threshold errors increased from2643 to2682 and rawQ/Qmin microAUC decreased from0.9422005882720956 to0.9291318644778391. The weak-three error total moved701→699, but that small gain did not transfer to the full population. Tokke_G2 improved321→293 and Vinje_G3 improved476→436; several other units worsened, including Tokke_G1 77→108 and Byrte20→70. All12 candidates had feasible incumbents, zero slacks and exact fixed-state replays; only2/12 proved optimal within the10s cap, with maximum relative gap7.57916e−5 and objective gap€1078.889. No fallback was needed. This does not justify a full cache or supervised continuation.

The all14 run completed in106.987s; receipt `artifacts/metric-trick-active-domain-all14-12-001/manifest.json`, SHA256 `87778fde701b8f693255cc67f4763502313cd55fc5c2b2212933bcbcb5c0c86d`. It preserves all solver bounds and replay proofs. These results test a capped mixed-integer surrogate jointly changing active-domain geometry and integrality; they do not establish the optimum of the nonlinear shared-conduit scheduling problem. Both negative experiments remain closed.
