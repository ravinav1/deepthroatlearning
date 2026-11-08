# Supplied-data-only attempt, 3 November 2026

**Outcome: public 0.99657, up from 0.99652, with supplied competition data only. The target of beating second place (0.99820) was not reached.** The organizer does not allow external data, so the team's 0.99952 / 0.99795 / 0.99690 entries, which blend public SINTEF SHOP solver outputs, are not eligible and were not used, directly or indirectly, in any model below. Full running log with per-experiment evidence: [`reports/supplied_only/log.md`](../reports/supplied_only/log.md).

## Information boundary

All numerical inputs are the ten supplied files plus caches whose ancestry is supplied-only (`lp-full-001`, `lp-joint-full-001`, `lp-shop-full-001`, the 98 anchor checkpoints), restored from the GitHub release after the original workspace disappeared mid-session. `scripts/supplied_only/common.py` refuses any path under the external-solver directories; every run writes an `inputs.json` with SHA256 hashes. No pseudo-labels, teachers, or tuning signals from external outputs. The research history of this project was exposed to those outputs earlier (see `docs/reconstruction_information_boundary.md`); that exposure is disclosed, not erased.

## Best eligible recipe (public 0.99657)

```text
A0   = preserved LGB664–SHOP nested anchor (public 0.99652, configs/hydraulic_surrogate_best.json)
S    = mean(startup16 seed 4173, startup16 seed 1)      pooled LightGBM, 1,400 trees, 31 leaves, lr 0.06
best = project_tokke(project_vinje(0.75·A0 + 0.25·S))
```

startup16 = the canonical115 generator-relative representation plus 16 fields built from the supplied topology (p_min, p_max, start and stop cost) combined with existing normalized margins and price windows (`scripts/supplied_only/startup_like.py`). The 664-column table was rebuilt from the research feature modules (`scripts/supplied_only/build_664.py`) and verified against the frozen column order. Export: `models/supplied_only/exports/so-bag2-anchor-startup16-001/` (CSV SHA256 `f7b0b52fc54e93e1db6171fdf85a7955486c3439897911291eef98c70a4f716a`), recipe `configs/supplied_only/bag2_anchor_startup16.json`, reproducible with `scripts/supplied_only/export_candidate.py`.

Development AUC (2021 selection / 2022 check, six-day purge, reused years): anchor 0.9951042 / 0.9938718 → best 0.9953148 / 0.9941981.

## What was tried

| Lever | 2021 | 2022 | Public | Decision |
|---|---:|---:|---:|---|
| D1 overlap-label diagnostic | agreement 0.878 at 1 day | | | cross-case consensus levers dropped before fitting |
| Tokke partial-order projection | +0.0000006 | +0.00000005 | | kept, negligible |
| Per-plant blend weights | +0.00023 | −0.00011 | | rejected |
| Global weights startup142 .25 + busbar144 .10 (lost with workspace) | +0.00012 | +0.00009 | | accepted, not exportable |
| canonical115 seed bag | +0.00002 at 0.25 | n/a | | stopped (matrices lost) |
| Legacy Round10-style member, 0.05 on anchor | +0.00011 | +0.00008 | 0.99649 | did not transfer, dropped |
| **startup16, 0.25 on anchor** | **+0.00019** | **+0.00030** | **0.99656** | accepted |
| startup16 + legacy 0.05 | +0.00027 | +0.00034 | 0.99654 | legacy dropped |
| **startup16 two-seed bag** | **+0.00021** | **+0.00033** | **0.99657** | accepted, current best |
| startup16 three-seed bag | +0.00021 | +0.00031 | | worse than two seeds |
| startup16 + cross-case LP states | +0.00021 | +0.00023 | | rejected |
| startup16 with volatility weighting | +0.00020 | +0.00025 | | rejected |

## Honest assessment

The supplied-only ceiling in this project remains around 0.9966 public. Everything that transferred was a small representation change on top of an already strong physics-surrogate ensemble. The gap to second place (0.0016) is the gap between an approximate LP surrogate and the actual SHOP optimizer; the only way this project crossed it was with the solver's own outputs, which are not allowed. Untried directions with any plausible upside are expensive and uncertain: a tuned neural sequence member over the 664-channel weeks, a per-generator model family with the startup fields, and the parked Haukeli tunnel reconstruction.

## Final-submission selection

On Kaggle, zero of two final submissions were selected at the time of writing, so the ineligible 0.99952 file would be auto-selected. The two eligible entries to select are `so-bag2-anchor-startup16-001` (0.99657) and the original anchor `submission-lgb-shop664-anchor-blend-001` (0.99652).
