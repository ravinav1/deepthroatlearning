# Current project handoff

Read root `AGENTS.md`, `GOALS.md`, [best_result.md](best_result.md), and
[DATA_AND_ARTIFACTS.md](DATA_AND_ARTIFACTS.md). The canonical repository is
**ravinav1/deepthroatlearning**. This handoff supersedes live-status statements in
the imported archive. The earlier Round pipeline handoff is preserved in
[PRE_HYDRAULIC_HANDOFF.md](PRE_HYDRAULIC_HANDOFF.md).

## Ready to use

The best recorded public result is **0.99652, rank 18**, observed 2026-11-02
22:04:37 UTC. Its CSV SHA256 is
`1c1d173816127997e711d54c8269406830666af66b6f4eb6d24f4f227b70d496`.
Private score and numeric submission ID are unknown. The recipe is
`configs/hydraulic_surrogate_best.json`; dated observations are merged into
`reports/kaggle_scores.json`. The older Round5 reference and Round10 candidates
remain intact with their code, configs and historical evidence.

`src/hydraulic_surrogate.py` provides setup, artifact restoration, exact export and
CPU replay of all 98 component checkpoints. Frozen source and protocols live in
`experiments/hydraulic_surrogate/`; restored artifacts use ignored
`models/hydraulic_surrogate/`. Follow [REPRODUCING.md](REPRODUCING.md).

## Latest measured outcomes

| Experiment | Evidence and decision |
| --- | --- |
| LGB664–SHOP best | Public 0.99652; purged development 0.9951042081969695 / 0.9938717710733546. Retain. |
| Earlier SHOP664 blend | Public 0.99651. It remains the frozen anchor for existing declared protocols. |
| Fullsample608 final | Complete submissions: primary 0.99643, standalone 0.99631. Below the best. Do not repeat saved v10 or either submission. |
| River726 | 2021 weight 0.25: 0.9951306250691139. Locked 2022: 0.9938778756239264, below matched LGB664 control 0.9938790447855327. Actual result and independent audit reject promotion. No conditional final fit. |
| Fullsample608 blended into current SHOP | Rejected on locked 2022; no rescue weights or export. |
| CPU vector14 | Synthetic capability passed. Timing pilot failed its declared 2 GiB limit; no full-fold result. |

River726 finished after the older source handoff was written. Actual evidence is
`lgb-river726-2022-001/manifest.json` and
`lgb-river726-2022-independent-audit-001/manifest.json`; migration verification
records the observed hashes. Archived statements that it is running are historical.

## Resume improvement

1. Verify data/artifact/source hashes and inspect actual local or hosted jobs.
   Historical process IDs, session handles, agent owners and browser state do not
   transfer to a new session. Prepared notebooks do not establish a hosted launch.
2. Reconcile `integer-water-budget-full-four-worker-001`. It was incomplete at the
   snapshot. Per-case outputs are preserved for recovery; case 2012 has a separately
   reviewed numerical exception. Read the exact continuation and aggregate
   protocols, inspect original live state, and avoid duplicate workers. All cases,
   receipts and independent physical audit must pass before using the cache as
   features. The 748 model remains conditional.
3. Check whether prepared **SHOP664 fullsample2021** and **additive745 2021** jobs
   were launched after the saved checkpoint; the last recorded queue did not yet
   observe launches. Read imported `docs/next_gpu_queue.md`,
   `cuda_fullsample_parallel_execution.md`, and `cuda_xgb745_execution.md`.
   Rebuild omitted private upload stages from supplied local inputs and preserved
   caches. Keep at most two saved GPU VMs and the frozen smoke, return, and
   2021-lock-before-2022 checks.
4. Canonical pooled early-2020 and CPU vector14 work are prospective. Read their
   v2 protocols, reviews and resource limits before deciding on a bounded next run.
   Do not retune rejected river/calibration/smoothing/fullsample comparisons on 2022.

Optional research/physics release groups preserve negative experiments, other
models and predictions, hosted returns, completed river cache and partial integer
state. The inventory and manifests distinguish preparation, failure, partial and
completed runs. Migration starts no new background training or Kaggle submission.

For difficult reasoning, earlier computer use showed “Pro” but did not verify a
GPT-6 generation. A consultation was pending in the earlier session; recheck access
and model identity. This migration session had no computer-use tool, so no new
GPT-6 Pro consultation is claimed. Prompts and observations remain in the imported
consultation records.

The 2021/2022 years are reused development sets. Preserve chronological purging,
all-hour scoring, frozen controls and nested projections. First place remains
unmet. The explicit migration request supersedes the older deferred-transfer rule.
Use new run IDs and keep this handoff, the experiment ledger and best recipe current.
