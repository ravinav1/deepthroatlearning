# Experiment ledger

The repository retains both experiment histories:

- [Round1–Round10 reports](../reports/) and [earlier handoff](PRE_HYDRAULIC_HANDOFF.md).
- [Hydraulic experiment ledger](../experiments/hydraulic_surrogate/docs/experiments.md),
  including failed fits, rejected hypotheses, protocols, source hashes and commands.
- [Merged public-score ledger](../reports/kaggle_scores.json); leaderboard results
  are distinct from development validation scores.

## Hydraulic migration — November 2026

Hypothesis: the accepted hydraulic ensemble and improvement workflow can coexist
with the existing Round pipelines and reproduce the same submission after
relocation. No model selection, refitting, or Kaggle submission is part of this run.
The user explicitly requested migration, commit, push, conflict resolution, and
the canonical `ravinav1` remote; this supersedes the old deferred-transfer rule.

Starting remote revision: `3a9ae32f4ab7ad53aa95c0bde517fa86d673fcf6`, fetched before
changes. The writable checkout fast-forward-pulled the freshly fetched tracking
ref locally because direct network access for `git pull` was unavailable in the
session. Original source workspace: `/Users/heap/programming/tdt4173`; its files
were preserved. The older sibling checkout was outside writable roots.

The copied source has no Git history. Every copied byte is recorded in
[`source_snapshot.json`](../reports/hydraulic_surrogate/source_snapshot.json).
The artifact inventory records snapshot UTC time, all file and archive SHA256s,
sizes, exclusions, and downloadable release URLs. Raw CSVs, environments,
credentials, duplicate staging copies and private dataset upload archives are
excluded from publication. Completed models, predictions, solver caches,
protocol evidence, and useful continuation state are preserved separately.

Implementation: retain the frozen research layout under
`experiments/hydraulic_surrogate/`; add a root adapter and named config; use
`data/raw/` and `models/hydraulic_surrogate/` for local data/artifacts. Maintain
the official scorer and existing Round trainers byte-for-byte. No score claims
are revised except to merge later dated observations into the current ledger.

Validation commands, environment, runtime, exact replay and checksums are recorded
in [`migration_verification.json`](../reports/hydraulic_surrogate/migration_verification.json).
The root adapter's replay/export also writes complete success or failure manifests
under its new run ID. Old whole-machine protocol guards remain in the frozen
archive; they are not silently weakened for relocation.

New experiments should append a dated entry here or link a structured report.
Record the details required by AGENTS.md, including failures and the actual
selection/confirmation chronology. Do not overwrite imported manifests.

### Migration verification outcomes

The source was integrated and pushed as `0ab7df44e5b53895aaf8d0eb512d0ba9c333e93c`.
All 975 pre-existing source/config/scorer/report files checked against the base
revision remain byte-identical. The merged score ledger is an intentional update.
All 643 frozen source files and 159,566 transferred artifact files passed hashes.

The imported suite passed 419 tests; the root suite passed 26, including 21 Codex
startup regressions. The first imported run had five missing code-only fixture
errors; the original fixtures were added to research archive 09 and the suite passed.
The source files themselves were not edited to make the tests pass.

The first two CPU replay attempts failed at XGBoost608. Four LightGBM components
matched exactly, but local CPU XGBoost 3.2.0 differed from the recorded CUDA 3.4.1
outputs (maximum absolute error 1.4007091522216797e-6). This motivated a **new
adapter-only** 1e-5 CPU portability threshold, selected after that observation.
It does not alter original validators, model selection, training or probabilities.

Full migration replay 003 subsequently checked all 98 models in 83.235s: all 70
LightGBM checkpoints matched bitwise; GPU608/GPU664 CPU differences were
1.4007091522216797e-6 /1.1622905731201172e-6. The exact submission uses the
hash-verified original CUDA arrays after these checks. Its SHA256 matches the
original best. The executed adapter snapshot and both failed attempts are retained
in `reports/hydraulic_surrogate/verification/`. The official scorer independently
reproduced both recorded fold AUCs, including reversed-ID alignment.

Two independent review/setup agents were explicitly requested as GPT-6 Astra with
ultra reasoning. Their review found and corrected missing child-model setting
validation and failure-provenance gaps. The separate nested CLI metadata startup
could not complete under sandbox/project-trust restrictions; that limitation is
recorded rather than treated as model availability. Offline startup checks passed
all 10 input and 504 best-artifact hashes; no personal trust/auth state was changed.

Release `hydraulic-surrogate-20261102` was published at 2026-11-02T23:06:58Z with
20 archives and 2 compressed inventories. All 22 server-reported sizes/SHA256s match
the committed index. The release targets the integration commit. Fresh remote
checkout, anonymous release download and recovery verification are recorded in
the final migration verification report.
