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
