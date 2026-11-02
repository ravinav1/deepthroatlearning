# Hydraulic surrogate research

This package preserves the model, feature, validation, audit and hosted-training
code imported from the `tdt4173` workspace. Its files are hash-pinned by the
original protocols and the root `reports/hydraulic_surrogate/source_snapshot.json`.

Start from the repository's [current handoff](../../docs/HANDOFF.md),
[best result](../../docs/best_result.md), and [setup](../../docs/REPRODUCING.md).
Use `python src/hydraulic_surrogate.py` from the repository root for setup,
artifact recovery, exact best export/replay, and archived script execution.

`SOURCE_AGENTS.md`, `SOURCE_README.md`, and `docs/` retain historical instructions
and observations verbatim. They are provenance, not current session state. The
root AGENTS.md and handoff supersede statements about deferred migration, old
browser access, active agents, pending scores, and completed/rejected experiments.
The supplied scorer inside this package is experimental; the root
`evaluation/kaggle_metric.py` remains the unchanged official scorer.

The archived `pyproject.toml` pins the Python 3.11 research environment. Keep it
separate from the root Python 3.12 stack. Some old launch/audit scripts require
macOS resource commands or exact historic environment identity. Their original
commands are preserved as evidence; use versioned new protocols for new fits on
another platform rather than editing a sealed protocol.
