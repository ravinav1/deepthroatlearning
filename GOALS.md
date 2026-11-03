# Competition objective and continuation

The reconstruction objective is to match or exceed historical **0.99952 public
micro ROC-AUC**, using a reproducible numerical pipeline whose inputs, fitted
parameters and teacher targets come only from supplied data and independent
local computation. Research is externally informed, with prior external test-output
exposure disclosed. This reconstruction objective is **unmet**.

The historical 0.99952 submission was observed at rank 1 around 2026-11-03 03:09 UTC
and directly used external SHOP predictions. The best measured supplied-numerical-
data score is 0.99653 (retrospective supplied covariates); latest submitted 0.99650.
These are dated observations, not live ranks or organizer acceptance. Private
scores remain unknown. See [results/provenance](docs/best_result.md).

**Current scope: incorporate findings and the deferred plan only.** Do not start
training, reconstruction experiments, hosted jobs or submissions from this update.
The full [reconstruction plan](docs/RECONSTRUCTION_PLAN.md) is ready for a later
explicit resumption. Preserve the existing source, artifacts and release.

On resumption, use supplied-label chronological development selection, purge
six overlapping daily starts, and retain all negative/per-unit results. The 2021
and 2022 years have been repeatedly reused. A score of 0.998 on both years remains
a development target, not a transfer prohibition or proof of generalization.
Successful Kaggle scoring is not a decision about external-research admissibility.

## Resume in a new Codex session

Open this repository. Root `AGENTS.md` points to the current handoff and evidence.
The [Codex startup guide](docs/CODEX_AUTOSTART.md) configures Astra/ultra preferences
and reusable research, experiment and independent-audit agents. Use its startup
checker to verify effective settings and artifact availability.
Useful first checks are:

```sh
git fetch origin
git pull --ff-only
python src/hydraulic_surrogate.py --help
```

After restoring data/artifacts as described in `docs/REPRODUCING.md`, use:

> When explicitly resuming the deferred reconstruction, continue from AGENTS.md, GOALS.md and
> docs/HANDOFF.md. Verify the current best and inspect actual local/hosted job
> state first. Pick up the highest-priority eligible experiment without repeating
> rejected comparisons or changing frozen protocols. Preserve the best models,
> record every result and failure, and keep the handoff current.

For an explicitly persistent Codex Goal, the user can start a new thread Goal with:

```text
/goal Achieve first place in the TDT4173 2026 Kaggle competition with a reproducible submission. Follow this repository's AGENTS.md and GOALS.md, preserve the current best, verify leaderboard observations and exact checkpoint-to-submission replay, and maintain docs/HANDOFF.md and the experiment ledger through each improvement cycle.
```

Codex [project instructions](https://developers.openai.com/codex/guides/agents-md)
travel with the repository. [Goals are thread-scoped](https://developers.openai.com/cookbook/examples/codex/using_goals_in_codex),
so the old thread's active state, token budget, process handles, and browser session
are not transferred or automatically activated by cloning this repository.
