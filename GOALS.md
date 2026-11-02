# Competition objective and continuation

The objective is **first place with a reproducible, valid submission** in the
[TDT4173 2026 competition](https://www.kaggle.com/competitions/machine-learning-task-tdt-4173-2026).
This objective is **unmet**. The latest stored observation is public AUC **0.99652**,
rank **18**, on 2026-11-02 at 22:04:37 UTC. Private score is unknown. The leaders
at that observation were 0.99820, 0.99820, and 0.99805; these are dated observations,
not claims about the live leaderboard.

Milestones retained from the earlier work are an actual public score above 0.998
and top-three rank, with 0.999 an earlier stretch target. The validation target is
at least 0.998 on **both** purged 2021 and 2022 development years with one frozen
selection rule. Current best scores are 0.9951042081969695 and 0.9938717710733546.
Do not equate validation with the leaderboard or mark the objective complete
because a smaller improvement or the repository migration is complete.

The user explicitly requested this migration before the old validation transfer
gate was met. The migration is independent of achieving the competition target.

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

> Continue the competition improvement work from AGENTS.md, GOALS.md and
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
