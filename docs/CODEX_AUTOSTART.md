# Start or resume with Codex

This repository carries its instructions, model preferences and four reusable
agent roles. A fresh Codex instance should read root [AGENTS.md](../AGENTS.md),
[GOALS.md](../GOALS.md), [HANDOFF.md](HANDOFF.md), and
[best_result.md](best_result.md), then reconcile real state before acting.
Opening or cloning the repository does not start training, create a persistent
Goal, restore private inputs, or submit to Kaggle.

## Fresh checkout

Install a current Codex CLI and sign in using its normal login flow. This setup
was checked against **codex-cli 0.166.0** on 2026-11-02. The startup checker needs
Python 3.11 or newer and only the standard library. The separate model execution
environments are described in [REPRODUCING.md](REPRODUCING.md).

```sh
cd deepthroatlearning
codex --version
bash scripts/codex-start.sh check
```

Review and trust this checkout in the normal Codex UI if it has not been trusted
yet. Codex deliberately skips project configuration in untrusted directories.
The checker refuses to claim the project setup is active if its configuration
layer is disabled. Do not automatically trust arbitrary downloaded repositories.

Fetch raw course inputs and the model release using
[DATA_AND_ARTIFACTS.md](DATA_AND_ARTIFACTS.md). That document identifies the course
archive/Kaggle entry point, exact destination paths, release groups and hashes.
Raw competition inputs are separate from the release. A source-only clone is
useful for code review but cannot reproduce the saved best without those files.
The preserved Round5/Round10 stack has its own historical artifact requirements.

```sh
# Inspect local files without contacting Codex. Model access remains unverified.
bash scripts/codex-start.sh check --offline

# Verify raw data and all best-file SHA256s as well as current model metadata.
bash scripts/codex-start.sh check --hashes

# Start an interactive coordinator and perform only the startup audit.
bash scripts/codex-start.sh start

# Resume authorized local improvement after restoration.
bash scripts/codex-start.sh improve
```

`start` is also the default when no argument is supplied. It allows missing data
so the agent can report the recovery steps, but the model/config check must pass.
`improve` requires all inputs and best artifacts to pass their recorded hashes.
It then asks the coordinator to inspect live jobs and free memory before choosing
one eligible hypothesis at a time. It does not launch a new hosted job or submit
to Kaggle from the startup request. The default budget is one hour **per
experiment**, including setup. Existing user authorization for larger time/resource
scope remains valid; the wrapper does not introduce another permission round.
Checkpoint after each bounded cycle and continue authorized ongoing goal work.
If an experiment cannot fit the authorized limits, preserve its protocol,
feasibility report and exact next step, then continue other useful authorized work.
A cycle boundary or small improvement is not competition success or cancellation
of a continuing goal. Respect the user's actual stopping conditions and blockers.

The wrapper writes small machine-readable reports under ignored
`models/codex-session/`. `check` exits 0 on successful checks, 1 for config/model
failures, and 2 for missing or invalid data/artifacts. `check --offline` can exit
0 without establishing model availability. `start` and `improve` hold an atomic
`models/codex-session/session.lock` while the CLI is open. Exit 73 means another
launcher lock exists. Inspect its host/PID and actual jobs before manually removing
a confirmed stale lock. Separate clones, direct CLI sessions and hosted jobs are
not covered by this local lock; the coordinator must inspect those too.

## Verify Astra and Ultra, then proceed

[.codex/config.toml](../.codex/config.toml) requests `gpt-6-astra` with `ultra`
reasoning for the coordinator and children. This is the user's requested choice;
it is not an account entitlement, a benchmark guarantee or proof of the model
running an already-open chat. Ultra is a supported-client reasoning/delegation
mode. It is separate from the `ultrafast` service tier, which this setup does not
enable. Raw API reasoning options and Codex client options can differ.

The checker starts a temporary stdio app-server, negotiates `initialize`, reads
`config/read`, and paginates `model/list`. It verifies the effective requested
model/effort and project configuration layer, and requires the same model/effort
to appear in the current catalog. It also requires effective child-agent settings
to enable agents, cap concurrency at three, and use that same model and effort;
missing, disabled or different settings fail startup. It uses a temporary SQLite state directory,
never reads credential files, sends no model turn and does not launch agent tools.
It stores only selected metadata, not a dump of personal config or raw logs.
Catalog evidence may come from the client's cache; it is not successful inference.

At the beginning of the actual session, inspect `/model` (including **More
reasoning…**) or the desktop model control. Record the selected model, effort,
Codex version, UTC time and evidence in the handoff. Also record successful or
failed initial model execution. Never infer model identity from an assistant's
self-description. Each delegated agent should report its runtime settings if the
client exposes them; otherwise mark that evidence unknown. Configuration changes
do not retroactively change an active chat.

There is no silent fallback. If Astra/Ultra is unavailable, retain the failure
and explicitly choose the highest suitable advertised setting. Prefer keeping
Astra and selecting `max` if the catalog advertises it but not Ultra. If Astra
itself is absent, use an available strong model such as `gpt-6.1-sol` with its
highest advertised reasoning level; do not assume a model or effort exists.
For example, **only after those exact values are advertised**:

```sh
DTL_CODEX_MODEL=gpt-6-astra DTL_CODEX_EFFORT=max bash scripts/codex-start.sh check
DTL_CODEX_MODEL=gpt-6-astra DTL_CODEX_EFFORT=max bash scripts/codex-start.sh start
```

The wrapper supplies the chosen pair both to the root and the `[agents]`
defaults. Role files deliberately inherit these settings so a fallback does not
leave children pinned to an unavailable model. Log the reason and chosen pair;
do not describe a fallback run as Astra/Ultra. For model-service errors, inspect
`codex doctor --summary` and normal login/network state. Missing access is not a
reason to bypass managed requirements. Offline source analysis can continue in
an already-working client while recording the limitation.

## Agent ownership and execution limits

The user explicitly requested automatic agent deployment during improvement.
The root coordinator should delegate independent work without another permission
round. At most **three spawned agents** may be active in addition to the root;
children do not spawn grandchildren. Use only roles that help the current phase.
The root performs the coordinator role; its reusable definition is also available
for delegating a coordination task in another workflow.

| Role | Owns | Required result |
| --- | --- | --- |
| [coordinator](../.codex/agents/coordinator.toml) | Shared handoff/experiment/best-result records; integration and resource leases | Reconciled state, selected protocol, decision and exact next action |
| [research](../.codex/agents/research.toml) | One assigned new protocol/research note | Prior evidence, legal features, controls, falsifiable test and resource estimate |
| [experiment](../.codex/agents/experiment.toml) | Assigned new source/config/run directories | Commands, manifests, hashes, measured comparisons and failures |
| [independent_audit](../.codex/agents/independent_audit.toml) | Read-only evidence review; assigned new audit report | Independent pass/reject/incomplete conclusion with evidence |

Before spawning, give each child a run ID, exclusive file list, output contract,
resource lease and stop condition. Children send shared-ledger changes back to the
coordinator. Custom role instructions do not create separate sandbox boundaries;
all children inherit the root's permissions. The wrapper retains workspace-write
with normal approval handling and does not bypass trust, rules or permissions.

Default resources are **one heavy local job and four total CPU threads**. The
wrapper exports BLAS/OMP thread limits; the experiment runner must also set the
model's `n_jobs`/`thread_count` and verify actual usage. Inspect free RAM against
the declared peak before launch, and obey stricter historical protocol limits.
Do not run full checkpoint replay concurrently with a feature build. Historical
GPU protocols allow at most two saved VMs, but that is a ceiling, not authorization
for a startup script to launch them. Hosted work requires explicit scope and a
coordinator lease after checking actual remote state.

For difficult modeling reasoning, root instructions still require a GPT-6 Pro
consultation through computer use when available. Record the actual model label,
prompt, recommendations, and experimental decisions. “Pro” without a verified
generation does not establish GPT-6 Pro; record the limitation and continue.

## Durable continuation

At each phase boundary and before stopping, the coordinator updates
[HANDOFF.md](HANDOFF.md) and [experiments.md](experiments.md), preserving failures
as well as wins. Include UTC time, actual model evidence, source revision,
run/config/protocol IDs, exact commands, data/output hashes, actual local and
hosted job state, resource owners, completed checks, decisions, blockers and the
next safe command. The full experiment record requirements remain in AGENTS.md.
An old process ID, queued notebook, browser handle or agent name is historical
until checked. Never restart an expensive job merely because its output is absent.

To continue in an app/IDE without the wrapper, use this prompt after selecting
the model and running the checker in a terminal:

> Read AGENTS.md and docs/CODEX_AUTOSTART.md, reconcile GOALS.md, docs/HANDOFF.md,
> current artifacts and real local/hosted jobs. Act as coordinator. Use the
> research, experiment and independent_audit roles when independent work helps;
> at most three children, one heavy local job, four CPU threads, exclusive paths
> and no grandchildren. Verify and record the actual model/effort. Continue one
> eligible bounded experiment cycles with frozen validation rules and independent
> audit, checkpointing between cycles and continuing the authorized ongoing goal.
> Preserve the best, record failures and the exact next step. Do not submit
> to Kaggle or launch new hosted jobs from this request.

The user's existing continuous first-place objective remains applicable when
improvement is resumed. A thread-scoped Goal can be started as described in
[GOALS.md](../GOALS.md); a repository document is not a running scheduler and does
not restore another thread's Goal or token budget. This distinction does not revoke
existing user authorization or impose an automatic stop after one experiment.

## Verification and official references

On 2026-11-02 the installed CLI help and generated app-server schemas were checked
against version 0.166.0. Local offline preflight found all **10** required input
files and **504** best artifact files with matching sizes. That observation
describes the migration workspace, not every clone. Full replay and model-array
portability evidence are maintained separately in [REPRODUCING.md](REPRODUCING.md).

The nested live CLI metadata attempt could not complete in the migration sandbox:
the new checkout was not yet trusted in personal Codex configuration and app-server
initialization returned `Operation not permitted`. No model inference or account
access is claimed from that attempt. Shell syntax and 21 deterministic fake-CLI
tests in [test_codex_startup.py](../tests/test_codex_startup.py) cover
complete/missing/corrupt inputs, unavailable model/effort, fallback propagation,
untrusted projects, duplicate locks and cleanup, unsafe arguments, and ignored,
disabled, mistyped or mismatched effective root/child settings. Run them with:

```sh
python3 -m unittest discover -s tests -p test_codex_startup.py -v
```

These tests use isolated temporary fixtures and test the wrapper, not OpenAI model
access. Rerun the real check in the user's normal trusted terminal before launching
from another instance.

Current official pages used for the configuration and startup protocol:

- [Config basics](https://learn.chatgpt.com/docs/config-file/config-basic): trusted project layers and precedence.
- [Configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference): model/effort and agent settings.
- [Subagents](https://learn.chatgpt.com/docs/agent-configuration/subagents): standalone TOML role schema, inheritance and concurrency.
- [Models](https://learn.chatgpt.com/docs/models): model selection and supported reasoning levels.
- [App server](https://learn.chatgpt.com/docs/app-server): metadata handshake, `config/read` and `model/list`.

The exact preference remains Astra/Ultra even if general documentation recommends
a different default. Availability and effective runtime selection must be checked
again on each new machine, account or client.
