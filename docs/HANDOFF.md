# Current project handoff

## Selected-pair delivery — 8 November 2026

The user chose catalogue **#3 (0.99657)** and **#4 (0.99654)** and requested the
report and notebooks to match. This supersedes the earlier prepared #3/#5 pair.
New local deliverables are in `output/final_submission_selected_3_4_20261108/`:
`canvas/` contains the seven-page `Report.pdf`, `Short notebook 1.ipynb`,
`Short notebook 2.ipynb`, and `requirements.txt`; the two ZIPs package Canvas files
and editable Overleaf source separately. The previous short-report paths under
`output/pdf/report_plain_language_20261108/` are also updated to this pair.
Authors: Hallvard Grundt 115526, Raviksan Navaratnarajah 109485,
Bøye Tvenge Einang 139410. The report and notebook headings use **Group 167**
only, at the user's request; the live Kaggle team name was not changed.

The final report review checked the grading guide, source titles/claims, model
settings, results and all citations. It now uses plainer explanations, clarifies
hyperparameter selection, keeps four relevant course references, and includes the
requested AI-use declaration. At the user's later request, the report's data-source
statement concerns only the two submitted models; historical provenance records
are unchanged. Earlier model comparisons, public milestones and unsuccessful
approaches remain in Sections 5–6, as the user clarified; only the outside-data
discussion was removed. See `output/final_submission_selected_3_4_20261108/evidence/final_report_audit.json`.
The delivered requirements file has no comments. Both deliverable ZIPs contain
the latest report/source; notebook code is unchanged by the final report review.
The user subsequently requested removal of the report's verification-status
paragraph and will handle that verification. The paragraph is removed; the actual
remaining execution checks below are unchanged. Notebook author headings and the
authoring template include Tvenge; executable notebook cells are unchanged.

Notebook 2 now reproduces the recorded single-startup-seed plus 5% separate-generator
legacy LightGBM recipe, including seeds 299/673, fixed tree counts, sampling,
CSV round-trip and both Vinje projections before the final Tokke projection.
Source is pinned to fetched `origin/main` revision `6ccab5c`. The selected-pair
delivery is published on `experiment/final-submission-20261108`, a clean branch
from that main revision. The separate experiment branch and historical
pipelines/artifacts are unchanged.

Code audit, notebook schema/import checks, five root hydraulic integration tests,
and 55 affected archived tests pass. Independent comparison of legacy features
across six cases and all 14 generators is bitwise equal to original source.
The PDF compiles cleanly and was visually checked. See
[`submission_pair_preparation_20261108.json`](../reports/submission_pair_preparation_20261108.json)
for hashes, checks and the linked independent audit. No fits, new AUCs, 2022 looks or
Kaggle actions were performed during preparation. The unrelated full release test suite
was interrupted; no full-suite pass is claimed.

**Remaining verification:** neither complete notebook has run from raw inputs
in the consolidated CUDA environment. Total runtime/peak RAM and exact output
reproduction remain unverified; #4's historical CSV/component arrays are absent
locally. This Mac has CPU-only XGBoost 3.2.0. The next step is an unchanged full
notebook run on pinned Python 3.11.15/CUDA XGBoost 3.4.1 with one T4, four CPU
cores and 32 GB RAM, checking the generated receipts against the historical CSV
hashes. Do not confuse these preparation checks with a successful complete replay
or renewed authorization for competition improvement/automatic release.

Runtime model identity/effort was not exposed by this client; the earlier startup
catalogue/configuration check does not prove actual runtime settings. Old process
and hosted-job records below remain historical; no hosted job was launched here.

Updated 3 November 2026. This is a **findings and plan incorporation**, not a new
reconstruction run. The latest instruction is to stop after adding this material
to DeepthroatLearning. Do not launch the deferred work until it is resumed.

## Recorded results

**3 November 2026, later the same day:** a supplied-only attempt reached **public 0.99657** (anchor + startup16 pooled LightGBM two-seed bag). The organizer disallows external data, so the 0.99952/0.99795/0.99690 entries are ineligible. See [supplied_only_attempt_20261103.md](supplied_only_attempt_20261103.md).

| Track | Public score | Evidence |
| --- | --- | --- |
| Historical external AllMIP blend | **0.99952**, rank 1 at approximately 03:09 UTC | Matching SHOP 17.17 envC solver states blended with the earlier external NoMIP ensemble |
| Historical external NoMIP blend | **0.99795**, rank 5 at 03:07 UTC | Half prior 0.99690 predictions, half SHOP 17.17 envC relaxed commitment states |
| Best supplied numerical ancestry | **0.99653** | Retrospective971; future historical covariates were supplied, but prospective availability/admissibility is not established |
| Latest submitted supplied numerical ancestry | **0.99650** | Canonical115 63-leaf capacity blend, submitted 04:15:15 UTC |
| Startup142 and busbar144 | Unsubmitted | Existing completed/audited final artifacts; no public score |

See [best_result.md](best_result.md), the [score ledger](../reports/kaggle_scores.json),
and [preserved evidence index](../reports/reconstruction_20261103/evidence_index.json).
Private scores/current ranks are unknown; no leaderboard refresh was performed.

The external dataset was discovered through autonomous experimentation and Kaggle
submissions. The research is externally informed and previously exposed to
external test-date outputs. Supplied numerical ancestry is a separate claim.
No externally derived predictions or constants may silently enter the proposed
reconstruction. No new external test-date inspection is part of that plan.

## Findings and next plan

- [Information boundary](reconstruction_information_boundary.md): supplied files,
  verified artifact ancestry, external numerical use and research-selection exposure.
- [Mechanism evidence](reconstruction_mechanism_evidence.md): empirical unit ordering,
  startup economics, shared losses, water opportunity costs, pressure/tunnel routing,
  integrality, measured counterexamples and per-unit limitations.
- [Deferred reconstruction plan](RECONSTRUCTION_PLAN.md): full objective, numerical
  search requirements, validation, restricted replay and deliverables.
- [Organizer question](reconstruction_organizer_question.md): prepared, **not sent**.

The strongest unresolved hydraulic defect is unrestricted square-root tangent
extrapolation in signed tunnel flow. Earlier five-update beginning/end diagnostics
improved Q-ratio ranking but retained large nonlinear residuals and wrong-direction
flow. Residual backtracking is only a proposal: no implementation, frozen protocol,
tests or solver run was produced during this incorporation.

Existing startup142 and busbar144 final recipes/audits are preserved as research
records. The older Cat144 2021 GPU result selected 1,800 trees / 0.25; its 2022 bundle
and notebook were prepared, but no 2022 launch/result is established by this update.
Recheck actual hosted state on later resumption; do not duplicate historical jobs.

## Preservation and practical availability

The original imported `experiments/hydraulic_surrogate/`, its hash-pinned sources,
`configs/hydraulic_surrogate_best.json`, all existing numerical artifacts and the
20261102 release remain unchanged. The legacy adapter's `export-best` and
`replay-best` commands still reproduce **0.99652**, not the external winner.
Newer numerical artifacts remain in the original `tdt4173/artifacts/` workspace;
this update imports documentation, source-map metadata and small receipts only.
It does not claim new model portability or a restricted raw-input-to-model replay.
Original workspace paths in copied reports are historical references.

The evidence index explicitly records one unavailable referenced receipt for the
latent-model submission; no replacement was invented. A prior handoff is preserved
as [historical context](../reports/reconstruction_20261103/research/pre_incorporation_handoff.md).

## Work performed and stopped in this incorporation

Existing sources, protocols and receipts were read; Git origin was fetched.
The startup metadata checker failed (`Codex metadata service exited`), so its
requested Astra/ultra catalog identity remains unverified. It reported 10/10 raw
files and 504/504 baseline artifact files present, not a new numerical replay.
An OS restriction availability probe failed with `sandbox_apply: Operation not
permitted`; no restricted replay, guard/interposer, training or new physics solve
was launched. The process-level fallback was deferred before implementation.

Computer use located an existing research consultation, but no new prompt or
verified GPT-6 Pro consultation occurred. No new external-output data was read
for the incorporation. Agents stopped the proposed reconstruction work when the
scope was narrowed. No organizer message or new Kaggle submission was sent.

Run documentation/metadata checks before committing. On later resumption, read
AGENTS.md and the deferred plan, verify actual jobs and model identity, and freeze
one bounded experiment before implementation. Existing empirical gains alone do
not establish reproduction of 0.99952.

## Final submission preparation, 8 November 2026

The user selected the 0.99657 two-seed startup16 and 0.99656 single-seed startup16
recipes for Group 167, team deepthroatlearning. This supersedes the earlier
suggestion to pair 0.99657 with the 0.99652 anchor. See `final_submission/` for
plain LaTeX (no added packages), two standalone notebooks, software requirements,
verification receipts and the release asset index.

Both selected historical CSVs recompose byte-for-byte. Representative fresh
hydraulic solves, the rebuilt 664-column table using archived LP blocks, the
training sample, one newly fitted LightGBM unit and sampled pooled-model
predictions match their archived counterparts. Complete CUDA training of the
consolidated notebooks and their combined time/memory limits remain unverified.
Full names and student IDs are still placeholders. No Canvas submission or
Kaggle final-selection state was changed. The ordinary deadline in the supplied
guide is 8 November 2026 at 22:00 Oslo time.
