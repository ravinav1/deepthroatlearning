# TDT4173 hydropower scheduling

## Final course submission — 8 November 2026

The updated **Group 167** package selects submissions **#3 (0.99657)** and
**#4 (0.99654)**. It supersedes the earlier pair in `final_submission/`.

- [Updated report PDF](output/final_submission_selected_3_4_20261108/canvas/Report.pdf)
- [Canvas ZIP: report, two notebooks and requirements](output/final_submission_selected_3_4_20261108/TDT4173_Canvas_selected_3_4.zip)
- [Overleaf ZIP: editable LaTeX](output/final_submission_selected_3_4_20261108/TDT4173_Overleaf_selected_3_4.zip)
- [Submission instructions and verification status](output/final_submission_selected_3_4_20261108/READ_ME_FIRST.txt)

Predict hourly ON/OFF probabilities for **14 hydropower generators over a
168-hour week**: **2,352 outputs per case**, evaluated with **micro ROC-AUC**.
The labels are schedules produced by an optimizer, and the model receives the
supplied prices, water values, reservoir conditions and operating constraints.

## Start here

- [Project handoff](docs/HANDOFF.md): current model, results, known weaknesses and how to continue.
- [Data and artifact downloads](docs/DATA_AND_ARTIFACTS.md): raw-data sources, release assets, restore commands and checksums.
- [Competition goals and Codex continuation](GOALS.md): durable objective and a fresh-session starting prompt.
- [Automatic Codex startup and agents](docs/CODEX_AUTOSTART.md): Astra/ultra preference, model checks, agent roles and resource ownership.
- [Results and provenance](docs/best_result.md): historical external-output winner and supplied-data models.
- [Reconstruction plan](docs/RECONSTRUCTION_PLAN.md): recorded for later implementation; currently deferred.
- [Mechanism findings](docs/reconstruction_mechanism_evidence.md): supported explanations, failures and open questions.
- [Information boundary](docs/reconstruction_information_boundary.md): supplied numerical ancestry and external research exposure.
- [Run on another computer](docs/REPRODUCING.md): setup, data placement and a new validation run.
- [Supplied files](docs/PROJECT_FILES.md): original task, dataset and topology documents.
- [Latest experiments](reports/round10_report_20261018.md): price ranks, temporal correction and hydraulic timing.

## Final submission package

The [Group 167 submission materials](final_submission/READ_ME_FIRST.txt) include
package-free LaTeX, two self-contained notebook implementations and verification
evidence for the selected **0.99657 / 0.99656** recipes. Compiled PDFs, exact CSVs
and importable ZIPs are release assets indexed in
[the delivery asset manifest](final_submission/release_assets.json).
Full names/student IDs and a complete CUDA raw-data retraining/resource check
remain pending. No Canvas upload or Kaggle selection has been performed.

## Current status

**Update, 3 November 2026 (later):** the organizer does not allow external data, so the external SHOP entries below are ineligible. The best eligible, supplied-only result is **public 0.99657**; see [the supplied-only attempt](docs/supplied_only_attempt_20261103.md).

The highest recorded public score is **0.99952**, observed at rank 1 around
**3 November 2026, 03:09 UTC**. It directly uses external SINTEF SHOP solver
outputs. The external dataset was discovered during autonomous experimentation
and the Kaggle submission sequence. It is not an independently reconstructed
optimizer or a supplied-data-only result.

The best recorded model with supplied-only numerical ancestry scored **0.99653**;
it uses retrospective historical covariates included in the supplied files. The
latest submitted supplied-numerical-data model scored **0.99650**. Numerical
ancestry does not establish organizer acceptance or clean-room research. The
research has prior external-output exposure; private scores are unknown.

This update incorporates findings and a **deferred reconstruction plan**. It
launches no new training, optimizer experiment, hosted run or submission. The
existing [artifact release](https://github.com/ravinav1/deepthroatlearning/releases/tag/hydraulic-surrogate-20261102)
still contains the older 0.99652 baseline's 98 component checkpoints; it has not
been relabelled as the 0.99952 winner or updated with the new numerical artifacts.

| Candidate | Historical result | Public Kaggle result |
| --- | --- | --- |
| **Supplied-only startup16 bag (3 Nov, later)** | 2021/2022: 0.9953148 / 0.9941981 | **0.99657**, best eligible entry; see [docs/supplied_only_attempt_20261103.md](docs/supplied_only_attempt_20261103.md) |
| External SHOP AllMIP blend | Reused 2021/2022: 0.99929508 / 0.99961071 | **0.99952, rank 1** at the dated observation |
| External SHOP NoMIP blend | Reused 2021/2022: 0.99680263 / 0.99796882 | **0.99795**, external outputs |
| Supplied retrospective971 | Supplied numerical data; retrospective covariates | **0.99653**, best measured in that track |
| Supplied canonical115 capacity | Latest submitted supplied-numerical-data model | **0.99650** |
| Supplied startup142 / busbar144 | Completed and audited locally before this update | Unsubmitted; no public score |
| Hydraulic LGB664–SHOP | Purged 2021/2022: 0.99510421 / 0.99387177 | **0.99652, rank 18**, dated UI observation |
| Round5 reference | Established comparison baseline | 0.99172, user-reported |
| Round10 price ranks | +0.00011 to +0.00016 AUC across three checks | Not recorded |
| Round10 temporal correction | +0.00002 to +0.00003 on the two 2022 checks | Not recorded |

The newer candidates are separate experiments. Historical checks reuse
validation years; their gains do not guarantee a public or private leaderboard
gain. See the [score ledger](reports/kaggle_scores.json) and
[Round10 report](reports/round10_report_20261018.md).

## First run

For a fresh Codex session, use `bash scripts/codex-start.sh check`, then
`bash scripts/codex-start.sh start`. After data/artifact recovery, `improve` starts
the documented local improvement workflow. The configuration requests Astra/ultra;
the startup guide explains project trust and actual runtime verification.

For the preserved **0.99652 baseline**, follow [REPRODUCING.md](docs/REPRODUCING.md) to set up the
separate Python 3.11 environment and restore artifacts, then run:

```sh
.venv-hydraulic/bin/python src/hydraulic_surrogate.py export-best --run-id first-export-001
.venv-hydraulic/bin/python src/hydraulic_surrogate.py replay-best --run-id first-replay-001
```

These legacy commands still target 0.99652, despite their `best` names.
Export reconstructs that CSV from saved probabilities; replay rebuilds
features and executes all 98 classifiers on CPU. Both require the original checksum.

For the existing Round10 pipeline, use Python 3.12 and start from a
**newly named copy** of `configs/round10_ranks_seed299_2021.json`, using
`src/round5_lightgbm.py`. Existing run names already have reports in Git and
will deliberately refuse to overwrite them. `src/train_baseline.py` is the
older pooled baseline, not the complete current ensemble.

A fresh clone can train new models after the supplied data is added. Replaying
the complete historical ensemble also needs ignored checkpoints or prediction
caches transferred separately, or regenerated from their component recipes.

## Repository layout

| Path | Purpose |
| --- | --- |
| `data/raw/` | Unchanged supplied CSVs; obtain separately |
| `data/interim/`, `data/processed/` | Generated data, prediction caches and intermediate files |
| `src/` | Features, trainers, evaluation and submission code |
| `experiments/hydraulic_surrogate/` | Frozen hydraulic code, tests, protocols and research history |
| `configs/` | Topology and named experiment recipes |
| `evaluation/kaggle_metric.py` | Supplied metric; preserve unchanged |
| `notebooks/` | EDA and supplied visualization notebook |
| `models/`, `submissions/` | Generated checkpoints and competition CSVs |
| `reports/` | Metrics, experiment decisions, audits and small figures |
| `docs/` | Task materials, setup and handoff documentation |

Git includes code, configs, documentation, notebooks and report summaries.
Datasets, models, submissions, environments and console logs stay out of Git.
Hydraulic checkpoints/predictions and useful solver state restore from checksum-indexed
release assets; supplied raw data is obtained separately. Root `AGENTS.md` gives
fresh Codex sessions the objective, working method, handoff and checks.

## Validation rules

Keep cases chronological and purge training cases whose 168-hour targets
cross the validation boundary. Construct complete-week features before
sampling training hours; evaluate every hour and generator. Fit only the
supplied labelled cases, never competition IDs or true neighbouring validation
labels. Preserve the template's case/column order and submit probabilities.

The complete earlier README is preserved in
[earlier experiment notes](docs/EXPERIMENT_HISTORY.md). Later reports under
`reports/` supersede historical statements about which model was best.
