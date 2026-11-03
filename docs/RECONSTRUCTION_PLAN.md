# Externally informed reconstruction plan

Status: **recorded for later work; do not start from this incorporation request**.
The current task is to incorporate the findings and this plan into the repository.
No new training, reconstruction experiment, hosted run or Kaggle submission is
authorized by opening this document. Resume implementation when requested.

## Objective and historical reference

Reconstruct as much of the useful scheduling mechanism as possible from supplied
competition inputs, topology and training labels. Aim to match or exceed the
historical **0.99952 public micro ROC-AUC**, with a final numerical pipeline that
can be reproduced without downloaded solver outputs.

The external dataset was discovered during autonomous experiments and the Kaggle
submission sequence. The research is externally informed, with prior exposure to
external outputs for competition test cases. It is not a clean-room reconstruction.
Read [best_result.md](best_result.md) and the preserved
[historical recipe](../reports/reconstruction_20261103/research/best_result.md).
The historical winner directly blended SHOP 17.17.0 NoMIP and AllMIP outputs and
retained a SHOP 17.15.1 contribution. It was not an independently reconstructed
optimizer. Preserve that provenance and all existing artifacts.

## 1. Establish the actual information boundary

Inventory original supplied inputs, topology, constraints and training labels;
existing model artifacts whose complete numerical ancestry uses only supplied
data; external solver archives; and externally influenced experiments.

Distinguish research inspiration from numerical inputs, fitted parameters,
teacher targets and selection signals. Removing external files from inference
does not remove information already learned from them. Record prior exposure,
including historical leaderboard feedback. Do not assume a small set of
thresholds can reproduce the historical external-output score.

## 2. Extract mechanisms rather than an encoded answer table

Any further external-output investigation must use dates in the original
training period only. Do not use external outputs for competition test dates to
develop, select or tune the reconstruction. Record the research slice and why it
was chosen; a case selected after external-error inspection is not an unbiased
validation sample.

Investigate generator preference/allocation, minimum operating levels and
efficiency, startup/shutdown economics and persistence, shared penstocks and
plant coordination, tunnel bottlenecks and changing pressure/head, reservoir
opportunity costs and downstream coupling, terminal water values, environmental
constraints and forced operation, relaxed versus integer commitment, and possible
tie-breaking behavior.

For every hypothesis record its external inspiration, physical interpretation,
supplied-file evidence, alternatives and a test that could falsify it. External
solver behavior is not ground truth for selecting the reconstruction.

## 3. Independently establish numerical choices

Prefer parameters derived directly from supplied topology and explicit physical
relationships. Estimate other parameters through a real search on supplied
training labels. Define the domain from physical constraints, documentation or
training-data analysis and record every evaluated configuration and the selection
rule. Do not copy an external threshold and surround it with a token search.
Do not add date-specific or case-specific exceptions to reproduce archived
decisions.

Estimate and evaluate generator ordering, thresholds and switching behavior on
supplied labels. Measure competing explanations when multiple ones fit. Cite
papers only for mechanisms/formulas they establish, not unsupported numerical
values. Preserve known failures and limitations.

## 4. Build a working reconstruction

Prioritize mechanisms explaining measured errors in supplied-data models. Options
include a constrained hydraulic optimizer; explicit features from discharge,
storage, relaxed commitment, constraint slack and marginal water values;
supervised correction trained on supplied commitments; and generator allocation
or temporal decision models fitted on supplied labels.

Keep source code and explicit formulas for every feature. Keep external numerical
files, external teacher predictions and externally fitted constants out of final
feature generation, training, parameter selection and inference. Source every
constant and learned parameter. Existing cached features/checkpoints require
ancestry evidence, not just a statement that the latest script reads no external
files.

The next proposed hydraulic experiment is residual-controlled backtracking for
signed tunnel pressure equations. Its undamped predecessor has substantial
nonlinear feasibility errors. This proposal is **not an established improvement**;
declare its step domain and acceptance rule before running it. Use physical
residual and linear feasibility for numerical convergence, then supplied-label
validation for predictive selection. Do not interpret mixed-iterate dual values
as valid optimizer marginal water prices.

## 5. Validate the explanation

Use chronological splits and purge six daily case starts before each held-out
year to separate overlapping 168-hour horizons. The 2021/2022 periods have been
reused extensively and must be described as development sets.

For each change, retain the same supplied-data baseline, change one mechanism or
a clearly defined group, record the entire search domain/results/selection rule,
report pooled and per-unit performance, verify physical feasibility and test
transfer across years. Select on 2021, lock the choice, evaluate unchanged on
2022. Preserve negative outcomes. A small diagnostic chosen during external
research cannot substitute for this chronological evaluation.

Re-run the numerical pipeline with access restricted to approved supplied files,
verified supplied-ancestry artifacts and software dependencies. Test that the
restriction actually denies unapproved reads, and state its coverage/limitations.
This verifies computational provenance; it does not erase research history.
Do not claim a full raw-input-to-model rerun if only cached inference was replayed.

## 6. Deliver a reproducible result and accurate explanation

Provide the final model and CSV; exact commands, dependencies, seeds, parameters
and checksums; a source map for all features/constants/learned parameters; a table
showing which mechanisms improved measured performance; external inspiration
versus independently estimated quantities; and the measured gap to 0.99952.

Never invent searches, derivations, validations or attribution. Label unsupported
mechanisms. Successful Kaggle scoring is not organizer approval of the development
method. Prepare, but do not send without authorization, this question:

> During autonomous experimentation we discovered public SHOP outputs matching
> competition cases and previously evaluated submissions using them. For a
> subsequent model, may we use that research to formulate mechanisms, while
> deriving numerical parameters from supplied topology or estimating them only
> from supplied training labels, with no external predictions, teachers or
> test-date outputs used in further development? What disclosure or restrictions
> do you require given the prior exposure?

When resumed, continue through implementation and meaningful validation. If the
historical score cannot be reproduced, report the measured result and unresolved
mechanisms rather than rewriting provenance.
