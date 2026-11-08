TDT4173 -- Group 167 -- selected submissions #3 and #4

Canvas files (also in TDT4173_Canvas_selected_3_4.zip):
  Report.pdf                  Seven-page short report
  Short notebook 1.ipynb      #3, public ROC-AUC 0.99657, two startup16 seeds
  Short notebook 2.ipynb      #4, public ROC-AUC 0.99654, one startup16 seed + 5% legacy
  requirements.txt           Pinned software dependencies
Extract the ZIP and upload the four files according to the Canvas instructions.
The separate Overleaf ZIP contains editable main.tex and its compile instructions.

Authors:
  Hallvard Grundt -- 115526
  Raviksan Navaratnarajah -- 109485
  Bøye Tvenge Einang -- 139410
Group: 167.

Running the notebooks:
Use Python 3.11.15, the pinned packages and one NVIDIA Tesla T4 (16 GB).
Install packages before the offline run. Set RAW_DATA in each notebook to the
supplied sintef_project folder containing data/, prediction_mapping.csv and
sample_submission.csv. Run each notebook in a fresh kernel. Each is independent
and needs no repository, existing checkpoints, predictions or extra code files.
The only numerical inputs are the ten supplied files listed in the setup cell.

Chosen submissions (catalogue numbers, not Kaggle submission IDs):
#3: 3 November 2026 13:07:32 UTC; public 0.99657.
#4: 3 November 2026 12:53:37 UTC; public 0.99654.
Both historical CSV hashes are checked by the notebook export cells.
This preparation did not change live Kaggle selection controls or submit files.

Verification limits:
Code, embedded imports, feature construction and blend order have been checked.
A complete raw-data CUDA run is still required to measure total runtime/RAM and
confirm exact historical output. This Mac has no CUDA; full execution was not
performed here. The #4 historical CSV and component arrays are unavailable locally.
The final notebook cell records hashes, runtime and peak process memory, checks
12 hours/32 GiB, and rejects an output that misses its historical CSV hash.
The stated public scores are historical observations, not new notebook results.

Evidence and authoring directories are for your review, not required notebook inputs.
Historical #3/#5 checks are clearly labelled and do not certify the new #4 notebook.
No new validation scores, training runs, 2022 evaluations or Kaggle actions were made.
