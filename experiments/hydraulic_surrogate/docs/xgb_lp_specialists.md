# XGBoost LP specialists — predeclared experiment

Prepared2026-11-02 before fitting. Hypothesis: XGBoost may make complementary errors on Haukeli,Lio,Hogga, the difficult units, even when given the same input-only hydraulicLP features as the baseline LightGBM.

Baseline:`lgb-normalized-lp-001`, frozenLP v1, including physical/context features and monetary normalization. Rebuild exactly the same feature definitions/order and verify currentfeaturecode,data,andLPcache hashes. Train using the baseline's exact `train_sample_2021.npy` and `train_sample_2022.npy`, with identical purgedchronologicalcase IDs. No rowresampling, featurefitting or earlystopping is introduced.

Fixed XGBoost settings:binarylogistic objective,histCPU method,600trees,maxdepth6,learningrate0.05,minchildweight20,L2=4,L1=0,maxbin128,subsample0.9,colsamplebytree0.9,seed4173,threads2; all other effectiveparameters are recorded. Fit separate Haukeli_G1,Lio_G1,Hogga_G1 models for2021, then fit the same three for2022. Save sixUBJ checkpoints and rawspecialistprobabilities.

The featurebuilder parses the supplied labelCSV while constructing features, but only pre2022trainingtargets are retained. No2022heldoutlabels or probabilities enter selection. The2022validationNPZ is not loaded and2022models are not fitted until the2021selectionlock has been written.

For each candidate weightw in **[0,0.25,0.5,1]**, replace the same threecolumns jointly with `(1-w)*baseline + w*specialist`; all11othercolumns remain unchanged. Score pooledmicroAUC over all14units and168hours. Select the weight by2021only, favoring lowerw on exactties, and freeze it before evaluating2022. No Vinjeprojection or temporalsmoothing is applied in this ablation. Store rawperunitAUC and everyfullmatrixgridscore;2022grid is diagnostic, not a selection source.

Launch only after BOTH`lgb-normalized-lp-001` and`lgb-physics-large-001` manifests saycomplete, to avoid oversubscription:

```sh
.venv-ml/bin/python scripts/xgb_lp_specialists.py --run-id xgb-lp-specialists-001 > artifacts/xgb-lp-specialists-001.log 2>&1
```

The trainer refuses existingoutputdirectories, verifies allbaselineartifacts, archives itsfeature/trainingcode, logs failures and successes to`docs/experiments.md`, and records source/data/model/checkpoint/sample/prediction hashes plus environment. It does not fit test models, create a Kaggle submission, or interact with Kaggle.

Migration requires at least0.998 on BOTH chronologicalyears. PublicKaggle target remains at least0.999 with an actual verifiedtop3rank. These repeatedly reuseddevelopmentfolds do not constitute an untouchedholdout. A usefulspecialistgain belowthreshold remains experimental.

## Completed result: xgb-lp-specialists-001

Selected joint weight on2021:0.5.

| Year | Baseline micro AUC | Frozen-weight micro AUC |
|---|---:|---:|
| 2021 | 0.99437001 | 0.99453932 |
| 2022 | 0.99334854 | 0.99342383 |

Complete model/source/data provenance and every grid score:`artifacts/xgb-lp-specialists-001/manifest.json`. Checkpoint predictions and selected full matrices are saved peryear. No Kaggle score is implied.

## Replay and full fit:xgb-lp-specialists-final-001

All three2021UBJ checkpoints reproduced saved probabilities exactly(maxdifference0); recomposing the locked50%three-unit replacement reproduced the saved14-unit validationmatrix exactly. Verification:`artifacts/xgb-lp-specialists-final-001/replay_2021.json`.

Full fit uses all2922labelledcases and exactly196358sampledcase-hourrows from `lgb-normalized-lp-final-001/train_sample_final.npy`, unchanged600treeparams/seed4173/twothreads. Three finalUBJ files, raw725×168×3testprobabilities, case IDs and unitorder are in `artifacts/xgb-lp-specialists-final-001/`. These are componentoutputs, not a completeKaggleCSV; no submission was made.
