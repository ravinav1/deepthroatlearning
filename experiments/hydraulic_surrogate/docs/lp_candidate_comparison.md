# LP candidate comparison — declared protocol

Prepared2026-11-02 from the rootagent's fixed eight-candidate instruction. At metadata inspection, temporalcontext training was already complete andLPv2 was still running. This protocol was written before this comparison reads either2022 prediction matrix or its scores. Do not describe it as predating the existence of allcontext2022 results. These years remain heavily reuseddevelopmentdata.

Sources:raw `lgb-normalized-lp-001`, `lgb-normalized-lp-context-001`, `lgb-normalized-lp-v2-001`, and the existing frozen `vinje-count-combined-001` matrix. The latter already contains50%XGB replacements forHaukeli/Lio/Hogga and50%Vinje-count replacement. It is the **fixed prior accepted local candidate** for the acceptance gate.

The eight candidates are fixed:

1. BaseLPv1.
2. LP temporalcontext.
3. LPv2.
4. 50%base +50%context for allunits.
5. 75%context +25%base for allunits.
6. Base with onlyHaukeli replaced by50%base +50%v2.
7. Base with onlyHaukeli replaced by100%v2.
8. Existing combinedspecialist/count candidate, with no new weights.

Apply the same fixed Vinje probability-order projection to every candidate, with no smoothing. Select the highest whole14-unit microAUC on2021, with exactties resolved in the above order. Write the selectionlock before loading2022 NPZ arrays. Evaluate all2022 grid entries for transparency but do not use them to select a different candidate. If the2021winner fails to improve2022 over the fixedprior, retain the priorcandidate. If the prior itself wins2021, retain it.

Require identical labels, caseIDs, unitorder, case-hourshape and finite[0,1] probabilities across allsources. Verify fullmanifest-listed artifacts, archivedsourcecode, input/data hashes and sourcecompletion before the run. Copy source manifests and this protocol into the newartifact for provenance. No fitting, test inference, sourcecode changes, fullfit exports orKaggle interactions occur. Stats use a twothreadcap.

Wait for bothcontext andv2 manifests to saycomplete before running:

```sh
.venv-ml/bin/python scripts/lp_candidate_compare.py --run-id lp-candidate-comparison-001
```

Migration still requires at least0.998 on BOTH chronologicalyears; Kaggle target remains at least0.999 with an actualverifiedtop3rank. Neither is established by this unexecuted protocol.

## Completed comparison:lp-candidate-comparison-001

| Candidate | 2021 microAUC | 2022 microAUC |
|---|---:|---:|
| base | 0.99437222 | 0.99335146 |
| context | 0.99462818 | 0.99311132 |
| v2 | 0.99439120 | 0.99336856 |
| base50_context50 | 0.99477936 | 0.99355136 |
| context75_base25 | 0.99478111 | 0.99347025 |
| base_haukeli_v2_50 | 0.99449384 | 0.99342429 |
| base_haukeli_v2_100 | 0.99443418 | 0.99336051 |
| combined_specialist_count | 0.99457740 | 0.99345820 |

2021selected:**context75_base25**. Accepted:**context75_base25**. The2021 winner also improves2022 over the fixedprior; accept this development candidate. Meets>=.998bothyears:False.

Provenance/fullgrid/selectionlock:`artifacts/lp-candidate-comparison-001/manifest.json`. `selected_validation_YYYY.npz` preserves the2021winner even if rejected; `validation_YYYY.npz` contains the acceptedcandidate. No fullfit orKaggle submission was performed.
