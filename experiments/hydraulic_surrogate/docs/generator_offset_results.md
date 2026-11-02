# Generator offset screen result

The frozen 2019 gate rejected the bounded pairwise offset recipe. It improved full-year pooled AUC but lost in the second half, so the branch stops before 2020. No coefficient, regularization, pair pool or gate will be tuned from this result.

| Recipe | Full 2019 AUC | First half | Second half |
| --- | ---: | ---: | ---: |
| Projected identity | 0.9929505214135640 | 0.9910258924725790 | 0.9936564088662517 |
| Pairwise offsets | 0.9930746746206733 | 0.9915136457728021 | 0.9934742735805820 |
| Balanced BCE offsets | 0.9927988094401643 | 0.9910306844471669 | 0.9933351277285409 |

The pairwise gains were +0.000124153207 overall, +0.000487753300 in the first half, and −0.000182135286 in the second half. This is evidence that small generator shifts can move pooled ranking, but a constant shift learned from 2018 did not transfer consistently across 2019. Every unit retained exactly the same ranks, ties and AUC; Vinje order was preserved. Both optimizations converged, both model/pair-sampling replays matched exactly, and all prediction files were sealed before 2019 case targets were parsed. The fit used359 purged2018 cases. Runtime was4.417 seconds and peakRSS378,814,464 bytes.

Interpretation is limited to the frozen single608 LightGBM OOF base. The pairwise objective is mean pair loss (1/M), whereas matching balanced meanBCE zero-margin gradient and curvature normalization would require0.5/M. Thus identical lambda0.01 does not give identical effective regularization between objectives; this is a comparison of the two frozen recipes, not a pure causal isolation of loss choice. Earlier failed ridge calibration already represented the same offset ranking family.

Protocol: `docs/generator_offset_protocol.json`, SHA `452d1dff51081531263efbcb13df8146c15e54d9b66bb877d4b5c7e13bf11b24`. Independent source approval: `artifacts/generator-offset-independent-audit-001/manifest.json`, SHA `fde8531dc7fed16f777c317e62fb616076a03894f03dcbef0f457f90c4328254`. Result: `artifacts/generator-offset-2019-001/manifest.json`, SHA `e204561de8cecfdb1cf654f236cffd1a69cc14c5452ef205b2e86fad51ea226c`. Independent result review is pending; no public submission was produced.

Independent result review passed and confirmed rejection: `artifacts/generator-offset-2019-independent-audit-001/manifest.json`, SHA `ad1c4a99c23070a8de6e3b62fdea7fc349ed4d9c464a074c36a4ad311ed95445`. The reviewer reconstructed the1M pair draw, both objectives and box KKT conditions without rerunning the optimizer; independently checked projection, logistic transforms, ranks, full/half rank-sum AUC and lock timing. No2020 targets were read.
