# LightGBM664 feature ablation results

The2021 screen selected weight0.1; unchanged2022 confirmation also passed the frozen strict anchor/control gate. Both years passed independent feature/checkpoint/target/metric audits. These gains are measured on reused chronological development folds and do not establish a leaderboard improvement.

| LGB664 weight | Projected2021 microAUC |
|---:|---:|
|0|0.9951018232145258|
|0.1|0.9951042081969695|
|0.25|0.9950904829409322|
|0.5|0.9950222097396040|

At0.1, the same-weight originalLGB608 raw control scores0.9950939207833579. The selected gain is0.0000023849824437 versus the acceptedSHOP anchor and0.0000102874136116 versus the matched control. Raw unblendedLGB664 microAUC is0.9945940614003821.

All14 fixed350-tree checkpoints replayed all validation predictions exactly in float64; raw arrays use the original float32 storage convention. The exact old146899 sampled rows, purged dates, original parameters except authorized2CPUthreads,664 feature schema and source/input hashes are retained. Runtime1137.028seconds; measured peakRSS6873006080bytes (6.40GiB). No early stopping or post-result weight search occurred.

Result: `artifacts/lgb-shop664-2021-001/manifest.json`, SHA `88faf9b155271b5ab46488031b9510d5aecbb6a9bbd1abe0c19554808b5bf413`. Actual selectedlock: `selection_2021.json`, SHA `cc4c7e21967d61bbb6c9139ff32a6da2586a8686156d520f3bccf3eaab28a1b9`. Frozen protocol SHA `882e8c3124ba834322c8f062a7544e9ca3e9512b385e8830db387c38f7c97d08`; independent preflight SHA `45af410ce0e04f9bae88d6b7f93f4512a64d14e7cf17fa280438ea4d2230c5f6`.

Independent2021 fresh-feature/all-checkpoint/target/projection/grid verification passed in13.88s, proof SHA `31e11af7408082765541ba8cf45d7c4c7bcfc2884d80a664a7d5969961d0e005`.

Locked2022 weight0.1 scored **0.9938717710733546**, versus the acceptedSHOP anchor **0.9938546734526688** and same-weight originalLGB608 control **0.9938428198540179**. The gain is0.0000170976206858 versus the anchor and0.0000289512193367 versus the control. Only0 and the locked0.1 were scored. All14 fixed350-tree models replayed exactly, using the original171427-row sample. Runtime1297.663seconds; peakRSS6454132736bytes. Completed result `artifacts/lgb-shop664-2022-001/manifest.json`, SHA `fa10113f17fe184559921fd9b248aeb77245aa11ccc89dbd1318855344642adb`.

Independent2022 verification passed: `artifacts/lgb-shop664-2022-independent-audit-001/manifest.json`, SHA `35f65d33e51cd3ee88d3b477f5b7cd3fa0e02c90ad9bfaf0bd318013e8a22d76`. It rebuilt664features, replayed all14 checkpoints and checked originaltargets/sample/nativeparameters and the unchanged0.1 gate.

The conditional final protocol is sealed at `docs/lgb_shop664_final_protocol.json`, SHA `5513e67123a968ded3790ec6bcda898d50d3682685fa7f62020ea25108e322ad`. Exact original196358-row finalsample and reviewed350-tree/2thread recipe are pinned. Metadata/source/sample preflight passed, SHA `6403e747ce247ebc33e1bf5ecffc3ee43e3b7dfb636263ab6aa400226082faf9`; independent finalprotocol audit also passed, SHA `191f57b14b4bf85d4b7f54fb44806dd3131dfe28f472e9d17991fb644272897f`.

Authorized final training completed in `artifacts/lgb-shop664-final-001`; manifest SHA `2d3e2f85e40a7bbf8d849664a063a09a61eb30306746018935689754a6295ada`. All14 fixed350-tree checkpoints replayed every121800test-row probability exactly in float64; the savedfloat32 raw and replay arrays match. Runtime1481.774seconds; peakRSS5917081600bytes. Session70549 exited0 and Python PID19927 is gone. Persistent log `artifacts/lgb-shop664-final-001.log`; launch observation SHA `f63422683824d784eb648dc11713d1defd6370a304d90c21aef41216fad526ee`.

Independent full-test feature/checkpoint replay passed: `artifacts/lgb-shop664-final-independent-audit-001/manifest.json`, SHA `b63b5872411fa59eaea6c7c6cbc132803b6ea3ae445c7f871b3df67875d976b8`. All14 models were replayed on fresh664features for every121800testrow, with originalsample/nativeparameters/acceptance closure verified.

The primary submission is ready at `artifacts/submission-lgb-shop664-anchor-blend-001/submission.csv`, SHA `1c1d173816127997e711d54c8269406830666af66b6f4eb6d24f4f227b70d496`. Its formula is `project_vinje(0.9*saved_projected_SHOP664_anchor+0.1*raw_LGB664)` in float64. Export manifest SHA `be4ec369caf7850c1259aee81b3344a738d62c51b9c547382fed9daa1cb6247c`. Independent exactCSV/array/schema/ID reconstruction passed: `artifacts/submission-lgb-shop664-independent-audit-001/manifest.json`, SHA `529c5b4e7a8bf1106882de2e0594c54de2d016ec979e5e69e84a723dbcfc3c99`.

This candidate has no observed Kaggle submission or public score yet. The last confirmed public best remains the earlierSHOP664 blend at0.99651/rank18.
