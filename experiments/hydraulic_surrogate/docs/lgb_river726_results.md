# LGB726 river-feature validation

The fixed 62 river features passed the actual 2021 learned test against the original SHOP664 anchor and a same-weight raw LGB664 control. The recipe and anchor remain frozen despite the later public improvement from LGB664.

| 2021 weight | Micro AUC |
|---|---:|
| 0 | 0.9951018232145258 |
| 0.1 | 0.9951224345724307 |
| 0.25 | 0.9951306250691139 |
| 0.5 | 0.9950936546926039 |

The selected 0.25 blend beats the same-weight LGB664 control, 0.9950904829409322. The independent audit rebuilt the full 726 features, checked the unchanged 664 prefix, and replayed all 14 checkpoints. Its manifest is `artifacts/lgb-river726-2021-independent-audit-001/manifest.json`, SHA `571d4f5f9e8d96fabe075fbdbdff6ef21c9ca5e9e5d133afdd6581ad412ae3c0`.

The unchanged 2022 confirmation completed under the positive immutable 0.25 lock in 1,435.604 seconds; session 64428 exited 0. It scored **0.9938778756239264**, above the frozen SHOP anchor (0.9938546734526688) but below its matched LGB664 control (0.9938790447855327). The producer therefore rejects the candidate under the predeclared dual-control rule. Actual manifest: `artifacts/lgb-river726-2022-001/manifest.json`, SHA `61d82a2cfde93d0b5e9d117f2cd81617c1d2065122e23d54b5f52dcbb669e2bc`. Independent full replay/score rejection verification is pending the corrected audit source review. No final protocol, final fit, export or LGB726 submission is authorized. The approved conditional final templates remain preserved and unexecuted.
