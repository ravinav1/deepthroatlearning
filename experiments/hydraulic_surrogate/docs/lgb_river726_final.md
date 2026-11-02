# Conditional LGB726 final recipe

Preparation only: the final protocol may be created only from actual independently accepted 2021 and locked 2022 LGB726 results. No final protocol, full-data fit or CSV is authorized by these templates alone.

The recipe reuses the original 196,358-row final sample and 350-tree LightGBM parameters, with two CPU threads and no early stopping. It uses the accepted 726-feature builder: the original staged 664 columns followed by the independently verified 62 river features from cache002. The frozen validation protocol is `c71d6a026f7ca3e802464df4be7b8874ab8592f54d38d85cd97e2b9040762b7c`. The final weight must be the actual positive 2021 lock and must pass the unchanged 2022 anchor/control gate.

The full cache is separately pinned in the final recipe, trainer report and exporter. Every checkpoint is saved and reloaded; all 121,800 test rows must reproduce exactly in float64 before float32 raw probabilities are stored. An inference-only mode reloads the original checkpoints without fitting. Entry requires at least 40% free memory and each unit requires at least 20%.

The final formula is `project_vinje((1-w)*saved_projected_SHOP664_anchor + w*raw_LGB726)` in float64. The current accepted SHOP anchor remains fixed. The exporter requires an independent all14-checkpoint raw-final audit and produces one schema-preserving primary CSV. Independent final checkpoint and actual CSV verification remain mandatory before any submission.

`scripts/prepare_lgb_river726_final.py` takes explicit actual selection-run/manifest/audit hashes and actual 2022 confirmation-run/manifest/audit hashes. It verifies them before exclusively creating `docs/lgb_river726_final_protocol.json`. It preserves all frozen 664 sources and imports their reviewed acceptance helper. `scripts/train_lgb_river726_final.py` supports metadata preflight and final/replay execution under the frozen protocol and independent approval. `scripts/export_lgb_river726_final.py` performs the strict nested export after actual raw-final acceptance.
