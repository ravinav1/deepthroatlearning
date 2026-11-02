"""Freeze the canonical115 early-2020 comparison without building or fitting."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import screen_canonical_pooled as screen
from sintef_project.canonical_unit_features import feature_specification


def main():
    original = json.loads((ROOT / screen.BASE_PATH).read_text())
    assert screen.base.sha(ROOT / screen.BASE_PATH) == screen.BASE_SHA
    metadata = screen.pd.read_csv(ROOT / 'sintef_project/data/kernel/Unit_commitment_decisions.csv', usecols=['Run No', 'starttime'])
    train, validation, sample = screen.early_splits(metadata)
    sample_path = 'artifacts/full14-early-oof-001/train_sample_2020.npy'
    assert screen.base.sha(ROOT / sample_path) == '422aeceb10ef5890bf3373af41296450845754f4f5b96cd727dc5bf7a68fe20a'
    saved = screen.np.load(ROOT / sample_path, allow_pickle=False)
    assert saved.dtype == sample.dtype and saved.tobytes() == sample.tobytes()
    sources = ['scripts/screen_canonical_pooled.py', 'scripts/prepare_canonical_pooled_screen.py',
               'tests/test_canonical_pooled_screen.py', 'sintef_project/canonical_unit_features.py',
               'tests/test_canonical_unit_features.py', 'docs/canonical_pooled_screen.md',
               'scripts/train_lgb_shop664.py', 'scripts/cuda_xgb664_loader.py', 'scripts/cuda_xgb608_loader.py',
               'scripts/train_cuda_tcn.py', 'sintef_project/postprocess.py']
    prerequisite_paths = ['artifacts/canonical-unit-adapter-independent-audit-001/manifest.json',
                          'artifacts/full14-early-oof-result-independent-audit-001/manifest.json',
                          'artifacts/cuda-xgb664-package-feature-independent-audit-001/manifest.json']
    prerequisites = [{'path': path, 'sha256': screen.base.sha(ROOT / path)} for path in prerequisite_paths]
    for record in prerequisites:
        assert json.loads((ROOT / record['path']).read_text())['status'] == 'passed'
    oof_manifest = 'artifacts/full14-early-oof-001/manifest.json'
    raw_anchor = 'artifacts/full14-early-oof-001/oof_2020.npz'
    projected_anchor = 'artifacts/pooled-pairwise-residual-screen-001/screen_identity.npz'
    inputs = [screen.BASE_PATH, original['loader_protocol'], sample_path, oof_manifest, raw_anchor, projected_anchor,
              'docs/full14_early_oof_protocol.json', 'artifacts/canonical-pooled-timing-001/manifest.json',
              'artifacts/pooled-pairwise-residual-screen-001/manifest.json', *prerequisite_paths]
    protocol = {'version': screen.VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'base_protocol': screen.BASE_PATH, 'base_protocol_sha256': screen.BASE_SHA,
        'source': {name: screen.base.sha(ROOT / name) for name in sources},
        'inputs': {name: screen.base.sha(ROOT / name) for name in inputs},
        'environment': screen.base.environment(), 'prerequisites': prerequisites,
        'prefix_columns': original['feature_columns'], 'units': original['units'],
        'canonical_specification': feature_specification(original['feature_columns'], original['units']),
        'train_ids': metadata.iloc[train]['Run No'].tolist(), 'screen_ids': metadata.iloc[validation]['Run No'].tolist(),
        'train_dates': [metadata.iloc[train[0]].starttime, metadata.iloc[train[-1]].starttime],
        'screen_dates': [metadata.iloc[validation[0]].starttime, metadata.iloc[validation[-1]].starttime],
        'purge_days': 6, 'maximum_training_target_year': 2019,
        'sample': {'path': sample_path, 'sha256': screen.base.sha(ROOT / sample_path), 'hours': 122304, 'expanded_rows': 1712256},
        'oof_manifest': oof_manifest, 'raw_anchor': raw_anchor, 'projected_anchor': projected_anchor,
        'control_parameters': original['model_parameters'],
        'pooled_parameters': {**original['model_parameters'], 'n_estimators': 1400},
        'weights': screen.WEIGHTS, 'rounds': screen.ROUNDS,
        'selection': 'Highest full-year AUC; ties lower rounds then lower weight; selected positive blend must strictly beat anchor and matched-weight canonical control for full year and each calendar half; no rescue.',
        'fit_order': '14 controls in frozen unit order, then one 1400-round pooled trajectory; no early stopping.',
        'resources': {'entry_free_percent': 40., 'ongoing_free_percent': 20.,
            'observed_peak_limit_bytes': 8 * 1024 ** 3, 'fit_memory_target_bytes': 6 * 1024 ** 3,
            'observed_wall_limit_seconds': 35 * 60, 'model_threads': 2,
            'checks': 'Between phases/controls and every25 native rounds; observational, not OS-enforced.'},
        'screen_labels_after_all_models_and_raw_prediction_seal': True,
        'reused_development_screen': True, 'future_year_or_final_training_authorized': False}
    path = ROOT / 'docs/canonical_pooled_screen_protocol.json'
    with path.open('x') as stream:
        json.dump(protocol, stream, indent=2)
    print(screen.base.sha(path), flush=True)


if __name__ == '__main__':
    main()
