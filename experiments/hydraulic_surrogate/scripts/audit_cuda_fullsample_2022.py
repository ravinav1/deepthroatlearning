"""Independently confirm fullsample608 at its immutable2021 weight on2022."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--run-id', required=True)
parser.add_argument('--import-manifest-sha256', required=True)
parser.add_argument('--selection-manifest-sha256', required=True)
args = parser.parse_args()
assert Path(args.run_id).name == args.run_id and args.run_id not in ['', '.', '..']
OUT = ROOT / 'artifacts' / args.run_id
OUT.mkdir(exist_ok=False)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def project(values, units):
    result = values.copy()
    indices = [units.index(f'Vinje_G{i}') for i in [1, 2, 3]]
    x = values[:, indices]
    for i in range(3):
        result[:, indices[i]] = np.min(np.stack([
            np.max(np.stack([x[:, a:b + 1].mean(axis=1) for b in range(i, 3)]), axis=0)
            for a in range(i + 1)]), axis=0)
    return result


start = time.monotonic()
protocol_path = ROOT / 'docs/cuda_xgb608_fullsample_protocol.json'
assert sha(protocol_path) == 'd5add5941279d980f156015b4b5ca70da2af4b6d8b99c4841050569b02b7255e'
p = json.loads(protocol_path.read_text())
selection_dir = ROOT / 'artifacts/cuda-xgb608-fullsample-select-2022-001'
import_dir = ROOT / 'artifacts/verify-cuda-xgb608-fullsample-2022-return-001'
gpu_dir = import_dir / 'download/cuda-xgb608-fullsample-2022-001'
lock_path = ROOT / 'artifacts/cuda-xgb608-fullsample-select-2021-001/selection_2021.json'
lock_sha = 'd5fe5b3c2a6c64b0f9a35405411b629aff1698906c876e2a7f99fb73fc4d5e83'
assert sha(lock_path) == lock_sha
assert sha(import_dir / 'manifest.json') == args.import_manifest_sha256
assert sha(selection_dir / 'manifest.json') == args.selection_manifest_sha256
lock = json.loads(lock_path.read_text())
selection = json.loads((selection_dir / 'manifest.json').read_text())
imported = json.loads((import_dir / 'manifest.json').read_text())
gpu = json.loads((gpu_dir / 'manifest.json').read_text())
for directory, manifest in [(selection_dir, selection), (import_dir, imported), (gpu_dir, gpu)]:
    assert manifest['status'] == 'complete'
    for name, value in manifest['artifacts'].items():
        assert sha(directory / name) == value, name
assert sha(gpu_dir / 'manifest.json') == imported['hosted_manifest_sha256']
assert gpu['selection_lock_sha256'] == lock_sha == selection['selection_lock_sha256']
assert sha(gpu_dir / 'validation_2022.npz') == selection['gpu_validation_sha256']
assert gpu['protocol_sha256'] == sha(protocol_path) == lock['training_protocol_sha256']
assert gpu['train_ids'] == p['folds']['2022']['train_ids']
assert gpu['validation_ids'] == p['folds']['2022']['validation_ids']
assert gpu['features'] == p['feature_columns'] and gpu['units'] == p['units']
assert gpu['completed_units'] == 14 and gpu['completed_rounds_per_unit'] == 600
assert gpu['sample_sha256'] == p['sample_sha256']['2022']
for unit, model in gpu['models'].items():
    assert model['checkpoint_replay_max_error'] == 0 and model['replay_rows'] == 365 * 168
    native = json.loads((gpu_dir / f'config_2022_{unit}.json').read_text())['learner']
    assert native['generic_param']['device'] == 'cuda:1'
    assert int(native['gradient_booster']['gbtree_model_param']['num_trees']) == 600
    tree = native['gradient_booster']['tree_train_param']
    assert int(tree['max_depth']) == 6 and abs(float(tree['eta']) - .05) < 1e-7
with np.load(gpu_dir / 'validation_2022.npz') as z:
    raw, y, ids, units = z['predictions'].astype(np.float64), z['labels'], z['case_ids'], z['units'].tolist()
assert raw.shape == y.shape == (61320, 14) and np.isfinite(raw).all() and raw.min() >= 0 and raw.max() <= 1
values = {}
for kind in ['anchor', 'control']:
    item = p[kind]['2022']
    assert sha(ROOT / item['manifest']) == item['manifest_sha256']
    assert sha(ROOT / item['prediction']) == item['prediction_sha256']
    with np.load(ROOT / item['prediction']) as z:
        values[kind] = z['predictions'].astype(np.float64)
        np.testing.assert_array_equal(y, z['labels']); np.testing.assert_array_equal(ids, z['case_ids'])
        assert units == z['units'].tolist()
metadata = pd.read_csv(ROOT / 'sintef_project/data/kernel/Unit_commitment_decisions.csv', usecols=['Run No', 'starttime'])
chosen = metadata['Run No'].isin(ids).to_numpy()
np.testing.assert_array_equal(metadata.loc[chosen, 'Run No'], ids)
dates = pd.to_datetime(metadata.loc[chosen, 'starttime'], utc=True)
assert dates.min() == pd.Timestamp('2022-01-01', tz='UTC') and dates.max() == pd.Timestamp('2022-12-31', tz='UTC')
all_dates = pd.to_datetime(metadata['starttime'], utc=True)
train_cases = np.flatnonzero(all_dates < pd.Timestamp('2022-01-01', tz='UTC') - pd.Timedelta(days=6))
assert metadata.iloc[train_cases]['Run No'].tolist() == gpu['train_ids']
assert all_dates.iloc[train_cases].max() + pd.Timedelta(hours=167) < dates.min()
full_rows = (train_cases[:, None] * 168 + np.arange(168)).ravel()
sample = np.load(gpu_dir / 'train_sample_2022.npy', allow_pickle=False)
assert sample.dtype == full_rows.dtype and sample.tobytes() == full_rows.tobytes()
assert len(sample) == gpu['sampled_training_rows'] == p['sample_rows']['2022'] == 428568
assert len(np.unique(sample)) == len(sample)
columns = [f'result_committed_{unit}_t{h}' for unit in units for h in range(168)]
target = pd.read_csv(ROOT / 'sintef_project/data/kernel/Unit_commitment_decisions.csv',
                     skiprows=(np.flatnonzero(~chosen) + 1).tolist(), usecols=['Run No', *columns],
                     dtype={column: 'uint8' for column in columns})
np.testing.assert_array_equal(target[columns].to_numpy().reshape(365, 14, 168).transpose(0, 2, 1).reshape(-1, 14), y)
scores = {str(w): float(roc_auc_score(y.ravel(), project((1 - w) * values['anchor'] + w * raw, units).ravel()))
          for w in [0., lock['selected_weight']]}
assert set(scores) == {'0.0', str(lock['selected_weight'])}
assert set(scores) == set(selection['scores'])
for weight in scores:
    np.testing.assert_allclose(scores[weight], selection['scores'][weight], atol=1e-13, rtol=0)
selected = lock['selected_weight']
assert selected == lock['selected_weight'] == .25
control = float(roc_auc_score(y.ravel(), project((1 - selected) * values['anchor'] + selected * values['control'], units).ravel()))
np.testing.assert_allclose(control, selection['control_micro_auc'], atol=1e-13, rtol=0)
eligible = selected > 0 and scores[str(selected)] > max(scores['0.0'], control)
assert eligible == selection['accepted'] and lock['eligible_for_2022']
expected = project((1 - selected) * values['anchor'] + selected * raw, units)
with np.load(selection_dir / 'selected_validation_2022.npz') as z:
    np.testing.assert_array_equal(z['labels'], y)
    np.testing.assert_array_equal(z['case_ids'], ids)
    np.testing.assert_allclose(z['predictions'], expected, atol=5e-16, rtol=0)
report = {'run_id': OUT.name, 'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
          'accepted': bool(eligible), 'selection_lock_sha256': lock_sha,
          'protocol_sha256': sha(protocol_path), 'selection_manifest_sha256': sha(selection_dir / 'manifest.json'),
          'gpu_manifest_sha256': sha(gpu_dir / 'manifest.json'), 'import_manifest_sha256': sha(import_dir / 'manifest.json'),
          'selected_weight': selected, 'scores_2022': scores, 'control_micro_auc': control,
          'audit_source_sha256': sha(Path(__file__)), 'runtime_seconds': time.monotonic() - start,
          'checks': ['All import/selector/GPU artifact hashes exact', 'NativeGPU1/depth6/600trees/.05 settings and exact428568fullpurgedtrainingrows/feature/case provenance',
                     'Original2022 targets and all array schemas/ranges exact', 'Independent min-max isotonic14-unit blend reconstruction',
                     'Only fixed2021weight.25 and zero evaluated; matchedcontrol and strict confirmation exact'],
          'scope': 'Independent saved-array audit; does not repeat14 hostedcheckpoint inference, whose original exact replays and artifact closure were verified. Only2022 confirmation labels decoded; no training, alternate weight or remote action.',
          'environment': {'python': sys.version, 'platform': platform.platform()},
          'command': [sys.executable, *sys.argv]}
(OUT / 'audit.py').write_bytes(Path(__file__).read_bytes())
(OUT / 'manifest.json').write_text(json.dumps(report, indent=2))
print(json.dumps({'status': 'passed', 'accepted': bool(eligible), 'manifest_sha256': sha(OUT / 'manifest.json'), 'runtime': report['runtime_seconds']}))
