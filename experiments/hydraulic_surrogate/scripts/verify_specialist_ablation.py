"""Verify a saved scalar specialist blend without refitting or feature loading."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import sys

for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[variable] = '2'
import numpy as np
from sklearn.metrics import roc_auc_score

from lp_candidate_compare import digest, verify, ROOT


def main() -> None:
    """Check hashes, sample identity, label/unit alignment and blend arithmetic."""
    parser = argparse.ArgumentParser()
    parser.add_argument('run_id')
    args = parser.parse_args()
    source = ROOT / 'artifacts' / args.run_id
    out = ROOT / 'artifacts' / (args.run_id + '-verification')
    out.mkdir(exist_ok=False)
    manifest = json.loads((source / 'manifest.json').read_text())
    lock = json.loads((source / 'selection_2021.json').read_text())
    assert manifest['status'] == 'complete' and set(lock['folds']) == {'2021'}
    assert lock['selected_weight'] == manifest['selected_weight']
    verify(source, manifest['artifacts'])
    verify(source / 'source', manifest['source'])
    verify(ROOT, manifest['data'])
    verify(ROOT, manifest['inputs'])
    report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': shlex.join([sys.executable, *sys.argv]),
              'source_manifest_sha256': digest(source / 'manifest.json'),
              'selection_lock_sha256': digest(source / 'selection_2021.json'),
              'checkpoint_replay': 'not performed; this check verifies saved predictions and checkpoint hashes',
              'selected_weight': manifest['selected_weight'], 'folds': {}}
    base = ROOT / 'artifacts/lgb-normalized-lp-001'
    for year in [2021, 2022]:
        assert digest(source / f'train_sample_{year}.npy') == digest(base / f'train_sample_{year}.npy')
        with np.load(source / f'validation_{year}.npz', allow_pickle=False) as z, np.load(base / f'validation_{year}.npz', allow_pickle=False) as b:
            assert np.array_equal(z['case_ids'], b['case_ids'])
            assert np.array_equal(z['units'], b['units'])
            assert np.array_equal(z['labels'], b['labels'])
            units = z['units'].tolist()
            indices = [units.index(unit) for unit in z['specialist_units']]
            predicted = z['specialist_predictions']
            assert np.isfinite(predicted).all() and predicted.min() >= 0 and predicted.max() <= 1
            expected = b['predictions'].astype(np.float64)
            weight = manifest['selected_weight']
            expected[:, indices] = (1 - weight) * expected[:, indices] + weight * predicted
            assert np.array_equal(expected, z['predictions'])
            score = float(roc_auc_score(z['labels'].ravel(), expected.ravel()))
            assert score == manifest['folds'][str(year)]['selected_micro_auc']
            report['folds'][str(year)] = {'bitwise_blend_match': True, 'sample_identity': True,
                                         'labels_ids_units_match': True, 'recomputed_micro_auc': score,
                                         'prediction_sha256': digest(source / f'validation_{year}.npz')}
    copied = out / 'verify_specialist_ablation.py'
    shutil.copy2(__file__, copied)
    report['verifier_sha256'] = digest(copied)
    report['status'] = 'complete'
    (out / 'manifest.json').write_text(json.dumps(report, indent=2))
    with (ROOT / 'docs/experiments.md').open('a') as handle:
        handle.write(f'\n## {args.run_id}-verification — {report["timestamp_utc"]}\n\n'
                     f'Command: `{report["command"]}`. Complete: artifact/source/data/checkpoint hashes, '
                     'exact baseline sample identity, labels/IDs/units, bitwise specialist blend arithmetic '
                     'and recomputed two-fold micro AUC. No checkpoint replay, fitting or feature loading. '
                     f'Proof: `artifacts/{args.run_id}-verification/manifest.json`.\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
