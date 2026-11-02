"""Independently verify LP-context checkpoint replay and the frozen blend export."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shlex
import shutil
import sys
import time
import traceback

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sintef_project.postprocess import project_vinje


def sha256(path: Path) -> str:
    """Hash a file without loading large checkpoints into memory."""
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    """Compare independent checkpoint inference and reconstruct accepted weights."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='verify-lp-context-blend-001')
    args = parser.parse_args()
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    started = time.time()
    record = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': shlex.join([sys.executable, *sys.argv]),
              'python': sys.version, 'platform': platform.platform(),
              'numpy': np.__version__, 'pandas': pd.__version__, 'inputs': {},
              'training': 'none; integrity and deterministic reconstruction only',
              'kaggle': None}
    try:
        runs = ['lgb-normalized-lp-final-001', 'lgb-normalized-lp-context-final-001',
                'replay-lp-context-final-001', 'submission-lp-context-blend-001']
        manifests = {}
        for run in runs:
            directory = ROOT / 'artifacts' / run
            manifest = json.loads((directory / 'manifest.json').read_text())
            assert manifest['status'] == 'complete', run
            manifests[run] = manifest
            record['inputs'][f'artifacts/{run}/manifest.json'] = sha256(directory / 'manifest.json')
            for name, expected in manifest['artifacts'].items():
                assert sha256(directory / name) == expected, (run, name)
            for name, expected in manifest.get('data', {}).items():
                assert sha256(ROOT / name) == expected, name
        base, context, replay, exported = [ROOT / 'artifacts' / run for run in runs]
        original = np.load(context / 'test_predictions.npy')
        repeated = np.load(replay / 'test_predictions.npy')
        assert original.shape == repeated.shape == (725 * 168, 14)
        assert original.dtype == repeated.dtype == np.float32
        assert np.array_equal(original, repeated)
        raw_sha = sha256(context / 'submission.csv')
        assert raw_sha == sha256(replay / 'submission.csv')
        replay_manifest = manifests[runs[2]]
        assert replay_manifest['arguments']['inference_from'] == 'artifacts/lgb-normalized-lp-context-final-001'
        assert replay_manifest['features'] == manifests[runs[1]]['features']
        assert replay_manifest['data'] == manifests[runs[1]]['data']
        record['replay'] = {'bitwise_equal': True, 'shape': list(original.shape),
                            'dtype': str(original.dtype), 'max_abs_error': 0.,
                            'raw_csv_sha256': raw_sha,
                            'command': replay_manifest['command']}
        selection = ROOT / 'artifacts/lp-candidate-comparison-001'
        locked = json.loads((selection / 'selection_2021.json').read_text())
        chosen = json.loads((selection / 'manifest.json').read_text())
        assert set(locked['folds']) == {'2021'}
        assert locked['selected_candidate'] == chosen['accepted_candidate'] == 'context75_base25'
        units = chosen['folds']['2021']['units']
        template = pd.read_csv(ROOT / 'sintef_project/sample_submission.csv')
        columns = ['Run No'] + [f'result_committed_{unit}_t{hour}' for unit in units for hour in range(168)]
        assert template.columns.tolist() == columns
        # The exporter promotes source probabilities to float64 before weighting.
        rebuilt = np.zeros(original.shape, dtype=np.float64)
        rebuilt += .25 * np.load(base / 'test_predictions.npy').astype(np.float64)
        rebuilt += .75 * original.astype(np.float64)
        rebuilt = project_vinje(rebuilt, units)
        saved = np.load(exported / 'test_predictions.npy')
        assert np.array_equal(rebuilt, saved)
        frame = pd.read_csv(exported / 'submission.csv', float_precision='round_trip')
        assert frame.shape == template.shape == (725, 2353)
        assert frame.columns.tolist() == columns
        assert frame['Run No'].equals(template['Run No'])
        assert not frame['Run No'].duplicated().any()
        values = frame.iloc[:, 1:].to_numpy()
        assert np.isfinite(values).all() and values.min() >= 0 and values.max() <= 1
        wide = rebuilt.reshape(725, 168, 14).transpose(0, 2, 1).reshape(725, -1)
        error = float(np.max(np.abs(values - wide)))
        assert error <= 5.1e-10
        export_manifest = manifests[runs[3]]
        assert export_manifest['weights'] == [.25, .75]
        assert export_manifest['source_order'] == ['lgb-normalized-lp-001', 'lgb-normalized-lp-context-001']
        assert sha256(exported / 'submission.csv') == export_manifest['submission_sha256']
        record['blend'] = {'bitwise_equal': True, 'shape': list(frame.shape),
                           'weights_base_context': [.25, .75], 'unit_order': units,
                           'postprocessing': 'Vinje projection, no smoothing',
                           'csv_readback_max_abs_error': error,
                           'probability_range': [float(values.min()), float(values.max())],
                           'submission_sha256': export_manifest['submission_sha256'],
                           'validation_micro_auc': export_manifest['validation_micro_auc'],
                           'selection_manifest_sha256': sha256(selection / 'manifest.json'),
                           'selection_lock_sha256': sha256(selection / 'selection_2021.json')}
        record['status'] = 'complete'
        print(json.dumps({'status': 'complete', 'replay': record['replay'], 'blend': record['blend']}), flush=True)
    except BaseException:
        record['status'] = 'failed'
        record['error'] = traceback.format_exc()
        raise
    finally:
        for name in ['scripts/verify_lp_context_export.py', 'sintef_project/postprocess.py']:
            target = out / 'source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        record['runtime_seconds'] = time.time() - started
        record['artifacts'] = {str(p.relative_to(out)): sha256(p) for p in out.rglob('*') if p.is_file()}
        (out / 'manifest.json').write_text(json.dumps(record, indent=2))
        with (ROOT / 'docs/experiments.md').open('a') as handle:
            handle.write(f'\n## {args.run_id} — {record["timestamp_utc"]}\n\nStatus: {record["status"]}. '
                         f'Command: `{record["command"]}`. Independent LP-context checkpoint replay '
                         'compared bitwise with the original full-fit output; raw CSV hashes compared. '
                         'The locked 75% context / 25% base export was reconstructed from saved arrays, '
                         'with fixed Vinje projection, exact schema/ID checks and CSV readback. '
                         f'Proof: `artifacts/{args.run_id}/manifest.json`. No fit or Kaggle submission. '
                         'Chronological folds have been reused for research; both scores remain below 0.998.\n')


if __name__ == '__main__':
    main()
