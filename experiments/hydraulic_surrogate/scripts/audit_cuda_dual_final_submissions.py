"""Independently reconstruct accepted final blends and predeclared standalone CSVs."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sintef_project.postprocess import project_vinje


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def independent_projection(values, units):
    result = values.copy()
    indices = [units.index(f'Vinje_G{i}') for i in [1, 2, 3]]
    x = values[:, indices]
    for i in range(3):
        result[:, indices[i]] = np.min(np.stack([
            np.max(np.stack([x[:, a:b+1].mean(axis=1) for b in range(i, 3)]), axis=0)
            for a in range(i+1)]), axis=0)
    return result


def audit_one(name, primary, primary_sha, diagnostic, diagnostic_sha, protocol_path, protocol_sha):
    assert sha(primary/'manifest.json') == primary_sha
    assert sha(diagnostic/'manifest.json') == diagnostic_sha
    assert sha(protocol_path) == protocol_sha
    p = json.loads(protocol_path.read_text())
    m = json.loads((primary/'manifest.json').read_text())
    assert m['status'] == 'complete' and m['protocol_sha256'] == protocol_sha
    assert m['final_full_test_replay_bitwise'] and m['anchor_source_reconstruction_bitwise']
    assert sha(ROOT/'sintef_project/postprocess.py') == p['postprocess_sha256']
    for folder in {primary, diagnostic}:
        record = json.loads((folder/'manifest.json').read_text())
        assert record['status'] == 'complete'
        for member, expected in record['artifacts'].items():
            assert sha(folder/member) == expected, member
    matches = [path for path in (primary/'download').rglob('manifest.json') if sha(path) == m['hosted_manifest_sha256']]
    assert len(matches) == 1
    remote = matches[0].parent
    hosted = json.loads(matches[0].read_text())
    assert hosted['status'] == 'complete' and hosted['mode'] == 'final'
    assert hosted['completed_units'] == 14 and hosted['completed_rounds_per_unit'] == p['model_parameters']['n_estimators']
    assert hosted['model_parameters_requested'] == p['model_parameters']
    for member, expected in hosted['artifacts'].items(): assert sha(remote/member) == expected, member
    with np.load(remote/'test_predictions.npz', allow_pickle=False) as z:
        raw, ids, units = z['predictions'], z['case_ids'], z['units'].tolist()
        assert set(z.files) == {'predictions', 'case_ids', 'units'}
    replay = np.load(remote/'replay_test_predictions.npy', allow_pickle=False)
    assert raw.shape == replay.shape == (121800, 14)
    assert raw.dtype == replay.dtype == np.float32 and raw.tobytes() == replay.tobytes()
    assert ids.tolist() == p['test_case_ids'] and units == p['units']
    assert np.isfinite(raw).all() and 0 <= raw.min() <= raw.max() <= 1
    anchor_info = p['local_anchor']; anchor_dir = ROOT/anchor_info['run_path']
    assert sha(anchor_dir/'manifest.json') == anchor_info['manifest_sha256']
    assert sha(anchor_dir/'test_predictions.npy') == anchor_info['predictions_sha256']
    anchor = np.load(anchor_dir/'test_predictions.npy', allow_pickle=False)
    assert anchor.shape == raw.shape
    np.testing.assert_array_equal(project_vinje(anchor, units), anchor)
    template = pd.read_csv(ROOT/'sintef_project/sample_submission.csv')
    columns = ['Run No']+[f'result_committed_{unit}_t{hour}' for unit in units for hour in range(168)]
    assert template.columns.tolist() == columns
    assert len(set(ids.tolist())) == len(ids) == 725 and set(ids.tolist()) == set(template['Run No'].tolist())
    results = {}
    for label, folder, values, array_name, csv_name in [
        ('primary', primary, .75*anchor.astype(np.float64)+.25*raw.astype(np.float64), 'test_predictions.npy', 'submission.csv'),
        ('standalone', diagnostic, raw.astype(np.float64), 'standalone_test_predictions.npy', 'submission_standalone_projected.csv')]:
        expected = project_vinje(values, units)
        independent = independent_projection(values, units)
        np.testing.assert_allclose(expected, independent, atol=5e-16, rtol=0)
        saved = np.load(folder/array_name, allow_pickle=False)
        assert saved.dtype == expected.dtype == np.float64 and saved.shape == expected.shape
        assert saved.tobytes() == expected.tobytes()
        assert np.isfinite(expected).all() and 0 <= expected.min() <= expected.max() <= 1
        wide = expected.reshape(725, 168, 14).transpose(0, 2, 1).reshape(725, -1)
        frame = pd.DataFrame(wide, index=pd.Index(ids, name='Run No'), columns=columns[1:])
        frame = frame.loc[template['Run No']].reset_index()
        csv = folder/csv_name
        assert csv.read_bytes() == frame.to_csv(index=False, float_format='%.9g').encode()
        back = pd.read_csv(csv)
        assert back.shape == (725, 2353) and back.columns.tolist() == columns
        assert back['Run No'].equals(template['Run No'])
        error = float(np.max(np.abs(back.iloc[:, 1:].to_numpy()-frame.iloc[:, 1:].to_numpy())))
        assert error <= 5.1e-10
        results[label] = {'submission_path': str(csv.relative_to(ROOT)), 'submission_sha256': sha(csv),
                          'predictions_sha256': sha(folder/array_name), 'exact_array_bytes': True,
                          'exact_csv_bytes': True, 'independent_isotonic_max_error': float(np.max(np.abs(expected-independent))),
                          'csv_readback_max_error': error, 'shape': list(back.shape),
                          'probability_range': [float(expected.min()), float(expected.max())]}
    return {'primary_manifest_sha256': primary_sha, 'diagnostic_manifest_sha256': diagnostic_sha,
            'hosted_manifest_sha256': m['hosted_manifest_sha256'], 'protocol_sha256': protocol_sha,
            'anchor_predictions_sha256': anchor_info['predictions_sha256'], 'results': results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--shop-manifest-sha256')
    parser.add_argument('--capacity-manifest-sha256')
    parser.add_argument('--capacity-diagnostic-manifest-sha256')
    args = parser.parse_args()
    assert Path(args.run_id).name == args.run_id and args.run_id not in ['', '.', '..']
    assert args.shop_manifest_sha256 or args.capacity_manifest_sha256
    assert bool(args.capacity_manifest_sha256) == bool(args.capacity_diagnostic_manifest_sha256)
    out = ROOT/'artifacts'/args.run_id; out.mkdir(exist_ok=False)
    start = time.monotonic()
    report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
              'command': [sys.executable, *sys.argv], 'training': False, 'kaggle_submission': None,
              'environment': {'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__, 'pandas': pd.__version__},
              'scope': 'Independent actual array/ID/schema/range/CSV byte reconstruction and generic isotonic crosscheck; no new fit or local checkpoint inference. Hosted full replay and strict import closure verified.'}
    try:
        results = {}
        if args.shop_manifest_sha256:
            folder = ROOT/'artifacts/submission-cuda-xgb664-anchor-blend-001'
            results['shop664'] = audit_one('shop664', folder, args.shop_manifest_sha256, folder, args.shop_manifest_sha256,
                ROOT/'docs/cuda_xgb664_final_protocol.json', 'e571c6607bd797f65c76db5d7617dec7b031774a41af44e65283ccf1ad964ab7')
        if args.capacity_manifest_sha256:
            results['capacity608'] = audit_one('capacity608', ROOT/'artifacts/submission-cuda-xgb608-capacity-anchor-blend-002',
                args.capacity_manifest_sha256, ROOT/'artifacts/submission-cuda-xgb608-capacity-standalone-001', args.capacity_diagnostic_manifest_sha256,
                ROOT/'docs/cuda_xgb608_capacity_final_protocol.json', '840879d339732b14349d5355a886ef2e1b8e31101aa482ec8472df56a7be20d3')
        report.update(status='passed', results=results)
    except BaseException:
        report.update(status='failed', error=traceback.format_exc()); raise
    finally:
        shutil.copy2(__file__, out/Path(__file__).name)
        report['runtime_seconds'] = time.monotonic()-start
        report['audit_source_sha256'] = sha(Path(__file__))
        (out/'manifest.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({'status': report['status'], 'manifest_sha256': sha(out/'manifest.json'),
                          'runtime_seconds': report['runtime_seconds'], 'results': report.get('results')}), flush=True)


if __name__ == '__main__': main()
