"""Independently verify the frozen capacity final code, sample and anchor closure."""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'artifacts/cuda-xgb608-capacity-final-independent-audit-001'
OUT.mkdir(exist_ok=False)
START = time.monotonic()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def projection(values, units):
    """Generic min/max characterization of decreasing isotonic regression."""
    result = values.copy()
    indices = [units.index(f'Vinje_G{i}') for i in [1, 2, 3]]
    x = values[:, indices]
    for i in range(3):
        result[:, indices[i]] = np.min(np.stack([
            np.max(np.stack([x[:, a:b + 1].mean(axis=1) for b in range(i, 3)]), axis=0)
            for a in range(i + 1)]), axis=0)
    return result


report = {'status': 'running', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
          'command': f'{sys.executable} scripts/audit_cuda_capacity_final_package.py',
          'environment': {'python': sys.version, 'platform': platform.platform(),
                          'numpy': np.__version__, 'pandas': pd.__version__},
          'training': False, 'kaggle_submission': None}
try:
    protocol_path = ROOT / 'docs/cuda_xgb608_capacity_final_protocol.json'
    assert sha(protocol_path) == '840879d339732b14349d5355a886ef2e1b8e31101aa482ec8472df56a7be20d3'
    p = json.loads(protocol_path.read_text())
    directory = ROOT / 'artifacts/cuda-xgb608-capacity-final-package-001'
    m = json.loads((directory / 'manifest.json').read_text())
    assert m['status'] == 'complete' and m['protocol_sha256'] == sha(protocol_path)
    for name, expected in m['source'].items():
        assert sha(ROOT / name) == sha(directory / 'source' / name) == expected, name
    for name, expected in m['artifacts'].items():
        assert sha(directory / name) == expected, name
    code = directory / 'dataset/code'
    assert sha(code / 'package_manifest.json') == m['package_manifest_sha256'] == '9e8893ac7d46494a5b4001c68ef9090e365637436f2a456062479943cb489944'
    package = json.loads((code / 'package_manifest.json').read_text())
    assert package['visibility'] == 'PRIVATE' and package['raw_data_or_anchor_arrays_included'] is False
    assert {f.name for f in code.iterdir()} == {f['path'] for f in package['files']} | {'package_manifest.json'}
    for entry in package['files']:
        path = code / entry['path']
        assert not path.is_symlink() and path.stat().st_size == entry['size_bytes'] and sha(path) == entry['sha256']
    archive = directory / 'cuda-xgb608-capacity-final-private-dataset.zip'
    assert sha(archive) == m['dataset_archive_sha256'] == '2b76efff4319f0798c20c93f3b88942ce54233af48bda10c88d1c8a69f709cd2'
    with zipfile.ZipFile(archive) as zipped:
        expected = {path.relative_to(directory / 'dataset').as_posix(): path for path in code.iterdir()}
        assert len(zipped.namelist()) == len(set(zipped.namelist())) and set(zipped.namelist()) == set(expected)
        for name, path in expected.items():
            assert zipped.read(name) == path.read_bytes(), name
    val_path = ROOT / 'docs/cuda_xgb608_capacity_protocol.json'
    assert sha(val_path) == p['validation_protocol_sha256']
    validation = json.loads(val_path.read_text())
    assert p['model_parameters'] == validation['model_parameters']
    assert p['model_parameters']['n_estimators'] == 1200 and p['model_parameters']['max_depth'] == 8
    assert p['model_parameters']['learning_rate'] == .03 and p['model_parameters']['device'] == 'cuda:0'
    assert p['model_parameters']['n_jobs'] == 2
    for key, path in [('trainer_sha256', 'scripts/train_cuda_xgb608_capacity_final.py'),
                      ('exporter_sha256', 'scripts/export_cuda_xgb608_capacity_final.py'),
                      ('dataset_launcher_sha256', 'scripts/launch_cuda_xgb608_capacity_final_dataset.py'),
                      ('return_verifier_sha256', 'scripts/verify_cuda_xgb608_capacity_return.py'),
                      ('postprocess_sha256', 'sintef_project/postprocess.py')]:
        assert sha(ROOT / path) == p[key]
    for path, expected_sha in p['local_acceptance_inputs'].items():
        assert sha(ROOT / path) == expected_sha
    proof = json.loads((ROOT / 'artifacts/cuda-xgb608-capacity-2022-independent-audit-001/manifest.json').read_text())
    assert proof['status'] == 'passed' and proof['accepted'] is True
    assert proof['selection_lock_sha256'] == p['accepted_validation']['selection_lock_sha256']
    assert p['accepted_validation']['selected_weight'] == .25 and p['accepted_validation']['accepted'] is True
    for year in ['2021', '2022']:
        scores = p['accepted_validation']
        assert scores['selected_scores'][year] > max(scores['anchor_scores'][year], scores['control_scores'][year])
    train = pd.read_csv(ROOT / 'sintef_project/data/kernel/Unit_commitment_decisions.csv', usecols=['Run No', 'starttime'])
    mapping = pd.read_csv(ROOT / 'sintef_project/prediction_mapping.csv')
    assert p['train_case_ids'] == train['Run No'].tolist() and p['test_case_ids'] == mapping['Run No'].tolist()
    assert len(train) == 2922 and len(mapping) == 725 and p['training_label_year'] == 2022
    assert int(pd.to_datetime(train.starttime, utc=True).dt.year.max()) == 2022
    sample = np.sort(np.random.default_rng(4173).choice(len(train)*168, size=int(len(train)*168*.4), replace=False))
    buffer = io.BytesIO(); np.save(buffer, sample)
    assert hashlib.sha256(buffer.getvalue()).hexdigest() == p['train_sample_sha256']
    original_sample = np.load(ROOT / 'artifacts/lgb-normalized-lp-context-joint-final-001/train_sample_final.npy')
    assert sample.dtype == original_sample.dtype and sample.tobytes() == original_sample.tobytes()
    assert len(sample) == p['sampled_training_rows'] == 196358
    from scripts.export_cuda_xgb608_capacity_final import verify_anchor, compose
    anchor = verify_anchor(p)
    original = json.loads((ROOT / p['local_anchor']['original_final_protocol_path']).read_text())
    context = np.load(ROOT / original['local_anchor']['run_path'] / 'test_predictions.npy')
    with np.load(ROOT / p['local_anchor']['primary_raw_path']) as z:
        raw = z['predictions'].astype(np.float64)
    independent_anchor = projection(.5*context.astype(np.float64)+.5*raw, p['units'])
    assert independent_anchor.tobytes() == anchor.tobytes()
    # Reuse the old raw predictions as deterministic input to test the future nested export arithmetic.
    assert compose(anchor, raw, p['units']).tobytes() == projection(.75*anchor+.25*raw, p['units']).tobytes()
    result = subprocess.run([sys.executable, '-m', 'unittest', 'tests.test_cuda_xgb608_capacity_final'],
                            cwd=ROOT, text=True, capture_output=True)
    (OUT / 'tests.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0 and 'Ran 4 tests' in result.stderr
    dry = subprocess.run([sys.executable, str(code / 'launch_cuda_xgb608_capacity_final_dataset.py'),
                          '--input-root', str(directory / 'dataset'), '--protocol-sha256', sha(protocol_path),
                          '--package-sha256', sha(code / 'package_manifest.json'), '--mode', 'smoke',
                          '--run-id', 'audit-dryrun-only', '--dry-run'], cwd=ROOT, text=True, capture_output=True)
    (OUT / 'launcher_dry_run.log').write_text(dry.stdout + dry.stderr)
    assert dry.returncode == 0
    report.update(status='passed', protocol_sha256=sha(protocol_path),
                  package_manifest_sha256=sha(code / 'package_manifest.json'),
                  package_report_sha256=sha(directory / 'manifest.json'), dataset_archive_sha256=sha(archive),
                  sampled_training_rows=len(sample), original_sample_bitwise=True,
                  nested_anchor_independent_projection_bitwise=True, test_count=4,
                  review=['Fresh externally pinned genuine smoke; exact validation CUDA/runtime/native/source closure.',
                          'All fourteen 1200-round checkpoint fits and full121800-row exact GPU replay.',
                          'Locked accepted25% capacity nested into75% saved projected GPU608 anchor.',
                          'Strict archive/source/config/input/sample/probability/schema verification before export.',
                          'No new label selection, posthoc weight adjustment, GPU fit or submission in this audit.'])
except BaseException:
    report.update(status='failed', error=traceback.format_exc())
    raise
finally:
    shutil.copy2(__file__, OUT / Path(__file__).name)
    report['runtime_seconds'] = time.monotonic() - START
    report['artifacts'] = {path.name: sha(path) for path in OUT.iterdir() if path.is_file() and path.name != 'manifest.json'}
    (OUT / 'manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'status': report['status'], 'manifest_sha256': sha(OUT / 'manifest.json'),
                      'runtime_seconds': report['runtime_seconds']}), flush=True)
