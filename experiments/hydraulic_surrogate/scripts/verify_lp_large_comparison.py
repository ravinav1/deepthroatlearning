"""Independently reconstruct the frozen large-LP blend; no model inference."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sintef_project.postprocess import project_vinje


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Verify frozen provenance and reconstruct both folds in source order."""
    run = ROOT / 'artifacts/lp-large-blend-001'
    out = ROOT / 'artifacts/lp-large-blend-verification-001'
    out.mkdir(exist_ok=False)
    started = time.time()
    report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
              'source_sha256': digest(Path(__file__)), 'model_fits': 0, 'checkpoint_replay': False}
    try:
        manifest = json.loads((run / 'manifest.json').read_text())
        assert manifest['status'] == 'complete'
        for relative, expected in manifest['sources'].items():
            assert digest(ROOT / relative) == expected, relative
        for relative, expected in manifest['data'].items():
            assert digest(ROOT / relative) == expected, relative
        for relative, expected in manifest['artifacts'].items():
            assert digest(run / relative) == expected, relative
        lock = json.loads((run / 'selection_2021.json').read_text())
        assert manifest['selected_scalar_weight'] == .25
        assert manifest['source_order'] == ['lgb-normalized-lp-001', 'lgb-normalized-lp-context-001', 'lgb-normalized-lp-large-001']
        assert manifest['selected_weights'] == [.1875, .5625, .25]
        report['inputs'] = {str(run / 'manifest.json'): digest(run / 'manifest.json'),
                           str(run / 'selection_2021.json'): digest(run / 'selection_2021.json')}
        report['folds'] = {}
        for year in [2021, 2022]:
            values, labels, ids, units = None, None, None, None
            for source, weight in zip(manifest['source_order'], manifest['selected_weights']):
                with np.load(ROOT / 'artifacts' / source / f'validation_{year}.npz', allow_pickle=False) as saved:
                    if values is None:
                        values = np.zeros(saved['predictions'].shape, dtype=np.float64)
                        labels, ids, units = saved['labels'], saved['case_ids'], saved['units']
                    assert np.array_equal(labels, saved['labels']) and np.array_equal(ids, saved['case_ids']) and np.array_equal(units, saved['units'])
                    values += saved['predictions'].astype(np.float64) * weight
            values = project_vinje(values, units.tolist())
            with np.load(run / f'selected_validation_{year}.npz', allow_pickle=False) as saved:
                assert saved['predictions'].dtype == values.dtype and saved['predictions'].tobytes() == values.tobytes()
                assert np.array_equal(saved['labels'], labels) and np.array_equal(saved['case_ids'], ids) and np.array_equal(saved['units'], units)
            score = float(roc_auc_score(labels.ravel(), values.ravel()))
            assert abs(score - manifest['folds'][str(year)]['selected_micro_auc']) < 1e-14
            report['folds'][str(year)] = {'bitwise_equal': True, 'micro_auc': score}
        joint_path = ROOT / 'artifacts/lp-joint-comparison-001/analysis.json'
        joint = json.loads(joint_path.read_text())
        report['joint_reference_sha256'] = digest(joint_path)
        report['below_newer_joint_on_both_years'] = all(report['folds'][str(y)]['micro_auc'] < joint['folds'][str(y)]['selected_micro_auc'] for y in [2021, 2022])
        report['recommendation'] = 'No full fit: locked blend improves its original context/base prior but remains below the newer accepted joint blend on both reused development years.'
        report['status'] = 'complete'
    except BaseException:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['runtime_seconds'] = time.time() - started
        (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        with (ROOT / 'docs/experiments.md').open('a') as stream:
            stream.write(f'\n## lp-large-blend-verification-001 — {report["timestamp_utc"]}\n\nStatus: {report["status"]}. Command: `.venv-ml/bin/python scripts/verify_lp_large_comparison.py`. All comparison source/data/artifact hashes checked; independent float64 source-order composition and Vinje projection reproduce both saved folds byte for byte. No model fit or checkpoint inference. Proof: `artifacts/lp-large-blend-verification-001/manifest.json`. {report.get("recommendation", "See failure proof.")}\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
