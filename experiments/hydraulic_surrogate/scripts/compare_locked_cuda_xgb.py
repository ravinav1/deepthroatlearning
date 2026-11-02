"""Compare two already fixed candidates descriptively without tuning weights."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shlex
import shutil
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_cuda_xgb import digest, require
from sintef_project.postprocess import project_vinje


def main() -> None:
    """Verify aligned arrays and report both frozen recipes on each reused year."""
    out = ROOT / 'artifacts/compare-locked-cuda-xgb-current608-001'
    out.mkdir(exist_ok=False)
    start = time.time()
    lock_path = ROOT / 'artifacts/cuda-xgb-select-2021-001/selection_2021.json'
    require(digest(lock_path) == '315c420fa43a6a046fe97facafd473069de1d6761ad5d0f372617893bf3db040', 'Frozen lock changed')
    protocol_path = ROOT / 'docs/cuda_xgb_protocol.json'
    require(digest(protocol_path) == '7317e241c9a579531def94aec898b4cb3ae3192a5f5818062ee12b94db5b9b7c', '395 protocol changed')
    protocol = json.loads(protocol_path.read_text())
    anchor_path = ROOT / 'artifacts/lp-joint-comparison-001/analysis.json'
    current_path = ROOT / 'artifacts/lp-context-joint-comparison-001/analysis.json'
    require(digest(anchor_path) == protocol['anchor']['analysis_sha256'], '395 branch anchor changed')
    require(digest(current_path) == '32f692df612148094d4bfa06e45d4d0f321c35109444de297d4b0513cf2a43d5', 'Current608 anchor changed')
    anchor, current = json.loads(anchor_path.read_text()), json.loads(current_path.read_text())
    report = {'run_id': out.name, 'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'status': 'initializing',
              'command': shlex.join([sys.executable, *sys.argv]), 'training': 'none', 'weight_tuning': False,
              'description': 'Descriptive comparison of two already fixed candidates; does not change either recipe or original acceptance rule.',
              'caveat': 'Chronological years are reused development evidence, not untouched tests or leaderboard results.',
              'environment': {'python': sys.version, 'platform': platform.platform(),
                   'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'scikit-learn']}},
              'inputs': {str(p.relative_to(ROOT)): digest(p) for p in [lock_path, protocol_path, anchor_path, current_path]},
              'folds': {}, 'kaggle': None}
    for year in [2021, 2022]:
        selected_dir = ROOT / f'artifacts/cuda-xgb-select-{year}-001'
        selected_manifest_path = selected_dir / 'manifest.json'
        selected_manifest = json.loads(selected_manifest_path.read_text())
        selected_path = selected_dir / f'selected_validation_{year}.npz'
        current_array_path = current_path.parent / f'selected_validation_{year}.npz'
        anchor_array_path = anchor_path.parent / f'selected_validation_{year}.npz'
        require(selected_manifest['status'] == 'complete' and selected_manifest['selected_weight'] == .25, 'Selected recipe changed')
        if year == 2022:
            require(selected_manifest['accepted'] is True and selected_manifest['selection_lock_sha256'] == digest(lock_path),
                    'Locked2022 acceptance not established')
        for path, manifest in [(selected_path, selected_manifest), (current_array_path, current), (anchor_array_path, anchor)]:
            require(digest(path) == manifest['artifacts'][path.name], 'Fixed probability bytes changed')
        raw_dir = ROOT / f'artifacts/verify-cuda-xgb-{year}-return-001/download/cuda-xgb-{year}-001'
        raw_manifest_path = raw_dir / 'manifest.json'
        raw_manifest = json.loads(raw_manifest_path.read_text())
        raw_path = raw_dir / f'validation_{year}.npz'
        require(digest(raw_path) == raw_manifest['validation_sha256'], 'Raw probability bytes changed')
        with np.load(selected_path, allow_pickle=False) as chosen, np.load(current_array_path, allow_pickle=False) as other, \
                np.load(anchor_array_path, allow_pickle=False) as prior, np.load(raw_path, allow_pickle=False) as raw:
            units, ids, labels = chosen['units'].tolist(), chosen['case_ids'], chosen['labels']
            for values in [other, prior, raw]:
                require(values['units'].tolist() == units and np.array_equal(values['case_ids'], ids)
                        and np.array_equal(values['labels'], labels), 'Candidate ordering/labels differ')
            first, second = chosen['predictions'], other['predictions']
            reconstructed = project_vinje(.75*prior['predictions'].astype(np.float64)+.25*raw['predictions'].astype(np.float64), units)
            require(reconstructed.tobytes() == first.tobytes(), 'Locked XGB reconstruction differs')
            xgb_auc, current_auc = [float(roc_auc_score(labels.ravel(), p.ravel())) for p in [first, second]]
            require(abs(xgb_auc-selected_manifest['selected_micro_auc']) <= 1e-12
                    and abs(current_auc-current['folds'][str(year)]['selected_micro_auc']) <= 1e-12, 'Independent scores differ')
            report['folds'][str(year)] = {'fixed_xgb_blend_micro_auc': xgb_auc, 'current608_blend_micro_auc': current_auc,
                'delta_xgb_minus_current608': xgb_auc-current_auc, 'xgb_probability_reconstruction_bitwise': True,
                'case_ids': ids.tolist(), 'units': units,
                'unit_delta_xgb_minus_current608': {u:float(roc_auc_score(labels[:, j],first[:, j])-roc_auc_score(labels[:, j],second[:, j])) for j,u in enumerate(units)}}
        for p in [selected_manifest_path, selected_path, current_array_path, anchor_array_path, raw_manifest_path, raw_path]:
            report['inputs'][str(p.relative_to(ROOT))] = digest(p)
    deltas = [f['delta_xgb_minus_current608'] for f in report['folds'].values()]
    report.update(status='complete', xgb_strictly_dominates_current608=all(d>0 for d in deltas),
                  current608_strictly_dominates_xgb=all(d<0 for d in deltas),
                  xgb_accepted_against_its_original_joint_anchor=True, transfer_gate_met=False,
                  action='Preserve both fixed candidates; this comparison makes no new selection or blend.', runtime_seconds=time.time()-start)
    shutil.copy2(__file__,out/Path(__file__).name)
    report['source_sha256']=digest(Path(__file__))
    (out/'manifest.json').write_text(json.dumps(report,indent=2))
    with (ROOT/'docs/experiments.md').open('a') as stream:
        stream.write(f'\n## {out.name} — {report["timestamp_utc"]}\n\nStatus:complete. Command:`{report["command"]}`. Descriptive comparison only, no weight tuning or training. Fixed25%GPU-XGB versus current608blend deltas2021/2022:{deltas}. Neither candidate dominates both years. Both arrays/labels/IDs/unit order verified; exact float64 lockedXGB reconstruction passed. The original395branch is accepted against its older declared jointanchor, but transfergate remains unmet. Exact source/inputs/hashes/environment:`artifacts/{out.name}/manifest.json`. NoKaggle score.\n')
    print(json.dumps({year:{k:v for k,v in f.items() if k in ['fixed_xgb_blend_micro_auc','current608_blend_micro_auc','delta_xgb_minus_current608']} for year,f in report['folds'].items()}),flush=True)


if __name__=='__main__':
    main()
