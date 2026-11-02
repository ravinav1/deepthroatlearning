"""Frozen two-capacity2021 selection and single locked2022 CPU-vector comparison."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.cpu_vector14_fold_common import read_protocol, validate_lock, choose, artifacts, verify_files, sha, WEIGHTS
from sintef_project.postprocess import project_vinje


def reference(record: dict, labels: np.ndarray, ids: np.ndarray, units: list[str]) -> np.ndarray:
    """Validate reference provenance and original target/ID ordering."""
    manifest = json.loads((ROOT / record['manifest']).read_text())
    assert sha(ROOT / record['manifest']) == record['manifest_sha256']
    path = ROOT / record['prediction']; assert sha(path) == record['prediction_sha256'] == manifest['artifacts'][path.name]
    with np.load(path, allow_pickle=False) as data:
        assert data['case_ids'].tolist() == ids.tolist() and data['units'].tolist() == units
        assert np.array_equal(data['labels'], labels)
        values = data['predictions'].astype(np.float64)
    assert values.shape == labels.shape and np.isfinite(values).all() and 0 <= values.min() <= values.max() <= 1
    return values


def main() -> None:
    ap = argparse.ArgumentParser()
    for name in ['protocol', 'fit-run', 'fit-audit']: ap.add_argument('--' + name, type=Path, required=True)
    for name in ['protocol-sha256', 'fit-manifest-sha256', 'fit-audit-sha256', 'run-id']: ap.add_argument('--' + name, required=True)
    for name in ['selection-lock', 'lock-audit']: ap.add_argument('--' + name, type=Path)
    for name in ['selection-lock-sha256', 'lock-audit-sha256']: ap.add_argument('--' + name)
    ap.add_argument('--year', type=int, choices=[2021, 2022], required=True); a = ap.parse_args()
    assert Path(a.run_id).name == a.run_id and a.run_id not in ['', '.', '..']
    out = ROOT / 'artifacts' / a.run_id; out.mkdir(exist_ok=False); start = time.monotonic()
    m = {'status': 'initializing', 'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'year': a.year,
         'command': [sys.executable, *sys.argv], 'training': False, 'protocol_sha256': a.protocol_sha256}
    try:
        p = read_protocol(a.protocol, a.protocol_sha256); lock = None
        if a.year == 2022: lock = validate_lock(a.selection_lock, a.selection_lock_sha256, p, a.protocol_sha256, a.lock_audit, a.lock_audit_sha256)
        else: assert a.selection_lock is None and a.lock_audit is None
        assert sha(a.fit_run / 'manifest.json') == a.fit_manifest_sha256 and sha(a.fit_audit) == a.fit_audit_sha256
        fm = json.loads((a.fit_run / 'manifest.json').read_text()); proof = json.loads(a.fit_audit.read_text())
        assert fm['status'] == 'complete' and fm['year'] == a.year and fm['protocol_sha256'] == a.protocol_sha256
        assert fm['rounds'] == (p['candidate_rounds'] if lock is None else [lock['selected_rounds']])
        assert fm['selection_lock_sha256'] == (None if lock is None else a.selection_lock_sha256)
        assert proof['status'] == 'passed' and proof['fit_manifest_sha256'] == a.fit_manifest_sha256
        assert proof['protocol_sha256'] == a.protocol_sha256 and proof['all_native_models_and_replays_verified']
        assert proof['all_input_bridge_bytes_verified']
        verify_files(a.fit_run, fm['artifacts']); bridge = Path(fm['bridge'])
        assert sha(bridge / 'manifest.json') == fm['bridge_manifest_sha256']
        bm = json.loads((bridge / 'manifest.json').read_text()); verify_files(bridge, bm['artifacts'])
        assert bm['year'] == a.year and bm['protocol_sha256'] == a.protocol_sha256
        if lock is not None: assert bm['selection_lock_sha256'] == a.selection_lock_sha256
        with np.load(bridge / 'validation_targets.npz', allow_pickle=False) as data:
            labels, ids, units = data['labels'], data['case_ids'], data['units'].tolist()
        assert units == p['units'] and ids.tolist() == p['folds'][str(a.year)]['validation_ids']
        anchor = reference(p['anchor'][str(a.year)], labels, ids, units)
        control = reference(p['control'][str(a.year)], labels, ids, units)
        rounds = p['candidate_rounds'] if lock is None else [lock['selected_rounds']]
        weights = WEIGHTS if lock is None else [0., lock['selected_weight']]
        scores, values = {}, {}
        for r in rounds:
            raw = np.load(a.fit_run / f'predictions_{r}.npy', allow_pickle=False)
            assert raw.dtype == np.float32 and raw.shape == labels.shape
            for w in weights:
                key = f'{r}:{w}'; result = project_vinje((1 - w) * anchor + w * raw.astype(np.float64), units)
                if w == 0: assert result.tobytes() == anchor.tobytes()
                scores[key] = float(roc_auc_score(labels.ravel(), result.ravel())); values[key] = result
        selected_r, selected_w = choose(scores) if lock is None else (lock['selected_rounds'], lock['selected_weight'])
        key = f'{selected_r}:{selected_w}'; anchor_auc = scores[f'{selected_r}:0.0']
        matched = project_vinje((1 - selected_w) * anchor + selected_w * control, units)
        control_auc = float(roc_auc_score(labels.ravel(), matched.ravel()))
        eligible = selected_w > 0 and scores[key] > max(anchor_auc, control_auc)
        np.savez_compressed(out / f'selected_validation_{a.year}.npz', predictions=values[key], labels=labels, case_ids=ids, units=np.asarray(units))
        m.update(status='complete', scores=scores, selected_rounds=selected_r, selected_weight=selected_w,
                 selected_micro_auc=scores[key], anchor_micro_auc=anchor_auc, control_micro_auc=control_auc,
                 fit_manifest_sha256=a.fit_manifest_sha256, fit_audit_sha256=a.fit_audit_sha256,
                 accepted=eligible if a.year == 2022 else None, eligible_for_2022=eligible if a.year == 2021 else None,
                 caveat='Reused development years; CPU/shared-vector architecture is a combined algorithm change. RawGPU664 control tests reweighting complementarity, not isolated architectural causality.')
        if lock is None:
            locked = {'version': 'cpu_vector14_2021_lock_v1', 'protocol_sha256': a.protocol_sha256,
                      'candidate_rounds': p['candidate_rounds'], 'candidate_weights': WEIGHTS, 'scores_2021': scores,
                      'selected_rounds': selected_r, 'selected_weight': selected_w, 'selected_2021_micro_auc': scores[key],
                      'anchor_2021_micro_auc': anchor_auc, 'control_2021_micro_auc': control_auc,
                      'eligible_for_2022': eligible, 'fit_manifest_sha256': a.fit_manifest_sha256,
                      **{k + '_prediction_sha256': p[k]['2021']['prediction_sha256'] for k in ['anchor', 'control']}}
            (out / 'selection_2021.json').write_text(json.dumps(locked, indent=2) + '\n'); m['selection_lock_sha256'] = sha(out / 'selection_2021.json')
        else: m['selection_lock_sha256'] = a.selection_lock_sha256
    except BaseException:
        m.update(status='failed', error=traceback.format_exc()); raise
    finally:
        shutil.copy2(__file__, out / Path(__file__).name)
        m.update(runtime_seconds=time.monotonic() - start, artifacts=artifacts(out))
        (out / 'manifest.json').write_text(json.dumps(m, indent=2) + '\n'); print(m['status'], sha(out / 'manifest.json'), flush=True)


if __name__ == '__main__': main()
