"""Test frozen fullsample608 complementarity against the accepted SHOP blend."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sintef_project.postprocess import project_vinje

WEIGHTS = [0., .1, .25, .5]


def sha(path: Path) -> str:
    """Hash exact immutable source or evidence bytes."""
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(value: bool, message: str) -> None:
    """Fail before selection if any frozen identity or gate differs."""
    if not value:
        raise ValueError(message)


def decision(scores: dict[str, float], control_auc: float, selected: float | None = None) -> tuple[float, bool]:
    """Break 2021 ties toward lower weights; apply strict matched-control gates."""
    weight = max(WEIGHTS, key=lambda w: (scores[str(w)], -w)) if selected is None else selected
    require(weight in WEIGHTS, 'Weight outside frozen grid')
    return weight, bool(weight > 0 and scores[str(weight)] > max(scores['0.0'], control_auc))


def load_fold(record: dict, ids: list[int], units: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Read one requested year only after exact source/evidence/lock guards."""
    path = ROOT / record['prediction']
    require(sha(path) == record['prediction_sha256'], 'Prediction changed')
    with np.load(path, allow_pickle=False) as saved:
        require(set(saved.files) == {'predictions', 'labels', 'case_ids', 'units'}, 'Unexpected array schema')
        require(saved['case_ids'].tolist() == ids and saved['units'].tolist() == units, 'Fold alignment changed')
        prediction, labels = saved['predictions'].astype(np.float64), saved['labels'].copy()
    require(prediction.shape == labels.shape == (len(ids) * 168, 14), 'Wrong fold shape')
    require(np.isfinite(prediction).all() and 0 <= prediction.min() <= prediction.max() <= 1, 'Invalid probabilities')
    require(np.isin(labels, [0, 1]).all(), 'Invalid binary labels')
    return prediction, labels


def check_lock(lock: dict, protocol_sha: str, selector_sha: str) -> None:
    """Require the exact positive eligible 2021 choice before opening 2022 arrays."""
    require(lock['version'] == 'fullsample608_shop_reuse_lock_v1' and lock['selection_year'] == 2021,
            'Wrong selection lock')
    require(lock['protocol_sha256'] == protocol_sha and lock['selector_sha256'] == selector_sha,
            'Lock source/protocol changed')
    require(lock['candidate_weights'] == WEIGHTS, 'Lock grid changed')
    chosen, eligible = decision(lock['scores_2021'], lock['control_2021_micro_auc'])
    require(eligible and lock['eligible_for_2022'] and chosen == lock['selected_weight'], 'Lock is not eligible')
    require(lock['selected_2021_micro_auc'] == lock['scores_2021'][str(chosen)]
            and lock['anchor_2021_micro_auc'] == lock['scores_2021']['0.0'], 'Lock scores changed')


def main() -> None:
    """Evaluate one development year without fitting or using public scores."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', type=Path, required=True)
    ap.add_argument('--protocol-sha256', required=True)
    ap.add_argument('--year', type=int, choices=[2021, 2022], required=True)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--preflight-audit', type=Path)
    ap.add_argument('--preflight-audit-sha256')
    ap.add_argument('--selection-lock', type=Path)
    ap.add_argument('--selection-lock-sha256')
    ap.add_argument('--lock-audit', type=Path)
    ap.add_argument('--lock-audit-sha256')
    args = ap.parse_args()
    require(Path(args.run_id).name == args.run_id and args.run_id not in ['', '.', '..'], 'Invalid run ID')
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {'status': 'initializing', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': [sys.executable, *sys.argv], 'year': args.year, 'training': False,
              'kaggle_submission': None, 'protocol_sha256': args.protocol_sha256,
              'environment': {'python': sys.version, 'platform': platform.platform(),
                              'packages': {x: importlib.metadata.version(x) for x in ['numpy', 'scikit-learn']}},
              'caveat': 'Both years are reused chronological development data, not untouched holdout evidence.'}
    try:
        require(sha(args.protocol) == args.protocol_sha256, 'Frozen protocol changed')
        p = json.loads(args.protocol.read_text())
        require(p['version'] == 'fullsample608_shop_reuse_v1' and p['candidate_weights'] == WEIGHTS,
                'Unexpected reuse recipe')
        for group in ['source', 'inputs']:
            for path, expected in p[group].items():
                require(sha(ROOT / path) == expected, 'Frozen ' + group + ' changed: ' + path)
        selector_sha = sha(Path(__file__))
        require(p['source']['scripts/select_fullsample608_shop_reuse.py'] == selector_sha, 'Selector differs')
        for name, record in p['folds'][str(args.year)]['components'].items():
            manifest = json.loads((ROOT / record['manifest']).read_text())
            require(manifest['status'] == 'complete', 'Incomplete component ' + name)
            require(manifest['artifacts'][Path(record['prediction']).name] == record['prediction_sha256'],
                    'Component artifact closure differs')
        if args.preflight:
            report.update(status='passed_preflight_only', labels_decoded=False)
            return
        require(args.preflight_audit is not None and sha(args.preflight_audit) == args.preflight_audit_sha256,
                'Independent exact-source preflight audit required')
        audit = json.loads(args.preflight_audit.read_text())
        require(audit['status'] == 'passed' and audit['protocol_sha256'] == args.protocol_sha256,
                'Preflight not approved')
        report['preflight_audit_sha256'] = args.preflight_audit_sha256
        lock = None
        if args.year == 2022:
            require(args.selection_lock is not None and args.lock_audit is not None, '2022 requires audited 2021 lock')
            require(sha(args.selection_lock) == args.selection_lock_sha256 and sha(args.lock_audit) == args.lock_audit_sha256,
                    'Lock evidence changed')
            lock = json.loads(args.selection_lock.read_text())
            check_lock(lock, args.protocol_sha256, selector_sha)
            lock_audit = json.loads(args.lock_audit.read_text())
            require(lock_audit['status'] == 'passed' and lock_audit['selection_lock_sha256'] == args.selection_lock_sha256
                    and lock_audit['eligible_for_2022'], '2021 result has no independent eligibility proof')
            report.update(selection_lock_sha256=args.selection_lock_sha256, lock_audit_sha256=args.lock_audit_sha256)
        else:
            require(args.selection_lock is None and args.lock_audit is None, '2021 must make a new choice')
        fold = p['folds'][str(args.year)]
        loaded = {name: load_fold(record, fold['validation_ids'], p['units']) for name, record in fold['components'].items()}
        anchor, labels = loaded['anchor']
        raw, raw_labels = loaded['candidate']
        control, control_labels = loaded['control']
        require(np.array_equal(labels, raw_labels) and np.array_equal(labels, control_labels), 'Component labels differ')
        weights = WEIGHTS if lock is None else [0., lock['selected_weight']]
        values = {str(w): project_vinje((1-w)*anchor + w*raw, p['units']) for w in weights}
        scores = {w: float(roc_auc_score(labels.ravel(), prediction.ravel())) for w, prediction in values.items()}
        selected = max(WEIGHTS, key=lambda w: (scores[str(w)], -w)) if lock is None else lock['selected_weight']
        matched = project_vinje((1-selected)*anchor + selected*control, p['units'])
        control_auc = float(roc_auc_score(labels.ravel(), matched.ravel()))
        selected, eligible = decision(scores, control_auc, selected)
        np.savez_compressed(out / f'selected_validation_{args.year}.npz', predictions=values[str(selected)],
                            labels=labels, case_ids=np.asarray(fold['validation_ids']), units=np.asarray(p['units']))
        report.update(status='complete', labels_decoded=True, scores=scores, selected_weight=selected,
                      selected_micro_auc=scores[str(selected)], anchor_micro_auc=scores['0.0'],
                      control_micro_auc=control_auc, beats_anchor_and_control=eligible)
        if lock is None:
            lock = {'version': 'fullsample608_shop_reuse_lock_v1', 'selection_year': 2021,
                    'locked_utc': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': args.protocol_sha256,
                    'selector_sha256': selector_sha, 'candidate_weights': WEIGHTS, 'selected_weight': selected,
                    'scores_2021': scores, 'selected_2021_micro_auc': scores[str(selected)],
                    'anchor_2021_micro_auc': scores['0.0'], 'control_2021_micro_auc': control_auc,
                    'eligible_for_2022': eligible, 'formula': p['formula']}
            (out / 'selection_2021.json').write_text(json.dumps(lock, indent=2) + '\n')
            report['selection_lock_sha256'] = sha(out / 'selection_2021.json')
        else:
            report.update(accepted=eligible, meets_transfer_gate=min(lock['selected_2021_micro_auc'], scores[str(selected)]) >= .998)
        print(json.dumps({k: report[k] for k in ['status', 'selected_weight', 'selected_micro_auc', 'anchor_micro_auc', 'control_micro_auc', 'beats_anchor_and_control']}), flush=True)
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        shutil.copy2(__file__, out / Path(__file__).name)
        shutil.copy2(args.protocol, out / 'protocol.json')
        report['runtime_seconds'] = time.monotonic() - started
        report['artifacts'] = {x.name: sha(x) for x in out.iterdir() if x.is_file() and x.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
