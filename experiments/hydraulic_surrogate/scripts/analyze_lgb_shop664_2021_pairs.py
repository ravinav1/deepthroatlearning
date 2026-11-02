"""Describe the already selected 2021 blend's exact pooled AUC pair changes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import shutil
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    """Hash exact saved bytes."""
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pair_wins(y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Count wins plus half ties for each positive-unit/negative-unit pair."""
    wins = np.empty((14, 14), dtype=np.float64)
    for negative_unit in range(14):
        negative = np.sort(prediction[y[:, negative_unit] == 0, negative_unit])
        for positive_unit in range(14):
            positive = prediction[y[:, positive_unit] == 1, positive_unit]
            below = np.searchsorted(negative, positive, side='left')
            at_most = np.searchsorted(negative, positive, side='right')
            wins[positive_unit, negative_unit] = .5*np.sum(below+at_most, dtype=np.int64)
    return wins


def main() -> None:
    """Read only audited 2021 predictions; do not fit, select or decode 2022."""
    out = ROOT/'artifacts/lgb-shop664-2021-pair-decomposition-001'
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {'status':'initializing','timestamp_utc':datetime.now(timezone.utc).isoformat(),
              'training':False,'new_predictions':False,'new_weight_selection':False,
              'year':2021,'purpose':'Descriptive decomposition of an already selected development result; no protocol or acceptance changes.',
              'command':[sys.executable,*sys.argv],
              'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__}}
    try:
        audit_path = ROOT/'artifacts/lgb-shop664-2021-independent-audit-001/manifest.json'
        assert sha(audit_path) == '31e11af7408082765541ba8cf45d7c4c7bcfc2884d80a664a7d5969961d0e005'
        audit = json.loads(audit_path.read_text())
        run = ROOT/'artifacts/lgb-shop664-2021-001'
        assert sha(run/'manifest.json') == audit['candidate_manifest_sha256']
        m = json.loads((run/'manifest.json').read_text())
        ppath = ROOT/'docs/lgb_shop664_protocol.json'
        assert sha(ppath) == audit['protocol_sha256']
        p = json.loads(ppath.read_text()); units=p['units']
        apath = ROOT/p['anchor']['2021']['prediction']
        cpath = run/'selected_validation_2021.npz'
        assert sha(apath) == p['anchor']['2021']['prediction_sha256']
        assert sha(cpath) == m['artifacts'][cpath.name]
        with np.load(apath,allow_pickle=False) as a, np.load(cpath,allow_pickle=False) as b:
            assert np.array_equal(a['labels'],b['labels']) and np.array_equal(a['case_ids'],b['case_ids'])
            assert a['units'].tolist() == b['units'].tolist() == units
            y=a['labels'].copy(); anchor=a['predictions'].copy(); selected=b['predictions'].copy()
        assert y.shape == anchor.shape == selected.shape == (61320,14)
        assert np.isin(y,[0,1]).all() and np.isfinite(anchor).all() and np.isfinite(selected).all()
        positive=y.sum(axis=0,dtype=np.int64); negative=len(y)-positive
        denominator=int(positive.sum())*int(negative.sum())
        a=pair_wins(y,anchor); b=pair_wins(y,selected); delta=(b-a)/denominator
        assert abs(float(a.sum()/denominator)-m['anchor_micro_auc']) < 1e-13
        assert abs(float(b.sum()/denominator)-m['selected_micro_auc']) < 1e-13
        np.savez(out/'pair_matrices.npz',anchor_wins=a,selected_wins=b,delta_auc=delta,
                 positive_count=positive,negative_count=negative,units=np.asarray(units))
        cells=[{'positive_unit':units[i],'negative_unit':units[j],
                'delta_auc':float(delta[i,j]),'net_correct_pairs':float(b[i,j]-a[i,j])}
               for i in range(14) for j in range(14)]
        cells.sort(key=lambda c:c['delta_auc'],reverse=True)
        within=float(np.trace(delta)); total=float(delta.sum())
        report.update(status='complete',source_protocol_sha256=sha(ppath),
                      candidate_manifest_sha256=sha(run/'manifest.json'),independent_audit_sha256=sha(audit_path),
                      prediction_sha256={'anchor':sha(apath),'selected':sha(cpath)},selected_weight=.1,
                      anchor_auc=float(a.sum()/denominator),selected_auc=float(b.sum()/denominator),
                      delta_auc=total,within_unit_delta_auc=within,cross_unit_delta_auc=total-within,
                      global_positive_negative_pair_count=denominator,
                      top10_gaining_pairs=cells[:10],top10_losing_pairs=cells[-10:][::-1],
                      per_positive_unit_delta_auc={u:float(delta[i].sum()) for i,u in enumerate(units)},
                      per_negative_unit_delta_auc={u:float(delta[:,i].sum()) for i,u in enumerate(units)},
                      interpretation='Exact win/tie arithmetic is descriptive, not a causal attribution or significance estimate; no changes to the frozen 0.1 lock or pending 2022 gate.')
    except BaseException:
        report.update(status='failed',error=traceback.format_exc());raise
    finally:
        shutil.copy2(__file__,out/Path(__file__).name)
        report.update(runtime_seconds=time.monotonic()-started,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      source_sha256=sha(Path(__file__)))
        report['artifacts']={f.name:sha(f) for f in out.iterdir() if f.is_file()}
        (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'status':report['status'],'manifest_sha256':sha(out/'manifest.json'),
                          'delta_auc':report.get('delta_auc'),'within':report.get('within_unit_delta_auc'),
                          'cross':report.get('cross_unit_delta_auc')}),flush=True)


if __name__ == '__main__':
    main()
