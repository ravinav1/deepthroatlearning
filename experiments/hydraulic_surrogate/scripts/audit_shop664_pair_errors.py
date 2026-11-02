"""Independent Mann–Whitney and directed-endpoint audit of fixed SHOP error analysis."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
import numpy as np
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_hyljelihyl_river_cost_v2 import sha


def directed(positive: np.ndarray,negative: np.ndarray) -> tuple[float,int]:
    """Tie-corrected pair losses via concatenated average ranks, not searchsorted."""
    n,m=len(positive),len(negative);count=n*m
    ranks=rankdata(np.concatenate([positive,negative]),method='average')
    correct=float(ranks[:n].sum()-n*(n+1)/2)
    return float(count-correct),count


def main() -> None:
    out=ROOT/'artifacts/shop664-pair-error-independent-audit-001';out.mkdir(exist_ok=False);start=time.monotonic()
    report={'status':'initializing','timestamp_utc':datetime.now(timezone.utc).isoformat(),'training':False,'new_scores_or_weights_selected':False,'years':{}}
    try:
        run=ROOT/'artifacts/shop664-pair-error-diagnostic-001';mh='77d9b016be814698da69f0dc69ea0ffa24c214c3424410578c1d6bac5c6ad9be'
        assert sha(run/'manifest.json')==mh;m=json.loads((run/'manifest.json').read_text());assert m['status']=='complete'
        assert not m['training']and not m['selection_or_prediction_changes']
        for group in ['source','inputs']:
            for name,h in m[group].items():assert sha(ROOT/name)==h
        for name,h in m['artifacts'].items():assert sha(run/name)==h
        assert (run/'analyze_shop664_pair_errors.py').read_bytes()==(ROOT/'scripts/analyze_shop664_pair_errors.py').read_bytes()
        p=json.loads((ROOT/'docs/lgb_shop664_protocol.json').read_text());expected_rows=[]
        # Exact brute-force tie fixture independently validates the rank-sum convention.
        pos=np.array([0.,.5,.5,1.]);neg=np.array([0.,.5,1.])
        loss,count=directed(pos,neg);assert loss==sum(float(a<b)+.5*float(a==b)for a in pos for b in neg)and count==12
        for year in ['2021','2022']:
            candidates={'public_shop664':ROOT/p['anchor'][year]['prediction'],
                        'prepared_lgb664_blend':ROOT/f'artifacts/lgb-shop664-{year}-001/selected_validation_{year}.npz'}
            results={}
            for name,path in candidates.items():
                with np.load(path,allow_pickle=False)as z:y=z['labels'];v=z['predictions'];ids=z['case_ids'];units=z['units'].tolist()
                assert y.shape==v.shape==(61320,14)and np.isin(y,[0,1]).all()and np.isfinite(v).all()
                saved=np.load(run/f'{name}_{year}.npz',allow_pickle=False)
                assert saved['case_ids'].tolist()==ids.tolist()and saved['units'].tolist()==units
                errors=np.empty((14,14));counts=np.empty((14,14),dtype=np.int64)
                for i in range(14):
                    for j in range(14):errors[i,j],counts[i,j]=directed(v[y[:,i]==1,i],v[y[:,j]==0,j])
                assert np.array_equal(errors,saved['error_pair_counts'])and np.array_equal(counts,saved['pair_counts'])
                pooled_error,pooled_count=directed(v[y==1],v[y==0]);assert pooled_error==errors.sum()and pooled_count==counts.sum()
                auc=1-pooled_error/pooled_count;actual=m['years'][year][name]
                np.testing.assert_allclose(auc,actual['micro_auc'],atol=2e-15,rtol=0)
                endpoint=(errors.sum(axis=0)+errors.sum(axis=1))/(2*errors.sum());assert abs(endpoint.sum()-1)<1e-14
                rows=[]
                for j,unit in enumerate(units):
                    row={'year':year,'candidate':name,'unit':unit,'endpoint_error_share':float(endpoint[j]),
                        'within_unit_auc':float(1-errors[j,j]/counts[j,j]),
                        'mean_probability_minus_on_rate':float(v[:,j].mean()-y[:,j].mean())}
                    rows.append(row);expected_rows.append(row)
                ranked=sorted(rows,key=lambda r:-r['endpoint_error_share'])
                for actual_row,expected_row in zip(actual['units'],ranked,strict=True):
                    assert {k:actual_row[k]for k in ['year','candidate','unit']}=={k:expected_row[k]for k in ['year','candidate','unit']}
                    for k in ['endpoint_error_share','within_unit_auc','mean_probability_minus_on_rate']:
                        np.testing.assert_allclose(actual_row[k],expected_row[k],atol=2e-15,rtol=0)
                cross=1-np.trace(errors)/errors.sum();np.testing.assert_allclose(cross,actual['cross_unit_error_share'],atol=2e-15,rtol=0)
                results[name]={'micro_auc':auc,'directed14x14_counts_exact':True,'endpoint_ranking_exact':True,
                    'top_five_units':[r['unit']for r in ranked[:5]],'cross_unit_error_share':float(cross)}
            report['years'][year]=results
        rows=json.loads((run/'unit_summary.json').read_text());assert len(rows)==len(expected_rows)==56
        for a,b in zip(rows,expected_rows,strict=True):
            assert [a[k]for k in ['year','candidate','unit']]==[b[k]for k in ['year','candidate','unit']]
            for k in ['endpoint_error_share','within_unit_auc','mean_probability_minus_on_rate']:np.testing.assert_allclose(a[k],b[k],atol=2e-15,rtol=0)
        report.update(status='passed',diagnostic_manifest_sha256=mh,all_input_source_artifact_hashes_verified=True,
            directed_pair_matrices_independently_recomputed=True,pooled_and_unit_rank_sum_auc_verified=True,
            endpoint_shares_and_unit_rankings_verified=True,scope='Descriptive existingfold analysis only; no new recipe, calibration or unit splice.')
    except BaseException:report.update(status='failed',error=traceback.format_exc());raise
    finally:
        shutil.copy2(__file__,out/Path(__file__).name);report['runtime_seconds']=time.monotonic()-start
        report['artifacts']={q.name:sha(q)for q in out.iterdir()if q.is_file()and q.name!='manifest.json'}
        (out/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps({'status':report['status'],'manifest_sha256':sha(out/'manifest.json'),'seconds':report['runtime_seconds']}),flush=True)

if __name__=='__main__':main()
