"""Rebuild physical matrices and verify stored all14 incumbents and fixed-LP KKT without solving."""
from pathlib import Path
from datetime import datetime,timezone
from unittest.mock import patch
from types import SimpleNamespace
import argparse,hashlib,json,platform,shutil,sys,time,traceback
from scipy.stats import rankdata
import numpy as np
import pandas as pd
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root));start=time.monotonic()
parser=argparse.ArgumentParser();parser.add_argument('--manifest-sha256',required=True);args=parser.parse_args()
from sintef_project.lp_features import HydroCaseData,UNIT_ORDER
from sintef_project.lp_shop_relaxation import ShopRelaxationLP
out=root/'artifacts/all14-integer-water-budget-twelve-result-independent-audit-001';out.mkdir(exist_ok=False);run=root/'artifacts/all14-integer-water-budget-twelve-001'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def arrays(p):
 with np.load(p,allow_pickle=False)as z:return{k:z[k]for k in z.files}
def same(a,b):
 assert set(a)==set(b)
 for k in a:assert a[k].shape==b[k].shape and a[k].dtype==b[k].dtype and a[k].tobytes()==b[k].tobytes(),k
report={'status':'initializing','timestamp_utc':datetime.now(timezone.utc).isoformat(),'optimization_calls':0,'labels_decoded':False,'cases':[]}
try:
 assert sha(run/'manifest.json')==args.manifest_sha256
 m=json.loads((run/'manifest.json').read_text());p=json.loads((run/'protocol.json').read_text())
 assert sha(run/'protocol.json')=='14711cf1aa752bcd0932932d59cad48371e1c0c9482d296301e67b2fd0207abc'
 assert m['status']=='complete'and m['physical_feasibility_passed']and all(m['physical_gates'].values())and m['labels_decoded']
 for name,h in m['artifacts'].items():assert sha(run/name)==h,name
 for category in ['source','data','inputs']:
  for name,h in p[category].items():assert sha(root/name)==h,name
 assert len(m['cases'])==12
 assert len({r['pid']for r in m['cases']if r['case_id']not in [1,1826]})==10
 original_selection=json.loads((root/'docs/integer_water_budget_protocol.json').read_text())
 assert [r['case_id']for r in m['cases']]==original_selection['case_ids']and p['blocks']==original_selection['blocks']
 lock=json.loads((run/'physical_lock.json').read_text());assert sha(run/'physical_lock.json')==m['physical_lock_sha256']and all(lock['physical_gates'].values())
 assert datetime.fromisoformat(lock['timestamp_utc'])<=datetime.fromisoformat(m['labels_opened_utc'])
 for case_id,expected in lock['proof_sha256'].items():assert sha(run/f'case_{int(case_id):04d}'/'proof.json')==expected
 original=pd.read_csv;target=root/'sintef_project/data/kernel/Unit_commitment_decisions.csv'
 def guarded(path,*a,**kw):
  if isinstance(path,(str,Path))and Path(path).resolve()==target:assert not a and kw=={'usecols':['Run No','starttime']}
  return original(path,*a,**kw)
 with patch.object(pd,'read_csv',guarded):data=HydroCaseData(root)
 for selected,record in zip(p['cases'],m['cases']):
  case_id=record['case_id'];assert case_id==selected['case_id'];folder=run/f'case_{case_id:04d}'
  assert json.loads((folder/'proof.json').read_text())==record
  case=arrays(run/f'case_{case_id:04d}_inputs.npz');same(case,arrays(root/p['cache']/f'case_{case_id:04d}_inputs.npz'))
  fresh=data.case(selected['metadata_position']);fresh['natural']=fresh['natural'].astype(np.float64)
  for k,field in [('pmin','p_min'),('start_cost','startcost_const'),('stop_cost','stopcost_const')]:fresh[k]=np.asarray([data.model['generator'][u][field]for u in UNIT_ORDER],dtype=np.float64)
  same(case,fresh)
  result=arrays(folder/'result.npz');lp=arrays(folder/'fixed_lp_1.npz');same(lp,arrays(folder/'fixed_lp_2.npz'))
  proof=record['proof'];assert record['status']=='complete'and proof['selected_units']==UNIT_ORDER
  assert proof['mip_status']==0 and proof['proven_optimal']and proof['mip_rel_gap_requested']==0
  assert proof['mip_gap'] is not None and np.isfinite(proof['mip_gap'])and proof['mip_gap']>=0
  assert proof['mip_time_limit_seconds']==60 and proof['fixed_lp_time_limit_seconds_each']==1
  assert record['all_slacks_zero']and all(np.max(np.abs(result[k]))==0 for k in ['minimum_slack','maximum_slack','bypass_slack','vest_slack'])
  assert bool(lp['success'])and int(lp['status'])==0 and result['integer_states'].shape==(14,168)
  solver=ShopRelaxationLP(data.network);capture={};scale=1+np.abs(case['price']).mean()
  def no_solve(c,**kw):
   capture.update(c=c,**kw)
   return SimpleNamespace(success=True,status=0,x=lp['solution'],fun=float(lp['objective']),nit=int(result['iterations']),eqlin=SimpleNamespace(marginals=lp['eqlin_marginals']))
  with patch('sintef_project.lp_shop_relaxation.linprog',no_solve):rebuilt=solver.solve(case,1.,1.)
  for k in rebuilt:np.testing.assert_allclose(rebuilt[k],result[k],rtol=2e-11,atol=1e-5,err_msg=k)
  selected_states=solver.state.ravel();x=lp['solution'];bounds=capture['bounds'].copy();states=result['integer_states'].ravel()
  assert np.isin(states,[0,1]).all();np.testing.assert_array_equal(x[selected_states],states)
  bounds[selected_states,0]=bounds[selected_states,1]=states
  ae,be,au,bu,c=[capture[k]for k in ['A_eq','b_eq','A_ub','b_ub','c']]
  residuals={'eqlin_residual':be-ae@x,'ineqlin_residual':bu-au@x,'lower_residual':x-bounds[:,0],'upper_residual':bounds[:,1]-x}
  for k,v in residuals.items():np.testing.assert_allclose(v,lp[k],atol=1e-7,rtol=1e-8)
  stationarity=float(np.max(np.abs(c-ae.T@lp['eqlin_marginals']-au.T@lp['ineqlin_marginals']-lp['lower_marginals']-lp['upper_marginals'])))
  assert stationarity<1e-5 and lp['ineqlin_marginals'].max()<1e-7 and lp['lower_marginals'].min()>-1e-7 and lp['upper_marginals'].max()<1e-7
  finite=np.isfinite(bounds[:,1]);complementarity=max(float(np.max(np.abs(lp['ineqlin_marginals']*lp['ineqlin_residual']))),float(np.max(np.abs(lp['lower_marginals']*lp['lower_residual']))),float(np.max(np.abs(lp['upper_marginals'][finite]*lp['upper_residual'][finite]))))
  assert complementarity<1e-5
  dual=float(lp['eqlin_marginals']@be+lp['ineqlin_marginals']@bu+lp['lower_marginals']@bounds[:,0]+lp['upper_marginals'][finite]@bounds[finite,1])
  np.testing.assert_allclose([c@x,dual],[lp['objective']]*2,atol=1e-5,rtol=1e-11)
  incumbent=result['integer_solution'];original_bounds=capture['bounds']
  violation=max(float(np.max(np.abs(ae@incumbent-be))),float(np.max(au@incumbent-bu)),float(np.max(original_bounds[:,0]-incumbent)),float(np.max(incumbent-original_bounds[:,1])))
  assert violation<1e-5;np.testing.assert_allclose(incumbent[selected_states],states,atol=1e-5,rtol=0)
  np.testing.assert_allclose(c@incumbent,proof['mip_objective_scaled'],rtol=2e-11,atol=1e-5)
  assert proof['mip_dual_bound_scaled']<=proof['mip_objective_scaled']+1e-7
  control=arrays(root/p['cache']/f'case_{case_id:04d}_result.npz');three=arrays(root/p['comparison']/f'case_{case_id:04d}_integer.npz')
  assert result['objective']<=min(control['objective'],three['objective'])+1e-4 and 0<record['peak_rss_bytes']<2*1024**3
  for key,expected in [('objective_loss_vs_relaxed_eur',control['objective']-result['objective']),('objective_loss_vs_three_unit_eur',three['objective']-result['objective'])]:np.testing.assert_allclose(record[key],expected,atol=1e-5,rtol=1e-11)
  report['cases'].append({'case_id':case_id,'physical_incumbent_max_violation':violation,'fixed_lp_stationarity_max':stationarity,
    'fixed_lp_complementarity_max':complementarity,'native_relative_gap':proof['mip_gap'],'native_absolute_gap_eur':record['native_absolute_gap_eur'],
    'conditional_lp_objective_gap_eur':proof['objective_gap_eur'],'seconds':record['runtime_seconds'],'peak_rss_bytes':record['peak_rss_bytes'],
    'objective_loss_vs_three_unit_eur':record['objective_loss_vs_three_unit_eur'],
    'unit_state_changed_hours_vs_three_unit':{u:int(np.count_nonzero(result['state'][j]!=three['state'][j]))for j,u in enumerate(UNIT_ORDER)}})
 # Independently parse only the fixed pre2021 rows and recompute rank-sum metrics.
 ids=original_selection['case_ids'];meta=pd.read_csv(target,usecols=['Run No','starttime']);keep=meta['Run No'].isin(ids).to_numpy()
 assert meta.loc[keep,'Run No'].tolist()==ids
 assert (pd.to_datetime(meta.loc[keep,'starttime'],utc=True)+pd.Timedelta(hours=167)<pd.Timestamp('2021-01-01',tz='UTC')).all()
 columns=[f'result_committed_{u}_t{h}'for u in UNIT_ORDER for h in range(168)]
 frame=pd.read_csv(target,skiprows=(np.flatnonzero(~keep)+1).tolist(),usecols=['Run No',*columns],dtype={c:'uint8'for c in columns})
 assert frame['Run No'].tolist()==ids
 truth=frame[columns].to_numpy().reshape(12,14,168);assert np.isin(truth,[0,1]).all()
 saved=arrays(run/'diagnostic.npz');assert np.array_equal(saved['labels'],truth)and saved['case_ids'].tolist()==ids and saved['units'].tolist()==UNIT_ORDER
 weak=[UNIT_ORDER.index(u)for u in ['Hogga_G1','Lio_G1','Haukeli_G1']]
 def auc(y,v):
  y=np.asarray(y).ravel();v=np.asarray(v).ravel();positive=y.astype(bool);n=int(positive.sum());negative=len(y)-n
  return None if n==0 or negative==0 else float((rankdata(v,method='average')[positive].sum()-n*(n+1)/2)/(n*negative))
 def compare(observed,expected):
  assert set(observed)==set(expected)
  for key,value in expected.items():
   if isinstance(value,dict):compare(observed[key],value)
   elif value is None:assert observed[key]is None
   elif isinstance(value,list):assert observed[key]==value
   else:np.testing.assert_allclose(observed[key],value,rtol=0,atol=2e-12,err_msg=key)
 derived={}
 for kind in ['continuous','three_binary','all14_binary']:
  physical=[arrays(root/p['cache']/f'case_{i:04d}_result.npz')if kind=='continuous'else arrays(root/p['comparison']/f'case_{i:04d}_integer.npz')if kind=='three_binary'else arrays(run/f'case_{i:04d}'/'result.npz')for i in ids]
  cases=[arrays(run/f'case_{i:04d}_inputs.npz')for i in ids]
  q=np.asarray([r['flow']/c['qmin'][:,None]for r,c in zip(physical,cases)]);power=np.asarray([r['power']/c['pmin'][:,None]for r,c in zip(physical,cases)]);states=np.asarray([r['state']for r in physical])
  for suffix,value in [('qratio',q),('pratio',power),('state',states)]:assert saved[kind+'_'+suffix].dtype==value.dtype and saved[kind+'_'+suffix].shape==value.shape and saved[kind+'_'+suffix].tobytes()==value.tobytes()
  qproxy=np.maximum(q,0)/(1+np.maximum(q,0));pproxy=np.maximum(power,0)/(1+np.maximum(power,0));hard=power>=.5
  metric={'micro_auc_qratio':auc(truth,qproxy),'micro_auc_pratio':auc(truth,pproxy),'weak3_micro_auc_qratio':auc(truth[:,weak],qproxy[:,weak]),'power_cutoff_hour_errors':int(np.sum(hard!=truth)),'power_cutoff_error_fraction':float(np.mean(hard!=truth)),
   'mean_absolute_case_unit_onfraction_error':float(np.abs(hard.mean(axis=-1)-truth.mean(axis=-1)).mean()),
   'mean_absolute_case_unit_switch_count_error':float(np.abs(np.sum(np.diff(hard.astype(int),axis=-1)!=0,axis=-1)-np.sum(np.diff(truth.astype(int),axis=-1)!=0,axis=-1)).mean()),'per_unit':{},'per_date_block':{},
   'weak3_power_cutoff_hour_errors':int(np.sum(hard[:,weak]!=truth[:,weak])),'power_cutoff_on_hours':int(hard.sum()),'weak3_power_cutoff_on_hours':int(hard[:,weak].sum()),'case_unit_power_cutoff_on_hours':hard.sum(axis=-1).tolist(),
   'state_cutoff_on_hours':int((states>=.5).sum()),'state_cutoff_hour_errors':int(np.sum((states>=.5)!=truth))}
  for j,unit in enumerate(UNIT_ORDER):metric['per_unit'][unit]={'auc_qratio':auc(truth[:,j],qproxy[:,j]),'auc_pratio':auc(truth[:,j],pproxy[:,j]),'hour_errors':int(np.sum(hard[:,j]!=truth[:,j]))}
  for block in dict.fromkeys(p['blocks']):
   selected=np.asarray(p['blocks'])==block;metric['per_date_block'][block]={'auc_qratio':auc(truth[selected],qproxy[selected]),'hour_errors':int(np.sum(hard[selected]!=truth[selected]))}
  compare(m['metrics'][kind],metric);derived[kind]=metric
 assert m['target_on_hours']==int(truth.sum())and m['target_weak3_on_hours']==int(truth[:,weak].sum())and m['target_case_unit_on_hours']==truth.sum(axis=-1).tolist()
 report.update(labels_decoded=True,selected_target_case_ids=ids,independent_rank_sum_metrics=derived,all_diagnostic_arrays_byte_exact=True,physical_lock_precedes_labels=True)
 report.update(status='passed',result_manifest_sha256=sha(run/'manifest.json'),protocol_sha256=sha(run/'protocol.json'),
  caveat='Descriptive reused12case proxy metrics do not establish fullvalidation predictive gain or a fullcache decision. Native MILP bounds provide global certificate; fixedLP primal/dual KKT independently checked without solving.')
except BaseException:report.update(status='failed',error=traceback.format_exc());raise
finally:
 shutil.copy2(__file__,out/Path(__file__).name)
 report.update(runtime_seconds=time.monotonic()-start,audit_source_sha256=sha(Path(__file__)),command=' '.join([sys.executable,*sys.argv]),environment={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__})
 (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'proof_sha256':sha(out/'manifest.json'),'case_count':len(report['cases']),'seconds':report['runtime_seconds']}))
