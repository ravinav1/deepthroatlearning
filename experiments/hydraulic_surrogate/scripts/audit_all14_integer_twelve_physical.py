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
out=root/'artifacts/all14-integer-water-budget-twelve-physical-independent-audit-001';out.mkdir(exist_ok=False);run=root/'artifacts/all14-integer-water-budget-twelve-001'
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
 assert m['status']=='complete'and not m['physical_feasibility_passed']and not m['labels_decoded']
 assert 'metrics'not in m and 'labels_opened_utc'not in m and not(run/'diagnostic.npz').exists()
 for name,h in m['artifacts'].items():assert sha(run/name)==h,name
 for category in ['source','data','inputs']:
  for name,h in p[category].items():assert sha(root/name)==h,name
 assert len(m['cases'])==12
 assert len({r['pid']for r in m['cases']if r['case_id']not in [1,1826]})==10
 original_selection=json.loads((root/'docs/integer_water_budget_protocol.json').read_text())
 assert [r['case_id']for r in m['cases']]==original_selection['case_ids']and p['blocks']==original_selection['blocks']
 lock=json.loads((run/'physical_lock.json').read_text());assert sha(run/'physical_lock.json')==m['physical_lock_sha256']and lock['physical_gates']==m['physical_gates']
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
  assert proof['mip_status']in[0,1]and proof['proven_optimal']==(proof['mip_status']==0)and proof['mip_rel_gap_requested']==0
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
  report['cases'].append({'case_id':case_id,'mip_status':proof['mip_status'],'proven_optimal':proof['proven_optimal'],'physical_incumbent_max_violation':violation,'fixed_lp_stationarity_max':stationarity,
    'fixed_lp_complementarity_max':complementarity,'native_relative_gap':proof['mip_gap'],'native_absolute_gap_eur':record['native_absolute_gap_eur'],
    'conditional_lp_objective_gap_eur':proof['objective_gap_eur'],'seconds':record['runtime_seconds'],'peak_rss_bytes':record['peak_rss_bytes'],
    'objective_loss_vs_three_unit_eur':record['objective_loss_vs_three_unit_eur'],
    'unit_state_changed_hours_vs_three_unit':{u:int(np.count_nonzero(result['state'][j]!=three['state'][j]))for j,u in enumerate(UNIT_ORDER)}})
 expected_gates={'all_fixed_cases':True,'all_proven':all(r['proven_optimal']for r in report['cases']),'all_lp_replays':True,'all_zero_slacks':True,'all_subset_objectives':True,'all_rss_below_2gib':True}
 assert expected_gates==m['physical_gates']
 unproven=[r['case_id']for r in report['cases']if not r['proven_optimal']]
 assert unproven==[2186]
 r=next(r for r in m['cases']if r['case_id']==2186);proof=r['proof']
 gap=(proof['mip_objective_scaled']-proof['mip_dual_bound_scaled'])/abs(proof['mip_objective_scaled'])
 np.testing.assert_allclose(gap,proof['mip_gap'],rtol=1e-9,atol=1e-15)
 report.update(physical_feasibility_passed=False,proven_case_count=11,unproven_case_ids=unproven,all_nonoptimality_physical_gates_passed=True,no_labels_or_scored_subset=True,original_physical_gates=expected_gates)
 report.update(status='passed',result_manifest_sha256=sha(run/'manifest.json'),protocol_sha256=sha(run/'protocol.json'),
  caveat='Original60second screen failed native optimality on case2186; its feasible incumbent is not promoted. No labels opened. A separately authorized600second proof-only completion is a new frozen execution; original evidence remains immutable.')
except BaseException:report.update(status='failed',error=traceback.format_exc());raise
finally:
 shutil.copy2(__file__,out/Path(__file__).name)
 report.update(runtime_seconds=time.monotonic()-start,audit_source_sha256=sha(Path(__file__)),command=' '.join([sys.executable,*sys.argv]),environment={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__})
 (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'proof_sha256':sha(out/'manifest.json'),'case_count':len(report['cases']),'seconds':report['runtime_seconds']}))
