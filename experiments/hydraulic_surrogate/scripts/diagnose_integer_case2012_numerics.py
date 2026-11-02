"""Read-only native incumbent versus conditional LP diagnosis; zero optimization."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.diagnose_integer_water_budget import arrays,digest
from scripts.probe_all14_integer_water_budget_v2 import identical_arrays
from sintef_project.lp_features import HydroCaseData,UNIT_ORDER
from sintef_project.lp_hyljelihyl_v2 import capture_problem,kkt,feasibility


def main():
    out=ROOT/'artifacts/integer-case2012-numerical-diagnostic-001';out.mkdir(exist_ok=False)
    started=time.monotonic();report={'status':'initializing','timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'optimization_calls':0,'labels_decoded':False,'case_id':2012,'replicas':[]}
    target=ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv';original=pd.read_csv
    def guarded(path,*args,**kwargs):
        if isinstance(path,(str,Path))and Path(path).resolve()==target:assert not args and kwargs=={'usecols':['Run No','starttime']}
        return original(path,*args,**kwargs)
    with patch.object(pd,'read_csv',guarded):data=HydroCaseData(ROOT)
    case_path=ROOT/'artifacts/lp-shop-full-001/case_2012_inputs.npz';case=arrays(case_path)
    rebuilt=data.case(2011);rebuilt['natural']=rebuilt['natural'].astype(np.float64)
    for key,field in [('pmin','p_min'),('start_cost','startcost_const'),('stop_cost','stopcost_const')]:
        rebuilt[key]=np.array([data.model['generator'][u][field]for u in UNIT_ORDER],dtype=np.float64)
    assert identical_arrays(case,rebuilt)
    model,problem=capture_problem(data.network,case,1.);scale=float(1+np.abs(case['price']).mean())
    selected=model.state[[0,5,10]].ravel();report['scale']=scale;report['input_sha256']=digest(case_path)
    source=ROOT/'artifacts/integer-water-budget-full-four-worker-001/cases/case_2012'
    all_native=[];all_lp=[]
    variable_names={}
    for owner,names in [(model.base,['q','v','bypass','spill','low_slack','high_slack','bypass_slack','vest_slack']),
                        (model,['state','start','stop'])]:
        for name in names:
            indices=getattr(owner,name)
            for coords in np.ndindex(indices.shape):variable_names[int(indices[coords])]=name+str(coords)
    def row_name(index):
        t=168;r=9;b=10;base_rows=model.base.a_ub.shape[0]
        if index<r*t:return 'minimum_volume_'+data.network.pools[index//t]+'_'+str(index%t)
        if index<2*r*t:
            k=index-r*t;return 'maximum_volume_'+data.network.pools[k//t]+'_'+str(k%t)
        if index<(2*r+b)*t:
            k=index-2*r*t;return 'minimum_bypass_'+str(k//t)+'_'+str(k%t)
        if index<base_rows:return 'vest_minimum_'+str(index-(2*r+b)*t)
        k=index-base_rows
        if k<14*t*4:return UNIT_ORDER[k//(t*4)]+'_'+['lower_P','upper_P','lower_Q','upper_Q'][k%4]+'_'+str((k%(t*4))//4)
        k-=14*t*4;return 'transition_'+UNIT_ORDER[k//(2*(t-1))]+'_'+str(k%(2*(t-1)))
    for replica in [1,2]:
        folder=source/f'replica_{replica}_60s';record=json.loads((folder/'proof.json').read_text())
        assert record['status']=='failed'and record['input_sha256']==digest(case_path)
        for name,h in record['artifacts'].items():assert digest(folder/name)==h
        native=arrays(folder/'failed_incumbent.npz');lp=arrays(folder/'fixed_lp_1.npz')
        assert identical_arrays(lp,arrays(folder/'fixed_lp_2.npz'));all_native.append(native);all_lp.append(lp)
        xi=native['integer_solution'];states=native['integer_states'].ravel();x=lp['solution']
        fixed={**problem,'bounds':problem['bounds'].copy()};fixed['bounds'][selected,0]=states;fixed['bounds'][selected,1]=states
        saved={**lp,'x':x};certificate=kkt(fixed,saved)
        c=problem['c'];eq=problem['A_eq']@xi-problem['b_eq'];ub=problem['A_ub']@xi-problem['b_ub']
        lower=xi-fixed['bounds'][:,0];finite=np.isfinite(fixed['bounds'][:,1]);upper=xi[finite]-fixed['bounds'][finite,1]
        contributions={'equality':lp['eqlin_marginals']*eq,'inequality':lp['ineqlin_marginals']*ub,
            'lower_bound':lp['lower_marginals']*lower,'upper_bound':lp['upper_marginals'][finite]*upper}
        details=[]
        for kind,values in contributions.items():
            for i in np.argsort(np.abs(values))[-8:]:
                index=int(np.flatnonzero(finite)[i])if kind=='upper_bound'else int(i)
                name=(data.network.pools[index//168]+'_'+str(index%168)if kind=='equality'else
                      row_name(index)if kind=='inequality'else variable_names[index])
                details.append({'kind':kind,'index':index,'name':name,'native_minus_lp_scaled':float(values[i]),'eur':float(values[i]*scale)})
        details=sorted(details,key=lambda r:abs(r['native_minus_lp_scaled']),reverse=True)
        delta=float(lp['objective']-record['proof']['mip_objective_scaled'])
        report['replicas'].append({'replica':replica,'proof_sha256':digest(folder/'proof.json'),
            'native_objective_scaled':record['proof']['mip_objective_scaled'],'native_bound_scaled':record['proof']['mip_dual_bound_scaled'],
            'fixed_lp_objective_scaled':float(lp['objective']),'lp_minus_native_scaled':delta,'lp_minus_native_eur':delta*scale,
            'native_dot_objective_scaled':float(c@xi),'fixed_dot_objective_scaled':float(c@x),
            'native_original_feasibility':feasibility(problem,xi),'native_fixed_feasibility':feasibility(fixed,xi),
            'integrality_max_error':float(np.max(np.abs(xi[selected]-states))),'conditional_lp_kkt':certificate,
            'conditional_replay_byte_exact':True,'decomposition_scaled_sums':{k:float(v.sum())for k,v in contributions.items()},
            'decomposition_scaled_total':float(sum(v.sum()for v in contributions.values())),
            'top_native_minus_lp_contributions':details[:16],
            'violated_inequalities':[{'index':int(i),'name':row_name(int(i)),'violation':float(ub[i]),'dual':float(lp['ineqlin_marginals'][i])}
                for i in np.flatnonzero(ub>1e-10)],
            'violated_lower_bounds':[{'index':int(i),'name':variable_names[int(i)],'violation':float(-lower[i]),'dual':float(lp['lower_marginals'][i])}
                for i in np.flatnonzero(lower < -1e-10)]})
    assert identical_arrays(all_native[0],all_native[1])and identical_arrays(all_lp[0],all_lp[1])
    report.update(status='complete_read_only',replicas_byte_identical=True,runtime_seconds=time.monotonic()-started,
        source_sha256=digest(Path(__file__)),interpretation='Original failure remains closed; this diagnosis changes no tolerance, source, solve, retry or eligibility.')
    shutil_source=out/'source.py';shutil_source.write_bytes(Path(__file__).read_bytes())
    (out/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
