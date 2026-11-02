"""Re-run the narrow scheduling-context checks without fitting or writing artifacts.

Run from the repository root: .venv/bin/python reports/round10_water_constraint_checks.py
"""
import sys,json,hashlib,datetime
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,'src')
from scheduling_context import shared_river_constraint_context,distinct_inflow_timing
from features import load_data
from round5_lightgbm import generator_features,training_rows
checks=[]
def ok(name):checks.append(name)
st=pd.DatetimeIndex(['2020-01-01','2020-01-08'],tz='UTC');ts=pd.DatetimeIndex(np.repeat(st.array,168))+pd.to_timedelta(np.tile(np.arange(168),2),unit='h');r=np.full(336,5.);r[0]=14;r[167]=9
flow=np.zeros(336);flow[[0,168+167]]=100;inputs={'min_flow':pd.DataFrame({'r_Vest_Vassdraget':r},index=ts),'inflow':pd.DataFrame({'Staavatn':flow,'Venemo':flow[::-1]},index=ts)}
z=shared_river_constraint_context(st,ts,inputs);wait=z['shared_river_hours_until_next_change_or_end'].reshape(2,168);delta=z['shared_river_next_change_delta_m3_s'].reshape(2,168);rem=z['shared_river_remaining_required_release_million_m3'].reshape(2,168)
np.testing.assert_array_equal(wait[0,[0,1,166,167]],[1,166,1,1]);np.testing.assert_array_equal(delta[0,[0,1,166,167]],[-9,4,4,0]);np.testing.assert_array_equal(wait[1],np.arange(168,0,-1));np.testing.assert_array_equal(delta[1],0);np.testing.assert_allclose(rem[0,0],r[:168].sum()*.0036);np.testing.assert_allclose(rem[0,-1],9*.0036);ok('First-hour and last-hour changes, multiple changes, constant requirement, horizon censoring and signed deltas verified independently.')
f=distinct_inflow_timing(st,ts,inputs)
for n in ['Staavatn','Venemo']:
 hourly=f[f'distinct_natural_inflow_{n}_m3_s'].reshape(2,168);cum=f[f'distinct_cumulative_natural_inflow_{n}_million_m3'].reshape(2,168);remain=f[f'distinct_remaining_natural_inflow_{n}_million_m3'].reshape(2,168);np.testing.assert_allclose(cum[:,0],hourly[:,0]*.0036);np.testing.assert_allclose(remain[:,-1],hourly[:,-1]*.0036);np.testing.assert_allclose(cum+remain-hourly*.0036,np.repeat(hourly.sum(axis=1)[:,None]*.0036,168,axis=1));assert not np.array_equal(cum[0],cum[1])
ok('Equal-volume different-timing inflows remain distinguishable; cumulative/remaining current-hour convention and mass identities verified.')
dup_st=pd.DatetimeIndex([st[0],st[0]]);dup_ts=pd.DatetimeIndex(np.repeat(dup_st.array,168))+pd.to_timedelta(np.tile(np.arange(168),2),unit='h');d=shared_river_constraint_context(dup_st,dup_ts,inputs)
for v in d.values():np.testing.assert_array_equal(v[:168],v[168:])
ok('Duplicate requested timestamps retain separate case ordering.')
for wrong in [ts[:-1],ts[::-1]]:
 try:shared_river_constraint_context(st,wrong,inputs)
 except ValueError:pass
 else:raise AssertionError('Invalid horizon accepted')
ok('Partial and incorrectly ordered weeks rejected.')
cases,cols,gens,inp,t=load_data(True);c=cases[cases['Run No'].isin([2438,2871,2872])].reset_index(drop=True);st=c.starttime;ts=pd.DatetimeIndex(np.repeat(st.array,168))+pd.to_timedelta(np.tile(np.arange(168),len(c)),unit='h');q=shared_river_constraint_context(st,ts,inp);real={}
for i,row in c.iterrows():
 if int(row['Run No']) not in [2871,2872]:continue
 cid=int(row['Run No']);j=i*168;expected={2871:(144,7.6896),2872:(120,6.912)}[cid];assert q['shared_river_hours_until_next_change_or_end'][j]==expected[0];np.testing.assert_allclose(q['shared_river_remaining_required_release_million_m3'][j],expected[1]);assert q['shared_river_next_change_delta_m3_s'][j]==-9;real[str(cid)]={'time_to_change':float(q['shared_river_hours_until_next_change_or_end'][j]),'required_Mm3':float(q['shared_river_remaining_required_release_million_m3'][j])}
ok('Real r_Vest known changes and exact release volumes match independent case calculations.')
base=json.loads(Path('configs/round5_lgb_peer_relative_seed299_2021.json').read_text());x=generator_features(c,'Lio_G1',gens.index('Lio_G1'),inp,t,base);assert len(x.columns)==113
for arm,flag,extra in [('constraint','use_shared_river_constraint_timing',3),('water','use_distinct_inflow_timing',6)]:
 false={**base,flag:False};xf=generator_features(c,'Lio_G1',gens.index('Lio_G1'),inp,t,false);pd.testing.assert_frame_equal(x,xf,check_exact=True);true={**base,flag:True};xt=generator_features(c,'Lio_G1',gens.index('Lio_G1'),inp,t,true);pd.testing.assert_frame_equal(x,xt[x.columns],check_exact=True);assert len(xt.columns)==113+extra;assert np.isfinite(xt.to_numpy()).all();rows=training_rows(c,4,299);assert len(rows)==len(c)*42;np.testing.assert_array_equal(xt.iloc[rows].to_numpy(),xt.to_numpy()[rows]);ok(f'{arm}: default false preserves all113 columns bit-exactly; enabled adds only{extra} finite columns, original values unchanged. Full weeks built before exact existing stride4 selection.')
paths=['src/scheduling_context.py','src/round5_lightgbm.py','src/features.py','src/tree_preprocessing.py','configs/round10_water_seed299_2021.json','configs/round10_water_seed299_2022.json','configs/round10_constraint_seed299_2021.json','configs/round10_constraint_seed299_2022.json','reports/round10_water_protocol.json','reports/round10_constraint_protocol.json'];result={'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':checks,'real_constraint_examples':real,'source_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},'training_performed':False}
print(json.dumps(result,indent=2))
