from pathlib import Path
import sys, json, hashlib, time, tempfile
import numpy as np, pandas as pd, lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path('/Users/heap/deepthroatlearning'); O=Path(__file__).resolve().parents[1]
R=B/'experiments/hydraulic_surrogate';sys.path.insert(0,str(R));sys.path.insert(0,str(B/'scripts/supplied_only'))
from sintef_project.postprocess import project_vinje
from blend import project_tokke
from sintef_project.lp_features import HydroCaseData, HydraulicLP,result_features, UNIT_ORDER
from sintef_project.lp_joint_plant import JointPlantLP,build_joint_curve,result_joint_features
from sintef_project.lp_shop_relaxation import ShopRelaxationLP
from sintef_project.lp_shop_features import compute_features
from sintef_project.features import build_features
from sintef_project.physics import add_physics_features
from sintef_project.context_features import add_context_features
from sintef_project.normalized import add_normalized_features
from sintef_project.lp_context import add_lp_context_features
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
audit={'verification_date':'2026-11-08','checks':{},'limitations':['No full raw-data notebook run with the historical CUDA environment','Combined runtime and peak memory are not measured','Live Kaggle final-selection state not accessible']}
# Recompose both final selected submissions from saved outputs independently of the exporter.
a=B/'models/hydraulic_surrogate'; cfg=json.load(open(B/'configs/hydraulic_surrogate_best.json'))
p={c['name']:(np.load(a/c['directory']/'test_predictions.npz')['predictions'] if c['name'].startswith('gpu') else np.load(a/c['directory']/'test_predictions.npy')).astype(np.float64) for c in cfg['components']}
anchor=project_vinje(sum(w*p[n] for n,w in cfg['base_weights'].items()))
for stage in cfg['nested_blends']:anchor=project_vinje((1-stage['weight'])*anchor+stage['weight']*p[stage['component']])
np.testing.assert_array_equal(anchor,np.load(a/'submission-lgb-shop664-anchor-blend-001/test_predictions.npy'))
audit['checks']['anchor_all_nested_projections_exact']=True
s0=np.load(B/'models/supplied_only/startup16_final/raw_predictions.npy');s1=np.load(B/'models/supplied_only/startup16_final_seed1/raw_predictions.npy')
template=pd.read_csv(B/'data/raw/sample_submission.csv')
for score,raw in [('0.99657',(s0+s1)/2),('0.99656',s0)]:
 out=project_tokke(project_vinje(.75*anchor+.25*raw))
 frame=pd.DataFrame(out.reshape(725,168,14).transpose(0,2,1).reshape(725,-1),columns=template.columns[1:]);frame.insert(0,'Run No',template['Run No'].to_numpy())
 text=frame.to_csv(index=False,float_format='%.9g');sha=hashlib.sha256(text.encode()).hexdigest()
 expected='f7b0b52fc54e93e1db6171fdf85a7955486c3439897911291eef98c70a4f716a' if score=='0.99657' else 'a45f8001a87afbc67b17d95fb39c9f1982593d24f91887d5747fec1f3c367ceb'
 assert sha==expected
 audit['checks']['csv_'+score]={'sha256':sha,'matches':True,'shape':list(frame.shape)}
print('Both selected CSVs recompose byte-for-byte',flush=True)
# Test all embedded module source independently.
sections=json.load(open(O/'authoring/runtime_sections.json'))
for n,s in sections['module_sources'].items():compile(s,n+'.py','exec')
audit['checks']['embedded_modules_compile']=len(sections['module_sources'])
# Execute fresh raw-data solves for representative early, late-training, and test cases.
data=HydroCaseData(R); solvers=(HydraulicLP(data.network),JointPlantLP(data.network),ShopRelaxationLP(data.network));checks=[]
caches={n:np.load(a/n/'features.npz') for n in ['lp-full-001','lp-joint-full-001','lp-shop-full-001']}
for i in [0,2921,2922]:
 case=data.case(i);old=solvers[0].solve(case,time_limit=20.0);f=result_features(case,old,data.network).to_numpy(dtype=np.float32)
 curves=[build_joint_curve(data,i,p) for p in ['Tokke','Vinje']];jr=solvers[1].solve(case,curves,time_limit=20.0);j=result_joint_features(case,jr,data.network).to_numpy(dtype=np.float32)
 case['natural']=np.asarray(case['natural'],dtype=np.float64)
 for name,field in [('pmin','p_min'),('start_cost','startcost_const'),('stop_cost','stopcost_const')]:case[name]=np.asarray([data.model['generator'][u][field] for u in UNIT_ORDER],dtype=np.float64)
 sr=solvers[2].solve(case,1.0);s=compute_features(case,sr,old,data.network)
 diffs={n:float(np.max(np.abs(v-caches[n]['features'][i]))) for n,v in zip(caches,[f,j,s])}
 checks.append({'case_id':int(data.cases.iloc[i]['Run No']),'max_abs_feature_differences':diffs})
 print('LP check',checks[-1],flush=True)
audit['checks']['raw_data_lp_case_checks']=checks
# Rebuild deterministic feature tables from raw covariates plus original LP caches.
base,labels,train,mapping,meta=build_features(R);base=add_physics_features(base,R)[0];base=add_context_features(base,R)[0]
lp=pd.DataFrame(caches['lp-full-001']['features'].reshape(-1,78),columns=caches['lp-full-001']['columns'])
joint=pd.DataFrame(caches['lp-joint-full-001']['features'].reshape(-1,87),columns=caches['lp-joint-full-001']['columns'])
shop=pd.DataFrame(caches['lp-shop-full-001']['features'].reshape(-1,56),columns=caches['lp-shop-full-001']['columns'])
f=pd.concat([base,lp],axis=1);f=add_lp_context_features(f,R)[0];f=add_normalized_features(f,R)[0];f=pd.concat([f,joint,shop],axis=1)
saved=np.load(B/'models/supplied_only/features664/features664.npy',mmap_mode='r');maxdiff=0
for start in range(0,len(f),5000):maxdiff=max(maxdiff,float(np.max(np.abs(f.iloc[start:start+5000].to_numpy(dtype=np.float32)-saved[start:start+5000]))))
assert maxdiff==0
audit['checks']['664_table_from_raw_covariates_and_archived_lp_exact']=True
sample=np.sort(np.random.default_rng(4173).choice(2922*168,size=int(2922*168*.4),replace=False))
np.testing.assert_array_equal(sample,np.load(a/'lgb-normalized-lp-context-joint-final-001/train_sample_final.npy'))
audit['checks']['training_sample_regenerates_exactly']=True
# Native gain importance, shared consistently with the selected two-seed startup member.
man=json.load(open(B/'models/supplied_only/startup16_final/manifest.json'))
importance=[]
for name in ['startup16_final','startup16_final_seed1']:
 model=lgb.Booster(model_file=str(B/'models/supplied_only'/name/'model.txt'));g=model.feature_importance(importance_type='gain');importance.append(g/g.sum())
imp=pd.Series(np.mean(importance,axis=0),index=man['columns']).sort_values(ascending=False)
imp.rename('mean_normalized_gain').to_csv(O/'evidence/feature_importance.csv')
fig,ax=plt.subplots(figsize=(7.4,4)); top=imp.head(12).iloc[::-1];ax.barh(top.index.str.replace('own_',''),100*top.values,color='#146b78');ax.set_xlabel('Mean normalized split gain (%)');fig.tight_layout();fig.savefig(O/'figures/importance.png',dpi=180);plt.close(fig)
# EDA computed from all supplied labels; grouping is by case start year.
y=np.column_stack([labels[u] for u in meta['units']]);years=pd.to_datetime(train.starttime).dt.year.to_numpy();eda={'n_train_cases':len(train),'n_test_cases':len(mapping),'n_units':14,'hours_per_case':168,'label_count':int(y.size),'positive_fraction':float(y.mean()),'units':{u:float(y[:,i].mean()) for i,u in enumerate(meta['units'])},'by_year':{str(year):float(y[np.repeat(years==year,168)].mean()) for year in sorted(set(years))},'top_gain_features':imp.head(12).to_dict()}
fig,ax=plt.subplots(figsize=(7.4,3.2));ax.bar(np.arange(14),y.mean(axis=0),color='#146b78');ax.set_xticks(np.arange(14),meta['units'],rotation=55,ha='right');ax.set_ylabel('Fraction ON');ax.set_ylim(0,1);fig.tight_layout();fig.savefig(O/'figures/prevalence.png',dpi=180);plt.close(fig)
for name in ['Historical_day_ahead_price_2015_2025.csv','Historical_inflow_1958_2025.csv','Historical_volume_2015_2024.csv','Synthetic_water_value_2015_2024.csv']:
 d=pd.read_csv(B/'data/raw/kernel'/name);eda[name]={'rows':len(d),'columns':len(d.columns),'missing_cells':int(d.isna().sum().sum()),'duplicate_dates':int(d['date'].duplicated().sum())}
price=f['norm_price_week_std'].to_numpy().reshape(-1,168)[:,0];all_year=pd.to_datetime(pd.concat([train,mapping]).starttime).dt.year.to_numpy();eda['relative_price_std_median']={k:float(np.median(price[mask])) for k,mask in {'2015-2020':all_year<=2020,'2021':all_year==2021,'2022':all_year==2022,'2023-2024':all_year>=2023}.items()}
# Sampled, descriptive correlations; not causal importance.
eda['sampled_pair_correlations']={u:float(np.corrcoef(f[f'lp_qmin_ratio_{u}'].iloc[sample],y[sample,j])[0,1]) for j,u in enumerate(meta['units'])}
(O/'evidence/eda.json').write_text(json.dumps(eda,indent=2));(O/'evidence/verification.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2));print(json.dumps(eda,indent=2))
