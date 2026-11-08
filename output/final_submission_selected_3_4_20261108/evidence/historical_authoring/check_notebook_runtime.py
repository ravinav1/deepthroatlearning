from pathlib import Path
import json, os, tempfile, sys
B=Path('/Users/heap/deepthroatlearning');O=Path(__file__).resolve().parents[1]
# Execute the actual notebook's setup and embedded module-writing cells in isolation.
nb=json.load(open(O/'canvas/Short notebook 1.ipynb'));env={}
setup=''.join(nb['cells'][1]['source']).replace('Path("sintef_project").resolve()',repr(str(B/'experiments/hydraulic_surrogate/sintef_project')))
setup=setup.replace('RAW_DATA = '+repr(str(B/'experiments/hydraulic_surrogate/sintef_project')),'RAW_DATA = Path('+repr(str(B/'experiments/hydraulic_surrogate/sintef_project'))+')')
os.chdir(O/'qa');exec(compile(setup,'notebook setup','exec'),env)
for c in nb['cells']:
 s=''.join(c['source'])
 if c['cell_type']=='code' and s.startswith('module_source ='):exec(compile(s,'embedded code cell','exec'),env)
# Check runtime imports using the installed CPU environment; CUDA is intentionally not trained.
sections=json.load(open(O/'authoring/runtime_sections.json'));exec(sections['imports'].split('assert xgb.__version__')[0],env)
import numpy as np,pandas as pd, lightgbm as lgb
cfg=json.load(open(B/'configs/hydraulic_surrogate_best.json'));a=B/'models/hydraulic_surrogate'
for name,ncols in [('lp',78),('joint',87),('shop',56)]:
 z=np.load(a/('lp-full-001' if name=='lp' else 'lp-joint-full-001' if name=='joint' else 'lp-shop-full-001')/'features.npz')
 env[name+'_frame']=pd.DataFrame(z['features'].reshape(-1,ncols),columns=z['columns'])
exec(sections['features'],env)
checks={}
for c in cfg['components']:
 m=json.load(open(a/c['directory']/'manifest.json'));f=env['feature_table'](c['name']);assert f.columns.tolist()==m['features'];checks[c['name']]={'columns':f.shape[1],'rows':len(f)}
 if c['name']=='lp':
  # One full-size 350-tree unit fit checks sampled-row, feature and training compatibility.
  params=m['model_parameters'];model=lgb.LGBMClassifier(**params)
  model.fit(f.iloc[env['sample']],env['Y'][env['sample'],0]);pred=model.predict_proba(f.iloc[env['NTRAIN']:])[:,1].astype(np.float32)
  original=np.load(a/c['directory']/'test_predictions.npy')[:,0]
  checks['retrained_lp_Hogga_G1_max_abs_error']=float(np.max(np.abs(pred-original)))
  print('LightGBM fresh fit error',checks['retrained_lp_Hogga_G1_max_abs_error'],flush=True)
 if c['name']=='lgb664':
  # Check the embedded pooled generator-relative transform and saved checkpoint prediction.
  spec=env['feature_specification'](f.columns.tolist(),env['UNITS']);top=env['yaml'].safe_load((env['PKG']/'data/extended/Tokke_Vinje_topology.yaml').read_text())['model']['generator'];ref=max(float(top[u]['p_max']) for u in env['UNITS'])
  const={u:dict(pmin=float(top[u]['p_min']),pmax=float(top[u]['p_max']),start=float(top[u]['startcost_const']),stop=float(top[u]['stopcost_const']),ref=ref) for u in env['UNITS']}
  rows=np.arange(env['NTRAIN'],env['NTRAIN']+1000);x=env['build_rows'](f.astype(np.float32),rows,spec,const)
  for n in ['startup16_final','startup16_final_seed1']:
   model=lgb.Booster(model_file=str(B/'models/supplied_only'/n/'model.txt'));p=model.predict(x,num_threads=4).reshape(-1,14);expected=np.load(B/'models/supplied_only'/n/'raw_predictions.npy')[:1000]
   np.testing.assert_array_equal(p,expected);checks[n+'_14000_predictions_exact']=True
 print('Verified schema',c['name'],f.shape,flush=True)
audit=json.load(open(O/'evidence/verification.json'));audit['checks']['executed_notebook_runtime']=checks
(O/'evidence/verification.json').write_text(json.dumps(audit,indent=2));print(json.dumps(checks,indent=2))
