"""Freeze the authorized LightGBM664 ablation without fitting or reading target values."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_lgb_shop664 import sha,environment,WEIGHTS


def main()->None:
    baseline_path=ROOT/'artifacts/lgb-normalized-lp-context-joint-001/manifest.json'
    baseline=json.loads(baseline_path.read_text())
    loader_path=ROOT/'docs/cuda_xgb664_protocol.json';loader=json.loads(loader_path.read_text())
    assert sha(loader_path)=='4e09f0fe59973d7a9c2c5e5792feb30c29b8af04a67c27bafbe1386ae9da8182'
    assert sha(baseline_path)==loader['baseline_manifest_sha256']
    sources=['scripts/train_lgb_shop664.py','scripts/prepare_lgb_shop664.py','tests/test_lgb_shop664.py','docs/lgb_shop664.md',
             'scripts/cuda_xgb664_loader.py','scripts/cuda_xgb608_loader.py','scripts/train_cuda_tcn.py',
             'sintef_project/postprocess.py','sintef_project/lp_shop_features.py']
    sources+=list(loader['feature_source_sha256'])
    inputs={str(baseline_path.relative_to(ROOT)):sha(baseline_path),str(loader_path.relative_to(ROOT)):sha(loader_path)}
    prerequisites={}
    for name in ['artifacts/cuda-xgb664-package-feature-independent-audit-001/manifest.json',
                 'artifacts/cuda-xgb664-select-2021-independent-audit-002/manifest.json',
                 'artifacts/cuda-xgb664-select-2022-independent-audit-001/manifest.json']:
        proof=json.loads((ROOT/name).read_text());assert proof['status']=='passed'
        inputs[name]=prerequisites[name]=sha(ROOT/name)
    stage_files={}
    for folder,expected in [('artifacts/kaggle-neural-bundle-001/payload',loader['bundle_manifest_sha256']),
                            ('artifacts/cuda-xgb608-package-001/supplement',loader['supplement_manifest_sha256']),
                            ('artifacts/cuda-xgb664-package-001/shop',loader['shop_bundle_manifest_sha256'])]:
        manifest_path=ROOT/folder/'bundle_manifest.json';assert sha(manifest_path)==expected
        inputs[str(manifest_path.relative_to(ROOT))]=expected
        for entry in json.loads(manifest_path.read_text())['payload_files']:
            assert entry['path']not in stage_files
            stage_files[entry['path']]={'sha256':entry['sha256'],'size_bytes':entry['size_bytes']}
    anchors={};controls={};samples={};folds={}
    for year in ['2021','2022']:
        for destination,directory,name in [(anchors,f'artifacts/cuda-xgb664-select-{year}-001',f'selected_validation_{year}.npz'),
                                          (controls,'artifacts/lgb-normalized-lp-context-joint-001',f'validation_{year}.npz')]:
            mp=ROOT/directory/'manifest.json';prediction=ROOT/directory/name;m=json.loads(mp.read_text())
            assert m['status']=='complete' and m['artifacts'][name]==sha(prediction)
            destination[year]={'manifest':str(mp.relative_to(ROOT)),'manifest_sha256':sha(mp),
                               'prediction':str(prediction.relative_to(ROOT)),'prediction_sha256':sha(prediction)}
            inputs[str(mp.relative_to(ROOT))]=sha(mp);inputs[str(prediction.relative_to(ROOT))]=sha(prediction)
        sample=baseline_path.parent/f'train_sample_{year}.npy'
        assert sha(sample)==baseline['artifacts'][sample.name]
        samples[year]={'path':str(sample.relative_to(ROOT)),'sha256':sha(sample)};inputs[str(sample.relative_to(ROOT))]=sha(sample)
        folds[year]={k:baseline['folds'][year][k]for k in ['train_ids','validation_ids','train_dates','validation_dates','purge_days','sampled_training_rows']}
        assert folds[year]==loader['folds'][year]
    params=dict(baseline['model_parameters']);params['n_jobs']=2
    protocol={'version':'lgb_shop664_feature_ablation_v1','created_utc':datetime.now(timezone.utc).isoformat(),
              'source':{name:sha(ROOT/name)for name in sources},'inputs':inputs,'data':baseline['data'],
              'environment':environment(),'baseline_manifest':str(baseline_path.relative_to(ROOT)),
              'loader_protocol':str(loader_path.relative_to(ROOT)),
              'stage':'artifacts/cuda-xgb664-package-feature-independent-audit-001/stage/merged664','stage_files':stage_files,
              'independent_prerequisites':prerequisites,'model_parameters':params,'samples':samples,'folds':folds,
              'feature_columns':loader['feature_columns'],'units':loader['units'],'anchor':anchors,'control':controls,
              'candidate_weights':WEIGHTS,'formula':'project_vinje((1-w)*saved_acceptedSHOP664blend+w*rawLGB664),float64',
              'gate':'2021 positive selected weight strictly above anchor and matched rawLGB608 at same weight; lower weight ties. Independently lock before2022, evaluating only0+lockedweight; strict same gates.',
              'resources':{'model_cpu_threads':2,'expected_peak_upper_estimate_bytes':8*1024**3,
                           'previous_real_feature_build_peak_rss_bytes':json.loads((ROOT/'artifacts/cuda-xgb664-package-feature-independent-audit-001/manifest.json').read_text())['peak_rss_bytes'],
                           'minimum_free_percent_before_build':40.,'minimum_free_percent_before_unit':20.,
                           'coexistence':'One2thread model job alongside independently controlled4singlethread physics workers; explicit resource coordination before fit.'},
              'early_stopping':False,'final_training':False,'reused_development_years':True,
              'ablation':'Exactly608+56 frozenSHOP columns; all original350tree31leaf recipe/samples except authorized n_jobs3to2.'}
    path=ROOT/'docs/lgb_shop664_protocol.json'
    with path.open('x')as f:json.dump(protocol,f,indent=2)
    print(sha(path),flush=True)


if __name__=='__main__':main()
