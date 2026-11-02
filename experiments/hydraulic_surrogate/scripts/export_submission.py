"""Apply a validation-selected transform and export a checked competition CSV."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sintef_project.postprocess import smooth_predictions, project_vinje


def sha(path: Path) -> str:
    """SHA256 of an artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-run', required=True)
    parser.add_argument('--selection-run', required=True)
    parser.add_argument('--validation-model', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    started=time.time()
    source=ROOT/'artifacts'/args.model_run
    selection_path=ROOT/'artifacts'/args.selection_run/'selection_2021.json'
    selection=json.loads(selection_path.read_text())
    selected=selection['results'][args.validation_model]['selected_config']
    model_manifest=json.loads((source/'manifest.json').read_text())
    if model_manifest.get('status') != 'complete':
        raise ValueError('Training has not completed')
    validation_source=ROOT/'artifacts'/args.validation_model
    validation=json.loads((validation_source/'manifest.json').read_text())
    if validation.get('status') != 'complete' or set(validation['folds']) != {'2021', '2022'}:
        raise ValueError('Expected both completed chronological validation folds')
    if (model_manifest['features'] != validation['features']
            or model_manifest['feature_spec'] != validation['feature_spec']
            or model_manifest['data'] != validation['data']):
        raise ValueError('Final and validation feature/data recipes differ')
    for name in ['trees', 'leaves', 'learning_rate', 'min_child_samples', 'sample_fraction', 'seed', 'feature_set']:
        if model_manifest['arguments'][name] != validation['arguments'][name]:
            raise ValueError(f'Final training recipe differs: {name}')
    for name, value in validation['model_parameters'].items():
        if name != 'n_jobs' and model_manifest['model_parameters'].get(name) != value:
            raise ValueError(f'Final model parameter differs: {name}')
    for manifest, directory in [(model_manifest, source), (validation, validation_source)]:
        for path, expected in manifest['artifacts'].items():
            if sha(directory/path) != expected:
                raise ValueError(f'Artifact changed: {directory/path}')
    if set(selection['results'][args.validation_model]['folds']) != {'2021'}:
        raise ValueError('Postprocessing selection must be locked before reading 2022')
    for path, expected in model_manifest['data'].items():
        if sha(ROOT/path) != expected:
            raise ValueError(f'Input changed: {path}')
    pp_source=ROOT/'sintef_project/postprocess.py'
    if sha(pp_source) != selection['source_snapshots']['sintef_project/postprocess.py']:
        raise ValueError('Postprocessing source differs from selected experiment')
    template=pd.read_csv(ROOT/'sintef_project/sample_submission.csv')
    mapping=pd.read_csv(ROOT/'sintef_project/prediction_mapping.csv')
    units=[c.removeprefix('result_committed_').removesuffix('_t0') for c in template.columns if c.endswith('_t0')]
    p=np.load(source/'test_predictions.npy')
    assert p.shape==(len(mapping)*168,len(units))
    p=smooth_predictions(p,selected['window'],selected['weight'])
    if selected['projection']:
        p=project_vinje(p,units)
    cube=p.reshape(len(mapping),168,len(units))
    data={}
    for column in template.columns[1:]:
        unit,hour=column.removeprefix('result_committed_').rsplit('_t',1)
        data[column]=cube[:,int(hour),units.index(unit)]
    submission=pd.DataFrame(data,index=mapping['Run No']).loc[template['Run No']].reset_index()
    assert submission.shape==template.shape and list(submission)==list(template)
    assert np.isfinite(submission.iloc[:,1:]).all().all()
    assert submission.iloc[:,1:].min().min()>=0 and submission.iloc[:,1:].max().max()<=1
    out=ROOT/'artifacts'/args.run_id
    out.mkdir(exist_ok=False)
    destination=out/'submission.csv'
    submission.to_csv(destination,index=False,float_format='%.9g')
    loaded=pd.read_csv(destination)
    assert list(loaded)==list(template) and loaded['Run No'].equals(template['Run No'])
    assert np.max(np.abs(loaded.iloc[:,1:].to_numpy()-submission.iloc[:,1:].to_numpy()))<=5.1e-10
    manifest={'timestamp_utc':datetime.now(timezone.utc).isoformat(), 'arguments':vars(args),
              'selected_postprocess':selected,'model_manifest':str(source/'manifest.json'),
              'model_manifest_sha256':sha(source/'manifest.json'), 'selection_manifest':str(selection_path),
              'validation_manifest_sha256':sha(validation_source/'manifest.json'),
              'final_validation_recipe_match':True, 'registered_artifacts_verified':True,
              'selection_sha256':sha(selection_path),'raw_predictions_sha256':sha(source/'test_predictions.npy'),
              'submission':str(destination),'submission_sha256':sha(destination),
              'shape':list(submission.shape),'range':[float(p.min()),float(p.max())],
              'runtime_seconds':time.time()-started,
              'kaggle':{'submission_id':None,'score':None,'rank':None,'observation_time':None}}
    for path in [Path(__file__),pp_source]:
        shutil.copy2(path,out/path.name)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))
    with (ROOT/'docs/experiments.md').open('a') as f:
        f.write(f'\n## {args.run_id} — {manifest["timestamp_utc"]}\n\nExport of `{args.model_run}` using the postprocessing frozen on 2021 from `{args.selection_run}`: `{selected}`. Training/data/parameters inherited from the referenced manifests. Exact command: `.venv-ml/bin/python scripts/export_submission.py --model-run {args.model_run} --selection-run {args.selection_run} --validation-model {args.validation_model} --run-id {args.run_id}`. No refitting. Shape: {list(submission.shape)}; probability range {manifest["range"]}; readback precision and exact schema/IDs verified. SHA256 `{manifest["submission_sha256"]}`. Full provenance: `artifacts/{args.run_id}/manifest.json`. Runtime {manifest["runtime_seconds"]:.2f}s. Not yet submitted at export time.\n')

if __name__=='__main__':
    main()
