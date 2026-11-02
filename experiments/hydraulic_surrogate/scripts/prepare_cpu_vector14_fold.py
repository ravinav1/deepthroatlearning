"""Freeze prospective CPU664 vector14 inputs/selection; no feature build or fit."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import py_compile
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.cpu_vector14_portability import sha
from scripts.train_lgb_shop664 import check_closure


def main() -> None:
    """Pin the exact existing stage/sample/anchor/control and both environments."""
    oldpath = ROOT / 'docs/lgb_shop664_protocol.json'
    assert sha(oldpath) == '882e8c3124ba834322c8f062a7544e9ca3e9512b385e8830db387c38f7c97d08'
    old = json.loads(oldpath.read_text()); loader, metadata, folds = check_closure(old)
    sources = ['scripts/cpu_vector14_fold_common.py', 'scripts/cpu_vector14_portability.py',
               'scripts/build_cpu_vector14_inputs.py', 'scripts/train_cpu_vector14_fold.py',
               'scripts/select_cpu_vector14_fold.py', 'scripts/prepare_cpu_vector14_fold.py',
               'scripts/pilot_cpu_vector14_timing.py', 'scripts/smoke_cpu_vector14.py',
               'tests/test_cpu_vector14_fold.py', 'tests/test_cpu_vector14_portability.py']
    for name in sources: py_compile.compile(str(ROOT / name), doraise=True)
    source = {**old['source'], **{name: sha(ROOT / name) for name in sources}}
    inputs = {**old['inputs'], str(oldpath.relative_to(ROOT)): sha(oldpath)}
    prerequisite_paths = ['artifacts/cpu-vector14-smoke-002/manifest.json',
        'artifacts/cpu-vector14-smoke-result-independent-audit-001/manifest.json',
        'artifacts/cpu-vector14-timing-001/manifest.json', 'artifacts/cpu-vector14-timing-001/fit_manifest.json',
        'artifacts/cpu-vector14-portability-preflight-002/manifest.json',
        'artifacts/xgb341-cpu-environment-001/manifest.json']
    for name in prerequisite_paths: inputs[name] = sha(ROOT / name)
    assert inputs[prerequisite_paths[2]] == '1e8e99d2edab40236b5a6eaa976fa0abbc3315367e90002fcb3601577a6c99da'
    timing = json.loads((ROOT / prerequisite_paths[2]).read_text()); assert timing['status'] == 'failed'
    fit = json.loads((ROOT / prerequisite_paths[3]).read_text()); assert fit['status'] == 'complete'
    assert fit['prefix_native_trees_and_base_score_exact'] and fit['prefix_predictions_bitwise']
    parameters = dict(loader['model_parameters']); parameters.pop('n_estimators')
    parameters.update(device='cpu', multi_strategy='multi_output_tree')
    controls = {}
    for year in ['2021', '2022']:
        folder = ROOT / f'artifacts/verify-cuda-xgb664-{year}-return-001/download/cuda-xgb664-{year}-001'
        mp, pred = folder / 'manifest.json', folder / f'validation_{year}.npz'
        m = json.loads(mp.read_text()); assert m['status'] == 'complete' and m['artifacts'][pred.name] == sha(pred)
        controls[year] = {'manifest': str(mp.relative_to(ROOT)), 'manifest_sha256': sha(mp),
                          'prediction': str(pred.relative_to(ROOT)), 'prediction_sha256': sha(pred)}
        inputs[str(mp.relative_to(ROOT))] = sha(mp); inputs[str(pred.relative_to(ROOT))] = sha(pred)
        assert len(folds[year]['sample']) == {'2021': 146899, '2022': 171427}[year]
    p = {'version': 'cpu_vector14_664_v1', 'revision': 2, 'created_utc': datetime.now(timezone.utc).isoformat(),
         'source': source, 'inputs': inputs, 'data': old['data'], 'original_local_protocol': str(oldpath.relative_to(ROOT)),
         'feature_columns': old['feature_columns'], 'units': old['units'], 'samples': old['samples'], 'folds': old['folds'],
         'sample_rows': {'2021': 146899, '2022': 171427}, 'model_parameters': parameters,
         'candidate_rounds': [600, 1800], 'candidate_weights': [0., .1, .25, .5],
         'anchor': old['anchor'], 'control': controls,
         'formula': 'project_vinje((1-w)*saved_public_bestSHOP664blend+w*rawCPUvector14,float64)',
         'selection': 'One1800 trajectory on2021; evaluate fixed600prefix and1800. HighestAUC; ties lowerround thenlowerweight. Positivewinner strictly exceedsSHOPanchor and sameweightoriginal40%rawGPU664 reweighting control. Independently lockboth before2022; trainonlyselectedround and evaluateonly0+lockedweight. No post-result rescue or partial600 fallback.',
         'control_interpretation': 'Prospective reweighting-complementarity gate against an existing anchor component. CPU and vector sharing change together; this does not isolate architectural causality against GPUscalar.',
         'prefix_contract': 'Actual600 callbackcheckpoint, final600 native slice, and finalmodel iteration_range(0,600) must match treepayload/base/predictions. Synthetic16-of64 passed; independent600 fullfit equivalence is not claimed.',
         'resources': {'threads': 2, 'minimum_free_percent_before_build': 40,
             'minimum_free_percent_before_fit': 40, 'maximum_build_peak_rss_bytes': 10 * 1024**3,
             'maximum_fit_peak_rss_bytes': 6 * 1024**3, 'fit_timeout_seconds': 10800,
             'replay_timeout_seconds': 600, 'memory_enforcement': 'WorkerOSpeak checked after each round;6GiB is not a hardallocation limit. Parent hardwall termination/join. Failed resources preservepartial evidence and blockselection.',
             'priority': 'Singlelarge local slot. River726 takespriority; separate originalfeaturebuild process mustexit beforeisolatedfit. Actualschedule requiresrootauthorization.',
             'synthetic_linear_time_estimate_minutes': {'2021': {'600': 50.762234242422764, '1800': 152.2867027272683},
                 '2022': {'600': 59.23809916660977, '1800': 177.71429749982929}}},
         'environment_manifest': prerequisite_paths[-1], 'environment_manifest_sha256': inputs[prerequisite_paths[-1]],
         'early_stopping': False, 'final_training_authorized': False, 'preparation_only_at_creation': True,
         'reused_development_years': True,
         'limitations': 'No actual664 bridge/fullmodel/score measured. Historical2GiB syntheticpilot remainsfailed. Proposed6GiB budget is separate andsource-informed, not a measured memoryguarantee. Exact native600/1800 structure and all61320-row checkpoint replays required.'}
    target = ROOT / 'docs/cpu_vector14_fold_protocol_v2.json'
    with target.open('x') as stream: json.dump(p, stream, indent=2); stream.write('\n')
    print(sha(target), flush=True)


if __name__ == '__main__': main()
