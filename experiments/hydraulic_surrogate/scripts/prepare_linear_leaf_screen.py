"""Freeze the Pro-inspired linear-leaf mechanism screen without fitting competition models."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.screen_pooled_pairwise_residual import digest
from sintef_project.linear_leaf_residual import select_gain_features, residual_sample

SOURCE = ['scripts/prepare_linear_leaf_screen.py', 'scripts/screen_linear_leaf_residual.py',
          'sintef_project/linear_leaf_residual.py', 'tests/test_linear_leaf_residual.py',
          'sintef_project/pooled_pairwise_residual.py', 'scripts/screen_pooled_pairwise_residual.py',
          'sintef_project/postprocess.py', 'tests/test_pooled_pairwise_screen.py']


def main():
    out = ROOT/'artifacts/linear-leaf-residual-preflight-003'
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    oof_dir = ROOT/'artifacts/full14-early-oof-001'
    assert digest(oof_dir/'manifest.json') == '1016522707ec517f9e86dbabfe75686cf9b8d963aade721483d2f49bf0586146'
    oof = json.loads((oof_dir/'manifest.json').read_text())
    assert oof['status'] == 'complete' and oof['folds']['2018']['train_dates'][1] == '2017-12-25 00:00:00+00:00'
    inputs = {'artifacts/full14-early-oof-001/manifest.json': digest(oof_dir/'manifest.json'),
              'artifacts/full14-early-oof-result-independent-audit-001/manifest.json': '6e8bd061d8e00361f3593f32b5ca666b168242becbc479ca044f3a66bb645ed4',
              'artifacts/pro-swarm-consultation-001/manifest.json': '94b4557edae207df2f189def950c624fe7e8a35b0351a1109e4e43408afe2478',
              'artifacts/pooled-pairwise-residual-screen-001/manifest.json': '9bca57e1d51c9501a023825235b3075ecd922734118156a9ed962ec7eb95cd43',
              'artifacts/pooled-pairwise-residual-result-independent-audit-001/manifest.json': '6365be7f04a36062f84fc2e462f18bac5039cb0681cc8fcf595c5db6d53d55fa'}
    for name, expected in inputs.items(): assert digest(ROOT/name) == expected, name
    selection = {}
    for unit in oof['units']:
        checkpoint = oof_dir/f'fold2018_{unit}.txt'
        assert digest(checkpoint) == oof['artifacts'][checkpoint.name]
        base = lgb.Booster(model_file=str(checkpoint))
        assert base.num_feature() == 608 and base.current_iteration() == 350
        assert base.feature_name() == [name.replace(' ', '_') for name in oof['feature_names']]
        gains = base.feature_importance(importance_type='gain')
        indices = select_gain_features(gains, 32)
        selection[unit] = {'checkpoint': str(checkpoint.relative_to(ROOT)), 'checkpoint_sha256': digest(checkpoint),
                           'indices': indices.tolist(), 'names': [base.feature_name()[int(i)] for i in indices],
                           'raw_feature_names': [oof['feature_names'][int(i)] for i in indices],
                           'gains': gains[indices].tolist()}
        inputs[str(checkpoint.relative_to(ROOT))] = digest(checkpoint)
    sample = residual_sample(724*168)
    sample_path = out/'residual_training_rows.npy'
    np.save(sample_path, sample)
    assert len(sample) == 48652
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'tests.test_linear_leaf_residual',
                             'tests.test_pooled_pairwise_screen'], cwd=ROOT, text=True, capture_output=True)
    (out/'tests.log').write_text(tests.stdout+tests.stderr)
    assert tests.returncode == 0 and 'Ran 6 tests' in tests.stderr, tests.stderr
    p = {'version': 'linear_leaf_residual_early_screen_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
         'hypothesis': 'Small linear leaf models represent smooth hydraulic boundaries better than matched constant leaves.',
         'scope': 'Single608LGB OOF mechanism screen; not accepted GPU ensemble. Reused2020 development labels already observed by rejected earlier hypotheses.',
         'source': {name: digest(ROOT/name) for name in SOURCE}, 'data': oof['data'], 'inputs': inputs,
         'oof_directory': str(oof_dir.relative_to(ROOT)),
         'oof_audit_path': 'artifacts/full14-early-oof-result-independent-audit-001/manifest.json',
         'units': oof['units'], 'case_ids': {year: fold['prediction_ids'] for year, fold in oof['folds'].items()},
         'feature_selection': selection,
         'selection_rule': 'Per unit top32 gain features of its frozen2018basecheckpoint trained through2017-12-25; descendinggain, ties by originalcolumnindex; no residual or2020featureselection.',
         'residual_sample_sha256': digest(sample_path),
         'residual_sample': {'seed': 4173, 'fraction': .4, 'available_case_hour_rows': 121632, 'sampled_rows': 48652,
                             'rule': 'One sorted default_rng4173 choice over concatenated2018then2019 case-hours, shared across all14units and botharms.'},
         'normalization': {'training_scope': 'Only shared sampled48652trainingrows, separately for eachunit32selectednumericfeatures.',
                           'mean': 'NaN-aware arithmetic mean', 'std': 'NaN-aware population standard deviation ddof0',
                           'degenerate_scale': 'Use1 ifstd<=1e-12; rejectallmissingcolumn',
                           'missing': 'RetainNaN; realzero remains numeric, zero_as_missing=false; rejectinfinity; no screenfitting or clipping.'},
         'model_parameters': {'boosting_type': 'gbdt', 'tree_learner': 'serial', 'device_type': 'cpu',
                              'num_threads': 2, 'max_depth': 3, 'num_leaves': 8, 'learning_rate': .05,
                              'min_data_in_leaf': 512, 'min_sum_hessian_in_leaf': .001,
                              'lambda_l2': 10., 'lambda_l1': 0., 'linear_lambda': 10.,
                              'max_bin': 255, 'feature_fraction': 1., 'bagging_fraction': 1., 'bagging_freq': 0,
                              'seed': 4173, 'data_random_seed': 4173, 'feature_fraction_seed': 4173, 'bagging_seed': 4173,
                              'deterministic': True, 'force_col_wise': True, 'boost_from_average': False,
                              'use_missing': True, 'zero_as_missing': False, 'verbosity': -1},
         'rounds': 40, 'fit_order': ['constant', 'linear'],
         'objective': 'BinarysumBCE customgradient and exactdiagonalhessian+fixed1e-12floor, with frozen raw OOFlogit init_score. First gradient must equal initialoffset; saved model predicts residualonly.',
         'leaf_geometry': 'Only linear_tree differs; linear_lambda10 in both parameter maps. First treeconstant, later linear leaves use numeric branchfeatures. Serialize/check actualnonzerolinearcoefficients.',
         'prediction': 'Exact interior float64 logits, endpointonly1e-12 substitute, byteexactzeroidentity; sigmoid(base_logit+savedrawresidual), one finalVinjeprojection; no offset/temperature/blend search.',
         'resource_budget': {'wall_seconds': 600., 'peak_rss_bytes': 4*1024**3,
                             'measurement': 'time.monotonic and resource.getrusage(RUSAGE_SELF).ru_maxrss, bytesonmacOS/KiBconvertedonLinux.',
                             'checks': 'Eachhash/stage/boostround/predictionreplay and beforetargetdecoding. A single nativeoperation can exceedthebudget beforethenextcallback; no silentpartialmodelresult.',
                             'overrun': 'operationally_inconclusive, preservepartials, noadaptive retry/reduction and no targetdecode aftera failedprelabelcheck.'},
         'seal': 'All28models/normalizers/traces sealedbefore2020inputinference; all rawpredictions and28savedcheckpointreplays sealedbefore2020targetdecoder.',
         'gates': {'advance': 'Linear projectedmicroAUC strictlybeatsidentityANDconstantcontrol on full360case2020screen and both fixedcalendarhalves182/178.',
                   'substantial_gain': 'Atleast5%relative1-AUCreduction vsidentity, reportedseparately notaveto.',
                   'no_automatic_continuation': 'No2021/2022/finalfit/submission inthisrunner.'},
         'primary_sources': ['https://lightgbm.readthedocs.io/en/stable/Parameters.html#linear_tree',
                             'https://lightgbm.readthedocs.io/en/stable/Parameters.html#linear_lambda',
                             'https://arxiv.org/abs/1802.05640'],
         'consultation': 'Root browser consultedvisiblePro; exactgenerationunverified, preserved in pinned consultationmanifest. Root fixed resourcebudget and positiveoverall/halfyeargates beforethisprotocol.',
         'kaggle_submission': None}
    path = ROOT/'docs/linear_leaf_residual_protocol.json'
    with path.open('x') as stream: json.dump(p, stream, indent=2)
    for name in SOURCE:
        target = out/'source'/name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/name, target)
    shutil.copy2(path, out/'protocol.json')
    report = {'status': 'passed', 'timestamp_utc': p['created_utc'], 'command': f'{sys.executable} scripts/prepare_linear_leaf_screen.py',
              'protocol_sha256': digest(path), 'test_count': 6, 'competition_training': False,
              'source': p['source'], 'inputs': inputs, 'runtime_seconds': time.monotonic()-started,
              'environment': {'python': sys.version, 'platform': platform.platform(),
                              'packages': {name: importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','lightgbm']}},
              'artifacts': {path.relative_to(out).as_posix(): digest(path) for path in out.rglob('*') if path.is_file()}}
    (out/'manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'status': 'passed', 'protocol_sha256': digest(ROOT/'docs/linear_leaf_residual_protocol.json'),
                      'manifest_sha256': digest(out/'manifest.json'), 'runtime_seconds': report['runtime_seconds']}), flush=True)


if __name__ == '__main__': main()
