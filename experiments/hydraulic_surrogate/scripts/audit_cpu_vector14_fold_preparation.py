"""Independent metadata/source review of the prospective CPU vector14 recipe."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.cpu_vector14_fold_common import read_protocol, sha


def main() -> None:
    """Approve frozen source and selection contracts without data builds or fitting."""
    start = time.monotonic()
    out = ROOT / 'artifacts/cpu-vector14-fold-source-independent-audit-001'
    out.mkdir(exist_ok=False)
    expected = 'e4deebd4d7f9ac666c8da6c97e82f60f7ba55630816f2e7cf3e56008061ba121'
    report = {'status': 'initializing', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': [sys.executable, *sys.argv], 'training': False,
              'features_built': False, 'labels_decoded': False, 'execution_scheduled': False}
    try:
        protocol_path = ROOT / 'docs/cpu_vector14_fold_protocol_v2.json'
        p = read_protocol(protocol_path, expected)
        assert p['revision'] == 2 and p['early_stopping'] is False and p['final_training_authorized'] is False
        original_path = ROOT / p['original_local_protocol']
        original = json.loads(original_path.read_text())
        assert sha(original_path) == '882e8c3124ba834322c8f062a7544e9ca3e9512b385e8830db387c38f7c97d08'
        for key in ['feature_columns', 'units', 'samples', 'folds', 'anchor']:
            assert p[key] == original[key], key
        assert p['candidate_rounds'] == [600, 1800] and p['candidate_weights'] == [0., .1, .25, .5]
        assert p['sample_rows'] == {'2021': 146899, '2022': 171427}
        parameters = p['model_parameters']
        assert parameters == {
            'max_depth': 6, 'learning_rate': .05, 'min_child_weight': 20,
            'reg_lambda': 4., 'reg_alpha': 0., 'max_bin': 128, 'subsample': .9,
            'colsample_bytree': .9, 'colsample_bylevel': 1., 'colsample_bynode': 1.,
            'random_state': 4173, 'n_jobs': 2, 'objective': 'binary:logistic',
            'tree_method': 'hist', 'device': 'cpu', 'eval_metric': 'logloss',
            'grow_policy': 'depthwise', 'max_delta_step': 0.,
            'validate_parameters': True, 'verbosity': 0, 'multi_strategy': 'multi_output_tree'}
        assert p['resources']['maximum_fit_peak_rss_bytes'] == 6 * 1024 ** 3
        assert p['resources']['fit_timeout_seconds'] == 10800
        assert p['resources']['replay_timeout_seconds'] == 600
        assert p['resources']['maximum_build_peak_rss_bytes'] == 10 * 1024 ** 3
        assert p['environment_manifest_sha256'] == '2094900a418de72fa78499a0ad92b2fab2019c24002f60edfa048d17bec0b92e'
        assert sha(ROOT / p['environment_manifest']) == p['environment_manifest_sha256']
        preparation = ROOT / 'artifacts/cpu-vector14-fold-preparation-002/manifest.json'
        assert sha(preparation) == '8b9ce9a065851233b20c3cc301768871885fd407c8a8cd761ecfa555afa6883e'
        prep = json.loads(preparation.read_text())
        assert prep['protocol_sha256'] == expected
        command = [sys.executable, '-m', 'unittest', 'tests.test_cpu_vector14_fold',
                   'tests.test_cpu_vector14_portability']
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        (out / 'focused_tests.log').write_text(completed.stdout + completed.stderr)
        assert completed.returncode == 0
        report.update(status='passed', protocol_sha256=expected, source=p['source'],
                      preparation_manifest_sha256=sha(preparation),
                      focused_tests_command=command, focused_tests_exit_code=0,
                      review_findings=[
                          'Original664 loader and exact chronological samples build in the unchanged environment; blockwise native DataFrame/float32/NPY checks preserve input bytes.',
                          'Isolated3.4.1 fit/replay processes verify native input blocks, environment and protocol. Validation target payloads are hashed but never decoded by the fit worker.',
                          'One1800 trajectory on2021 records the600 callback snapshot and compares native trees/base with final slice; all full validation predictions must replay in UBJ and JSON.',
                          'Parent terminates and joins timed-out children. PeakRSS is checked between boosting rounds and after phases, not enforced as an allocation limit. Resource failure cannot become a600-only fallback.',
                          'Selector requires independent actual bridge/checkpoint proof before scoring; it chooses the frozen round/weight grid with deterministic ties and compares both anchor and existing rawGPU664 reweighting control.',
                          'Independent2021 lock gates2022 bridge construction, training and selection;2022 uses only the selected capacity and blend.',
                          'CPU and vector sharing change together, so the rawGPU control supports complementarity rather than isolated architecture causality.'],
                      limitations='Source/preparation approval only. No full664 bridge or model has run. Actual bridge/checkpoint/selection audits remain mandatory; no full-fit scheduling decision or measured6GiB guarantee is implied.')
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        shutil.copy2(__file__, out / Path(__file__).name)
        report['runtime_seconds'] = time.monotonic() - start
        report['artifacts'] = {q.name: sha(q) for q in out.iterdir() if q.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'status': report['status'], 'sha256': sha(out / 'manifest.json'),
                          'seconds': report['runtime_seconds']}), flush=True)


if __name__ == '__main__':
    main()
