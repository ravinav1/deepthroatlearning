"""Freeze fourteen paired linear/constant leaf residuals before reused2020 screening."""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shlex
import shutil
import sys
import time
import traceback

for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[variable] = '1'

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.screen_pooled_pairwise_residual import digest, evaluate, read_screen_labels
from sintef_project.linear_leaf_residual import (
    fit_normalizer, normalize, residual_sample, select_gain_features, fit_unit, leaf_geometry,
)
from sintef_project.pooled_pairwise_residual import apply_residual
from sintef_project.postprocess import project_vinje


class ResourceBudgetExceeded(RuntimeError):
    """An operational resource limit, never a negative model result."""


def peak_rss_bytes() -> int:
    """Read native process high-water resident memory with platform-correct units."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == 'darwin' else rss*1024)


class ResourceBudget:
    """Check elapsed wall time and process peakRSS at each bounded stage/boost round."""

    def __init__(self, seconds: float, rss_bytes: int, clock=time.monotonic, memory=peak_rss_bytes):
        self.seconds, self.rss_bytes = seconds, rss_bytes
        self.clock, self.memory = clock, memory
        self.start = clock()

    def check(self, stage: str) -> None:
        elapsed, rss = self.clock()-self.start, self.memory()
        if elapsed > self.seconds or rss > self.rss_bytes:
            raise ResourceBudgetExceeded(f'{stage}: elapsed={elapsed:.6f}s, peakRSS={rss}bytes')


def main() -> None:
    """Run only the frozen pre2021 mechanism experiment, with no adaptive retries."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--preflight-audit', type=Path, required=True)
    parser.add_argument('--preflight-audit-sha256', required=True)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in ['', '.', '..']:
        raise ValueError('Invalid immutable run ID')
    out = ROOT/'artifacts'/args.run_id
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {'run_id': args.run_id, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': shlex.join([sys.executable, *sys.argv]), 'status': 'initializing',
              'scope': 'Reused2020 development mechanism screen on single608LGB OOF anchor, not acceptedGPUensemble',
              'screen_labels_opened_utc': None, 'early_stopping': None, 'kaggle_submission': None,
              'models': {}, 'completed_models': 0,
              'environment': {'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
                              'packages': {name: importlib.metadata.version(name) for name in
                                           ['numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm']}}}
    try:
        protocol_path = ROOT/'docs/linear_leaf_residual_protocol.json'
        if digest(protocol_path) != args.protocol_sha256:
            raise ValueError('Frozen linear-leaf protocol differs')
        p = json.loads(protocol_path.read_text())
        budget = ResourceBudget(p['resource_budget']['wall_seconds'], p['resource_budget']['peak_rss_bytes'])
        if digest(args.preflight_audit) != args.preflight_audit_sha256:
            raise ValueError('Independent preflight audit hash differs')
        audit = json.loads(args.preflight_audit.read_text())
        if audit['status'] != 'passed' or audit['protocol_sha256'] != args.protocol_sha256:
            raise ValueError('Require independent approval of this exact protocol')
        for group in ['source', 'data', 'inputs']:
            for name, expected in p[group].items():
                if digest(ROOT/name) != expected:
                    raise ValueError(f'Changed {group}: {name}')
                budget.check('verify '+name)
        for name in p['source']:
            target = out/'source'/name
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/name, target)
        shutil.copy2(protocol_path, out/'protocol.json')
        report.update(protocol_sha256=args.protocol_sha256, preflight_audit_sha256=args.preflight_audit_sha256,
                      source=p['source'], data=p['data'], inputs=p['inputs'], model_parameters=p['model_parameters'],
                      feature_selection=p['feature_selection'], resource_budget=p['resource_budget'])
        directory = ROOT/p['oof_directory']
        manifest = json.loads((directory/'manifest.json').read_text())
        oof_audit = json.loads((ROOT/p['oof_audit_path']).read_text())
        if manifest['status'] != 'complete' or oof_audit['status'] != 'passed' or oof_audit['oof_manifest_sha256'] != digest(directory/'manifest.json'):
            raise ValueError('Require complete independently verified full14 OOF')
        for name, expected in manifest['artifacts'].items():
            if digest(directory/name) != expected:
                raise ValueError('OOF artifact changed: '+name)
            budget.check('verify OOF '+name)
        arrays = {}
        for year in [2018, 2019, 2020]:
            with np.load(directory/f'oof_{year}.npz', allow_pickle=False) as z:
                arrays[year] = {key: z[key] for key in z.files}
            np.testing.assert_array_equal(arrays[year]['case_ids'], p['case_ids'][str(year)])
            if arrays[year]['units'].tolist() != p['units'] or ('labels' in arrays[year]) != (year < 2020):
                raise ValueError('Changed unit or label-sealing schema')
        raw_features = {year: np.load(directory/f'features_{year}.npy', mmap_mode='r', allow_pickle=False)
                        for year in [2018, 2019, 2020]}
        train_probability = np.concatenate([arrays[year]['probability'] for year in [2018, 2019]]).reshape(-1, 14)
        train_labels = np.concatenate([arrays[year]['labels'] for year in [2018, 2019]]).reshape(-1, 14)
        sample = residual_sample(len(train_probability))
        np.save(out/'residual_training_rows.npy', sample)
        if len(sample) != 48652 or digest(out/'residual_training_rows.npy') != p['residual_sample_sha256']:
            raise ValueError('Shared fixed40% residual hour sample differs')
        report['training_case_ids'] = np.concatenate([arrays[y]['case_ids'] for y in [2018, 2019]]).tolist()
        report['sample_rows'] = len(sample)
        normalizers = {}
        models = {kind: {} for kind in p['fit_order']}
        for j, unit in enumerate(p['units']):
            choice = p['feature_selection'][unit]
            base = lgb.Booster(model_file=str(directory/f'fold2018_{unit}.txt'))
            indices = select_gain_features(base.feature_importance(importance_type='gain'), 32)
            np.testing.assert_array_equal(indices, choice['indices'])
            if [base.feature_name()[int(i)] for i in indices] != choice['names']:
                raise ValueError('Pre2018 gain-selected feature names changed')
            del base
            full = np.concatenate([raw_features[year][:, indices] for year in [2018, 2019]])
            selected = full[sample]
            del full
            mean, scale = fit_normalizer(selected)
            values = normalize(selected, mean, scale)
            del selected
            normalizers[unit] = (indices, mean, scale)
            np.savez_compressed(out/f'normalizer_{unit}.npz', indices=indices, mean=mean, scale=scale)
            report['models'][unit] = {}
            for kind in p['fit_order']:
                tick = time.monotonic()
                def progress(environment):
                    budget.check(f'train {kind} {unit} round{environment.iteration+1}')
                model, trace = fit_unit(values, train_labels[sample, j], train_probability[sample, j],
                                        p['model_parameters'], 40, kind == 'linear', progress)
                checkpoint = out/f'{kind}_{unit}.txt'
                model.save_model(str(checkpoint))
                (out/f'{kind}_{unit}_trace.json').write_text(json.dumps(trace, indent=2))
                geometry = leaf_geometry(model)
                if geometry['nonzero_linear_coefficients_by_tree'][0] != 0:
                    raise ValueError('Unexpected first-tree geometry')
                if kind == 'constant' and geometry['total_nonzero_linear_coefficients']:
                    raise ValueError('Constant-leaf control contains linear slopes')
                models[kind][unit] = model
                report['models'][unit][kind] = {'checkpoint_sha256': digest(checkpoint), 'rounds': model.current_iteration(),
                                                'runtime_seconds': time.monotonic()-tick, 'geometry': geometry,
                                                'offset_first_gradient_exact': True}
                report['completed_models'] += 1
                (out/'manifest.json').write_text(json.dumps(report, indent=2))
                print('LINEAR_SCREEN_MODEL', unit, kind, report['models'][unit][kind]['runtime_seconds'], flush=True)
                budget.check('checkpoint '+kind+' '+unit)
            del values
            gc.collect()
        if not any(record['linear']['geometry']['total_nonzero_linear_coefficients'] > 0 for record in report['models'].values()):
            raise ValueError('No fitted linear coefficient: unsupported geometric experiment')
        fit_lock = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': args.protocol_sha256,
                    'maximum_target_year': 2019, 'training_case_ids': report['training_case_ids'],
                    'training_rows_sha256': digest(out/'residual_training_rows.npy'),
                    'artifacts': {path.name: digest(path) for path in out.iterdir()
                                  if path.is_file() and (path.suffix == '.txt' or path.name.endswith('_trace.json') or path.name.startswith('normalizer_'))}}
        (out/'fit_lock.json').write_text(json.dumps(fit_lock, indent=2))
        report['fit_lock_sha256'] = digest(out/'fit_lock.json')
        screen = arrays[2020]
        base_probability = screen['probability']
        raw = {'identity': apply_residual(base_probability, np.zeros_like(base_probability))}
        if raw['identity'].dtype != base_probability.dtype or raw['identity'].tobytes() != base_probability.tobytes():
            raise ValueError('Zero residual must preserve raw base bytes')
        residuals = {kind: np.empty(base_probability.shape, dtype=np.float64) for kind in p['fit_order']}
        for j, unit in enumerate(p['units']):
            indices, mean, scale = normalizers[unit]
            values = normalize(raw_features[2020][:, indices], mean, scale)
            for kind in p['fit_order']:
                model = models[kind][unit]
                residual = model.predict(values, raw_score=True, num_threads=2)
                replay = lgb.Booster(model_file=str(out/f'{kind}_{unit}.txt')).predict(values, raw_score=True, num_threads=2)
                np.testing.assert_array_equal(residual, replay)
                if not np.isfinite(residual).all():
                    raise ValueError('Nonfinite screen residual')
                residuals[kind][:, :, j] = residual.reshape(len(base_probability), 168)
                budget.check('prediction replay '+kind+' '+unit)
            del values
        for kind, residual in residuals.items():
            raw[kind] = apply_residual(base_probability, residual)
        np.savez_compressed(out/'screen_input_predictions.npz', **raw, case_ids=screen['case_ids'], units=np.asarray(p['units']))
        np.savez_compressed(out/'screen_residuals.npz', **residuals, case_ids=screen['case_ids'])
        prediction_lock = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'fit_lock_sha256': report['fit_lock_sha256'],
                           'screen_predictions_sha256': digest(out/'screen_input_predictions.npz'),
                           'screen_residuals_sha256': digest(out/'screen_residuals.npz'),
                           'all_28_checkpoint_replays_exact': True, 'identity_bytes_exact': True}
        (out/'prediction_lock.json').write_text(json.dumps(prediction_lock, indent=2))
        report['prediction_lock_sha256'] = digest(out/'prediction_lock.json')
        budget.check('before any screen target decoding')
        report['screen_labels_opened_utc'] = datetime.now(timezone.utc).isoformat()
        labels = read_screen_labels(ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv', screen['case_ids'], p['units'])
        results = {kind: evaluate(values, labels, screen['dates'], p['units']) for kind, values in raw.items()}
        checks = {'whole_screen_beats_both_controls': results['linear']['projected_auc'] > max(results[k]['projected_auc'] for k in ['identity', 'constant'])}
        for half in ['first_half', 'second_half']:
            checks[half+'_beats_both_controls'] = results['linear']['halves'][half]['projected_auc'] > max(results[k]['halves'][half]['projected_auc'] for k in ['identity', 'constant'])
        gain = (results['linear']['projected_auc']-results['identity']['projected_auc'])/(1-results['identity']['projected_auc'])
        for kind, values in raw.items():
            projected = project_vinje(values.reshape(-1, 14), p['units']).reshape(values.shape)
            np.savez_compressed(out/f'screen_{kind}.npz', predictions=projected, raw_predictions=values,
                                labels=labels, case_ids=screen['case_ids'], units=np.asarray(p['units']))
        budget.check('complete screen')
        report.update(status='complete', results=results, gate_checks=checks, gate_passed=bool(all(checks.values())),
                      relative_error_reduction=gain, substantial_gain=bool(gain >= .05),
                      all_28_checkpoint_replays_exact=True, identity_bytes_exact=True,
                      residual_summary={kind: {'mean': float(delta.mean()), 'std': float(delta.std()),
                                               'min': float(delta.min()), 'max': float(delta.max())} for kind, delta in residuals.items()})
        print(json.dumps({key: report[key] for key in ['status', 'results', 'gate_checks', 'relative_error_reduction']}, indent=2), flush=True)
    except (ResourceBudgetExceeded, MemoryError):
        report.update(status='operationally_inconclusive', error=traceback.format_exc(), gate_passed=None)
        raise
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report['runtime_seconds'] = time.monotonic()-started
        report['peak_rss_bytes'] = peak_rss_bytes()
        report['artifacts'] = {path.relative_to(out).as_posix(): digest(path) for path in out.rglob('*')
                               if path.is_file() and path.name != 'manifest.json'}
        (out/'manifest.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__': main()
