"""Test the fixed river62 addition using the accepted LGB664 training recipe.

The original664 implementation and artifacts remain immutable. This runner
reuses its sample, reference, environment and selection helpers. A complete
independently audited physical cache is mandatory before feature construction.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_lgb_shop664 as base
from scripts.train_lgb_shop664 import np, lgb, roc_auc_score

BASE_PROTOCOL = 'docs/lgb_shop664_protocol.json'
BASE_SHA = '882e8c3124ba834322c8f062a7544e9ca3e9512b385e8830db387c38f7c97d08'
VERSION = 'lgb_river726_feature_ablation_v1'
LOCK_VERSION = 'lgb_river726_selection_lock_v1'


def import_path(path: Path, name: str):
    """Load a byte-pinned adapter without changing the original staged package."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_cache_contract(record: dict) -> tuple[dict, dict]:
    """Reject incomplete, unreviewed or changed cache metadata before label reads."""
    directory = ROOT / record['directory']
    assert base.sha(directory / 'manifest.json') == record['manifest_sha256']
    assert base.sha(directory / 'protocol.json') == record['protocol_sha256']
    assert base.sha(ROOT / record['audit']) == record['audit_sha256']
    manifest = json.loads((directory / 'manifest.json').read_text())
    protocol = json.loads((directory / 'protocol.json').read_text())
    audit = json.loads((ROOT / record['audit']).read_text())
    assert manifest['status'] == 'complete' and not manifest.get('failures')
    assert manifest['protocol_sha256'] == record['protocol_sha256']
    assert manifest['case_ids'] == protocol['case_ids'] and len(manifest['case_ids']) == 3647
    assert len(set(manifest['case_ids'])) == 3647
    assert set(manifest['completed_cases']) == {str(i) for i in manifest['case_ids']}
    assert manifest['all_cases_physical_and_replay_verified']
    assert audit['status'] == 'passed' and audit['case_count'] == 3647
    assert audit['cache_manifest_sha256'] == record['manifest_sha256']
    assert audit['cache_protocol_sha256'] == record['protocol_sha256']
    assert audit['all_cases_physical_and_replay_verified'] and audit['canonical_reuse_verified']
    assert base.sha(directory / 'features.npz') == manifest['features_sha256'] == audit['features_sha256']
    assert manifest['artifacts']['features.npz'] == manifest['features_sha256']
    for group in ['source', 'data']:
        for name, expected in protocol[group].items():
            assert base.sha(ROOT / name) == expected, name
    return manifest, protocol


def check_closure(protocol: dict) -> tuple[dict, dict, object, dict]:
    """Validate all declared inputs, original samples and full physical proof."""
    assert protocol['version'] == VERSION and protocol['candidate_weights'] == base.WEIGHTS
    assert protocol['base_protocol'] == BASE_PROTOCOL and protocol['base_protocol_sha256'] == BASE_SHA
    assert base.sha(ROOT / BASE_PROTOCOL) == BASE_SHA
    assert protocol['environment'] == base.environment()
    for group in ['source', 'inputs']:
        for name, expected in protocol[group].items():
            assert base.sha(ROOT / name) == expected, name
    original = json.loads((ROOT / BASE_PROTOCOL).read_text())
    loader_protocol, metadata, splits = base.check_closure(original)
    assert protocol['model_parameters'] == original['model_parameters']
    assert protocol['anchor'] == original['anchor']
    assert protocol['samples'] == original['samples'] and protocol['folds'] == original['folds']
    assert protocol['units'] == original['units']
    manifest, physical = check_cache_contract(protocol['cache'])
    assert physical['prefix_columns'] == original['feature_columns']
    assert protocol['feature_columns'] == original['feature_columns'] + physical['feature_specification']['columns']
    assert len(protocol['feature_columns']) == 726
    assert protocol['cache_feature_specification'] == manifest['feature_specification'] == physical['feature_specification']
    for year, record in protocol['control'].items():
        check_control(record, year)
    return original, loader_protocol, metadata, splits


def check_control(record: dict, year: str) -> None:
    """Bind each raw664 control to its completed independent checkpoint audit."""
    assert year in ['2021', '2022']
    assert record['prediction'] == f'artifacts/lgb-shop664-{year}-001/validation_{year}.npz'
    assert record['manifest'] == f'artifacts/lgb-shop664-{year}-001/manifest.json'
    assert record['audit'] == f'artifacts/lgb-shop664-{year}-independent-audit-001/manifest.json'
    for key in ['prediction', 'manifest', 'audit']:
        assert base.sha(ROOT / record[key]) == record[key + '_sha256']
    control = json.loads((ROOT / record['manifest']).read_text())
    audit = json.loads((ROOT / record['audit']).read_text())
    assert control['status'] == 'complete' and control['protocol_sha256'] == BASE_SHA
    assert control['artifacts'][f'validation_{year}.npz'] == record['prediction_sha256']
    assert audit['status'] == 'passed' and audit['protocol_sha256'] == BASE_SHA
    assert audit['candidate_manifest_sha256'] == record['manifest_sha256']
    assert audit['all14_native_checkpoints_replayed'] and audit['original_target_values_verified']


def validate_lock(path: Path, expected: str, protocol: dict, protocol_sha: str,
                  audit_path: Path, audit_sha: str) -> dict:
    """Require the independently verified 2021 choice before decoding 2022 labels."""
    assert base.sha(path) == expected and base.sha(audit_path) == audit_sha
    lock, audit = json.loads(path.read_text()), json.loads(audit_path.read_text())
    assert audit['status'] == 'passed' and audit['selection_lock_sha256'] == expected
    assert audit['protocol_sha256'] == protocol_sha and audit['eligible_for_2022']
    assert lock['version'] == LOCK_VERSION and lock['selection_year'] == 2021
    assert lock['protocol_sha256'] == protocol_sha
    assert lock['trainer_sha256'] == protocol['source']['scripts/train_lgb_river726.py']
    assert lock['candidate_weights'] == base.WEIGHTS
    chosen, eligible = base.choose(lock['scores_2021'], lock['control_2021_micro_auc'])
    assert eligible and lock['eligible_for_2022'] and chosen == lock['selected_weight']
    assert lock['selected_2021_micro_auc'] == lock['scores_2021'][str(chosen)]
    assert lock['anchor_2021_micro_auc'] == lock['scores_2021']['0.0']
    for kind in ['anchor', 'control']:
        assert lock[kind + '_prediction_sha256'] == protocol[kind]['2021']['prediction_sha256']
    return lock


def build_features(protocol: dict, original: dict, loader_protocol: dict, year: int) -> tuple:
    """Build untouched664 first, then append the strictly audited river62 cache."""
    loader = import_path(ROOT / 'scripts/cuda_xgb664_loader.py', 'river726_prefix_loader')
    x, labels, train, mapping, meta, baseline = loader.build_verified_features(
        ROOT / original['stage'], year, loader_protocol)
    assert x.columns.tolist() == original['feature_columns']
    # The original loader requires a fresh process and installs its pinned
    # package. Import the additive module only after this exact prefix exists.
    extension = import_path(ROOT / 'sintef_project/hyljelihyl_river_features.py', 'river726_additive_features')
    record = protocol['cache']
    x, river_spec = extension.add_river_cost_features(
        x, ROOT, ROOT / record['directory'], record['manifest_sha256'], record['protocol_sha256'],
        ROOT / record['audit'], record['audit_sha256'])
    assert x.shape == (3647 * 168, 726) and x.columns.tolist() == protocol['feature_columns']
    meta['feature_spec']['river_cost'] = river_spec
    return x, labels, train, mapping, meta, baseline


def main() -> None:
    """Run one fixed chronological fold, or metadata-only prerequisites."""
    parser = argparse.ArgumentParser()
    for name in ['protocol', 'independent-audit', 'selection-lock', 'lock-audit']:
        parser.add_argument('--' + name, type=Path, required=name == 'protocol')
        parser.add_argument('--' + name + '-sha256', required=name == 'protocol')
    parser.add_argument('--year', type=int, choices=[2021, 2022], required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    assert Path(args.run_id).name == args.run_id and args.run_id not in ['', '.', '..']
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {'run_id': args.run_id, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'initializing', 'command': shlex.join([sys.executable, *sys.argv]),
              'protocol_sha256': args.protocol_sha256, 'year': args.year, 'environment': base.environment(),
              'training': not args.preflight, 'models': {}, 'completed_units': 0, 'kaggle_submission': None,
              'caveat': 'Both chronological years are reused development data.'}
    try:
        assert base.sha(args.protocol) == args.protocol_sha256
        protocol = json.loads(args.protocol.read_text())
        original, loader_protocol, metadata, splits = check_closure(protocol)
        shutil.copy2(args.protocol, out / 'protocol.json')
        for name in protocol['source']:
            destination = out / 'source' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
        if args.preflight:
            report.update(status='passed_preflight_only', labels_decoded=False)
            return
        assert args.independent_audit and base.sha(args.independent_audit) == args.independent_audit_sha256
        review = json.loads(args.independent_audit.read_text())
        assert review['status'] == 'passed' and review['protocol_sha256'] == args.protocol_sha256
        report['independent_audit_sha256'] = args.independent_audit_sha256
        lock = None
        if args.year == 2022:
            assert args.selection_lock and args.selection_lock_sha256 and args.lock_audit and args.lock_audit_sha256
            lock = validate_lock(args.selection_lock, args.selection_lock_sha256, protocol, args.protocol_sha256,
                                 args.lock_audit, args.lock_audit_sha256)
            report.update(selection_lock_sha256=args.selection_lock_sha256, lock_audit_sha256=args.lock_audit_sha256)
        else:
            assert args.selection_lock is None and args.lock_audit is None
        report['memory_free_percent_before_build'] = base.memory_free_percent()
        if report['memory_free_percent_before_build'] < 40:
            report.update(status='operationally_deferred_resources', training=False)
            return
        x, labels, train, _, meta, _ = build_features(protocol, original, loader_protocol, args.year)
        assert train['Run No'].tolist() == metadata['Run No'].tolist()
        units, split = protocol['units'], splits[str(args.year)]
        sample, validation = split['sample'], split['validation_rows']
        y = np.column_stack([labels[u] for u in units]).astype(np.uint8)
        assert np.isin(y[sample], [0, 1]).all() and np.isin(y[validation], [0, 1]).all()
        xt, xv = x.iloc[sample].copy(), x.iloc[validation].copy()
        del x
        report.update(features=protocol['feature_columns'], feature_spec=meta['feature_spec'], units=units,
                      train_ids=protocol['folds'][str(args.year)]['train_ids'],
                      validation_ids=protocol['folds'][str(args.year)]['validation_ids'],
                      sample_sha256=protocol['samples'][str(args.year)]['sha256'], sampled_training_rows=len(sample),
                      model_parameters=protocol['model_parameters'], maximum_decoded_target_year=args.year)
        shutil.copy2(ROOT / protocol['samples'][str(args.year)]['path'], out / f'train_sample_{args.year}.npy')
        predictions = np.zeros((len(validation), 14), dtype=np.float32)
        for j, unit in enumerate(units):
            if base.memory_free_percent() < 20:
                report.update(status='operationally_deferred_resources')
                return
            tick = time.monotonic()
            model = lgb.LGBMClassifier(**protocol['model_parameters'])
            model.fit(xt, y[sample, j])
            assert model.booster_.num_trees() == 350 and model.booster_.num_feature() == 726
            predicted = model.predict_proba(xv)[:, 1]
            checkpoint = out / f'fold{args.year}_{unit}.txt'
            model.booster_.save_model(str(checkpoint))
            reloaded = lgb.Booster(model_file=str(checkpoint))
            replay = reloaded.predict(xv, num_threads=2)
            assert predicted.dtype == replay.dtype and predicted.tobytes() == replay.tobytes()
            predictions[:, j] = predicted
            report['models'][unit] = {'checkpoint_sha256': base.sha(checkpoint), 'trees': 350, 'features': 726,
                'replay_rows': len(validation), 'float64_checkpoint_replay_bitwise': True,
                'runtime_seconds': time.monotonic() - tick}
            report['completed_units'] = j + 1
            (out / 'progress.json').write_text(json.dumps(report, indent=2))
            print(json.dumps({'unit': unit, 'completed_units': j + 1, 'seconds': time.monotonic() - tick}), flush=True)
            del model, reloaded
        np.save(out / 'raw_predictions_before_selection.npy', predictions)
        seal = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'completed_units': 14,
                'prediction_sha256': base.sha(out / 'raw_predictions_before_selection.npy'),
                'models': {unit: record['checkpoint_sha256'] for unit, record in report['models'].items()}}
        (out / 'fit_and_prediction_seal.json').write_text(json.dumps(seal, indent=2))
        fold_labels = y[validation]
        ids = train.iloc[split['validation_cases']]['Run No'].to_numpy()
        np.savez_compressed(out / f'validation_{args.year}.npz', predictions=predictions,
                            labels=fold_labels, case_ids=ids, units=np.asarray(units))
        post = import_path(ROOT / 'sintef_project/postprocess.py', 'river726_projection')
        anchor = base.load_reference(protocol['anchor'][str(args.year)], fold_labels, ids, units)
        control = base.load_reference(protocol['control'][str(args.year)], fold_labels, ids, units)
        weights = base.WEIGHTS if lock is None else [0., lock['selected_weight']]
        values = {str(w): post.project_vinje((1-w)*anchor + w*predictions.astype(np.float64), units) for w in weights}
        assert np.array_equal(values['0.0'], anchor)
        scores = {w: float(roc_auc_score(fold_labels.ravel(), p.ravel())) for w, p in values.items()}
        selected = max(weights, key=lambda w: (scores[str(w)], -w)) if lock is None else lock['selected_weight']
        matched = post.project_vinje((1-selected)*anchor + selected*control, units)
        control_auc = float(roc_auc_score(fold_labels.ravel(), matched.ravel()))
        selected, eligible = base.choose(scores, control_auc, selected)
        np.savez_compressed(out / f'selected_validation_{args.year}.npz', predictions=values[str(selected)],
                            labels=fold_labels, case_ids=ids, units=np.asarray(units))
        report.update(status='complete', completed_rounds_per_unit=350, checkpoint_replay_bitwise=True,
                      scores=scores, selected_weight=selected, selected_micro_auc=scores[str(selected)],
                      anchor_micro_auc=scores['0.0'], control_micro_auc=control_auc, beats_anchor_and_control=eligible,
                      raw_micro_auc=float(roc_auc_score(fold_labels.ravel(), predictions.ravel())),
                      fit_and_prediction_seal_sha256=base.sha(out / 'fit_and_prediction_seal.json'))
        if lock is None:
            selection = {'version': LOCK_VERSION, 'selection_year': 2021, 'locked_utc': datetime.now(timezone.utc).isoformat(),
                'protocol_sha256': args.protocol_sha256, 'trainer_sha256': base.sha(Path(__file__)),
                'candidate_weights': base.WEIGHTS, 'scores_2021': scores, 'selected_weight': selected,
                'selected_2021_micro_auc': scores[str(selected)], 'anchor_2021_micro_auc': scores['0.0'],
                'control_2021_micro_auc': control_auc, 'eligible_for_2022': eligible,
                'anchor_prediction_sha256': protocol['anchor']['2021']['prediction_sha256'],
                'control_prediction_sha256': protocol['control']['2021']['prediction_sha256'],
                'raw_prediction_sha256': base.sha(out / 'validation_2021.npz'),
                'fit_and_prediction_seal_sha256': report['fit_and_prediction_seal_sha256']}
            (out / 'selection_2021.json').write_text(json.dumps(selection, indent=2))
            report['selection_lock_sha256'] = base.sha(out / 'selection_2021.json')
        else:
            report.update(accepted=eligible, meets_transfer_gate=min(lock['selected_2021_micro_auc'], scores[str(selected)]) >= .998)
        print(json.dumps({k: report[k] for k in ['status', 'selected_weight', 'selected_micro_auc', 'control_micro_auc', 'beats_anchor_and_control']}), flush=True)
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report.update(runtime_seconds=time.monotonic() - started, peak_rss_bytes=base.peak_rss())
        report['artifacts'] = {p.relative_to(out).as_posix(): base.sha(p) for p in out.rglob('*')
                               if p.is_file() and p.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
        with (ROOT / 'docs/experiments.md').open('a') as stream:
            stream.write(f'\n\n## {args.run_id} — {report["timestamp_utc"]}\n\n'
                f'Status {report["status"]}; frozen664+62 river feature test. Command `{report["command"]}`. '
                f'Year {args.year}, selected weight {report.get("selected_weight")}, AUC {report.get("selected_micro_auc")}, '
                f'anchor {report.get("anchor_micro_auc")}, matched LGB664 {report.get("control_micro_auc")}. '
                f'350trees/31leaves/lr.06/minleaf100/L2=1/featurefraction.9/seed4173/2threads/exact40% sample; no early stopping. '
                f'Runtime {report["runtime_seconds"]:.3f}s; peakRSS {report["peak_rss_bytes"]}bytes. '
                f'Exact source/data/cache/split/config/replay evidence: `artifacts/{args.run_id}/manifest.json`. No public score.\n')


if __name__ == '__main__':
    main()
