"""Frozen early-2020 from-scratch canonical pooling screen; no later-year fit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
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
from scripts.train_lgb_shop664 import np, pd, lgb, roc_auc_score

VERSION = 'canonical_unit115_pre2020_screen_v1'
BASE_PATH = 'docs/lgb_shop664_protocol.json'
BASE_SHA = '882e8c3124ba834322c8f062a7544e9ca3e9512b385e8830db387c38f7c97d08'
WEIGHTS = [0., .1, .25, .5]
ROUNDS = [700, 1400]


def import_path(path: Path, name: str):
    """Import a pinned additive source without polluting the staged package."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def early_splits(metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the third original RNG draw and six-day training purge."""
    dates = pd.to_datetime(metadata.starttime, utc=True)
    rng = np.random.default_rng(4173)
    for year in [2018, 2019, 2020]:
        begin = pd.Timestamp(f'{year}-01-01', tz='UTC')
        train = np.flatnonzero(dates < begin - pd.Timedelta(days=6))
        rows = (train[:, None] * 168 + np.arange(168)).ravel()
        sample = np.sort(rng.choice(rows, size=int(.4 * len(rows)), replace=False))
    screen = np.flatnonzero((dates >= pd.Timestamp('2020-01-01', tz='UTC')) &
                            (dates <= pd.Timestamp('2020-12-25', tz='UTC')))
    assert len(train) == 1820 and len(screen) == 360 and len(sample) == 122304
    assert dates.iloc[train].max() + pd.Timedelta(hours=167) < pd.Timestamp('2020-01-01', tz='UTC')
    return train, screen, sample


def check_closure(protocol: dict) -> tuple[dict, dict, pd.DataFrame, tuple]:
    """Check source, metadata-only chronology and immutable sample/stage closure."""
    assert protocol['version'] == VERSION and protocol['weights'] == WEIGHTS and protocol['rounds'] == ROUNDS
    assert protocol['base_protocol'] == BASE_PATH and protocol['base_protocol_sha256'] == BASE_SHA
    assert base.sha(ROOT / BASE_PATH) == BASE_SHA and protocol['environment'] == base.environment()
    for group in ['source', 'inputs']:
        for name, checksum in protocol[group].items():
            assert base.sha(ROOT / name) == checksum, name
    original = json.loads((ROOT / BASE_PATH).read_text())
    loader, metadata, _ = base.check_closure(original)
    assert protocol['control_parameters'] == original['model_parameters']
    assert protocol['pooled_parameters'] == {**original['model_parameters'], 'n_estimators': 1400}
    assert protocol['units'] == original['units'] and protocol['prefix_columns'] == original['feature_columns']
    train, screen, sample = early_splits(metadata)
    assert metadata.iloc[train]['Run No'].tolist() == protocol['train_ids']
    assert metadata.iloc[screen]['Run No'].tolist() == protocol['screen_ids']
    saved = np.load(ROOT / protocol['sample']['path'], allow_pickle=False)
    assert saved.dtype == sample.dtype == np.dtype('int64') and saved.shape == sample.shape
    assert saved.tobytes() == sample.tobytes() and base.sha(ROOT / protocol['sample']['path']) == protocol['sample']['sha256']
    oof = json.loads((ROOT / protocol['oof_manifest']).read_text())
    assert oof['status'] == 'complete' and oof['screen_2020_labels_read'] is False
    assert oof['all_42_checkpoint_replays_exact'] and oof['folds']['2020']['sample_sha256'] == protocol['sample']['sha256']
    for record in protocol['prerequisites']:
        proof = json.loads((ROOT / record['path']).read_text())
        assert proof['status'] == 'passed' and base.sha(ROOT / record['path']) == record['sha256']
    return original, loader, metadata, (train, screen, sample)


def read_screen_labels(path: Path, ids: np.ndarray, units: list[str]) -> np.ndarray:
    """Decode only the frozen 360 starts, after every model/prediction is sealed."""
    metadata = pd.read_csv(path, usecols=['Run No', 'starttime'])
    _, indices, _ = early_splits(metadata)
    assert metadata.iloc[indices]['Run No'].tolist() == ids.tolist()
    selected = np.zeros(len(metadata), dtype=bool)
    selected[indices] = True
    columns = [f'result_committed_{unit}_t{hour}' for unit in units for hour in range(168)]
    target = pd.read_csv(path, skiprows=(np.flatnonzero(~selected) + 1).tolist(),
                         usecols=['Run No', *columns], dtype={name: 'uint8' for name in columns})
    assert target['Run No'].tolist() == ids.tolist()
    labels = target[columns].to_numpy().reshape(360, 14, 168).transpose(0, 2, 1)
    assert np.isin(labels, [0, 1]).all()
    return labels


def scores(predictions: np.ndarray, labels: np.ndarray, dates: list[str]) -> dict:
    """Evaluate micro AUC on the fixed year and July-1 calendar halves."""
    assert predictions.shape == labels.shape and np.isfinite(predictions).all()
    assert 0 <= predictions.min() <= predictions.max() <= 1
    first = pd.to_datetime(dates, utc=True) < pd.Timestamp('2020-07-01', tz='UTC')
    masks = {'full': np.ones(len(first), dtype=bool), 'first_half': first, 'second_half': ~first}
    return {name: float(roc_auc_score(labels[mask].ravel(), predictions[mask].ravel())) for name, mask in masks.items()}


def select_candidate(grid: dict, controls: dict, anchor: dict) -> dict:
    """Pick full-year maximum once, then require all six strict comparison gates."""
    assert set(grid) == {f'{rounds}:{weight}' for rounds in ROUNDS for weight in WEIGHTS}
    rounds, weight = max(((r, w) for r in ROUNDS for w in WEIGHTS),
                         key=lambda pair: (grid[f'{pair[0]}:{pair[1]}']['full'], -pair[0], -pair[1]))
    selected = grid[f'{rounds}:{weight}']
    matched = controls[str(weight)]
    gates = {f'{scope}_beats_{reference}': selected[scope] > comparison[scope]
             for scope in ['full', 'first_half', 'second_half']
             for reference, comparison in [('anchor', anchor), ('matched_canonical_control', matched)]}
    return {'selected_rounds': rounds, 'selected_weight': weight, 'selected_scores': selected,
            'matched_control_scores': matched, 'gate_checks': gates,
            'eligible_for_later_validation': bool(weight > 0 and all(gates.values()))}


def resource_gate(protocol: dict, started: float, entry: bool = False) -> dict:
    """Observe headroom between native fits/blocks; no claim of hard RSS limits."""
    free = base.memory_free_percent()
    peak = base.peak_rss()
    elapsed = time.monotonic() - started
    assert free >= protocol['resources']['entry_free_percent' if entry else 'ongoing_free_percent'], 'Insufficient measured memory headroom'
    assert peak <= protocol['resources']['observed_peak_limit_bytes'], 'Observed process RSS ceiling exceeded'
    assert elapsed <= protocol['resources']['observed_wall_limit_seconds'], 'Observed screen time ceiling exceeded'
    return {'free_percent': free, 'peak_rss_bytes': peak, 'elapsed_seconds': elapsed}


def predict_replay(model, checkpoint: Path, features: np.ndarray, rounds: int, units: int) -> np.ndarray:
    """Use bounded prediction chunks and compare every saved-checkpoint byte."""
    restored = lgb.Booster(model_file=str(checkpoint))
    assert restored.num_trees() == rounds and restored.num_feature() == 115
    result = np.empty(len(features), dtype=np.float64)
    for begin in range(0, len(features), 32768):
        block = np.asarray(features[begin:begin + 32768])
        value = model.booster_.predict(block, num_iteration=rounds, num_threads=2)
        replay = restored.predict(block, num_threads=2)
        assert value.dtype == replay.dtype and value.tobytes() == replay.tobytes()
        result[begin:begin + len(value)] = value
    return result.reshape(-1, units)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--independent-audit', type=Path)
    parser.add_argument('--independent-audit-sha256')
    args = parser.parse_args()
    assert Path(args.run_id).name == args.run_id and args.run_id not in ['', '.', '..']
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {'status': 'initializing', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'run_id': args.run_id, 'protocol_sha256': args.protocol_sha256,
              'command': shlex.join([sys.executable, *sys.argv]), 'environment': base.environment(),
              'screen_labels_opened_utc': None, 'maximum_training_target_year': 2019,
              'models': {}, 'kaggle_submission': None, 'reused_development_screen': True}
    try:
        assert base.sha(args.protocol) == args.protocol_sha256
        protocol = json.loads(args.protocol.read_text())
        original, loader_protocol, metadata, (train, screen, sample) = check_closure(protocol)
        shutil.copy2(args.protocol, out / 'protocol.json')
        for name in protocol['source']:
            destination = out / 'source' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
        if args.preflight:
            report.update(status='passed_preflight_only', labels_decoded=False, feature_build=False,
                          exact_sample_bytes=True, exact_original_stage=True)
            return
        assert args.independent_audit and base.sha(args.independent_audit) == args.independent_audit_sha256
        audit = json.loads(args.independent_audit.read_text())
        assert audit['status'] == 'passed' and audit['protocol_sha256'] == args.protocol_sha256
        report['independent_audit_sha256'] = args.independent_audit_sha256
        report['resources_before_build'] = resource_gate(protocol, started, entry=True)
        loader = import_path(ROOT / 'scripts/cuda_xgb664_loader.py', 'canonical_frozen664_loader')
        x, labels, loaded_meta, _, feature_meta, _ = loader.build_verified_features(ROOT / original['stage'], 2019, loader_protocol)
        assert x.columns.tolist() == protocol['prefix_columns'] and x.shape == (3647 * 168, 664)
        assert loaded_meta['Run No'].tolist() == metadata['Run No'].tolist()
        # Additive modules are imported only after the loader's foreign-module guard.
        adapter = import_path(ROOT / 'sintef_project/canonical_unit_features.py', 'canonical_additive_adapter')
        spec = adapter.feature_specification(protocol['prefix_columns'], protocol['units'])
        assert spec == protocol['canonical_specification']
        screen_rows = (screen[:, None] * 168 + np.arange(168)).ravel()
        units = protocol['units']
        y = np.column_stack([labels[unit] for unit in units])[sample].astype(np.uint8)
        assert y.shape == (122304, 14) and np.isin(y, [0, 1]).all()
        np.save(out / 'training_labels.npy', y, allow_pickle=False)
        shutil.copy2(ROOT / protocol['sample']['path'], out / 'train_sample_2020.npy')
        report['feature_records'] = {
            'train': adapter.write_canonical_rows(x, sample, out / 'train_canonical.npy', spec, block_rows=2048),
            'screen': adapter.write_canonical_rows(x, screen_rows, out / 'screen_canonical.npy', spec, block_rows=2048)}
        report.update(feature_specification=spec, prefix_feature_specification=feature_meta['feature_spec'],
                      training_ids=protocol['train_ids'], screen_ids=protocol['screen_ids'],
                      sample_sha256=protocol['sample']['sha256'], sampled_hours=122304, pooled_training_rows=122304 * 14)
        del x, labels, feature_meta, loaded_meta
        gc.collect()
        report['resources_after_prefix_release'] = resource_gate(protocol, started)
        xt = np.load(out / 'train_canonical.npy', mmap_mode='r', allow_pickle=False)
        xv = np.load(out / 'screen_canonical.npy', mmap_mode='r', allow_pickle=False)
        controls = np.empty((360 * 168, 14), dtype=np.float64)
        categorical = [spec['columns'].index(name) for name in spec['categorical_columns']]
        assert categorical == [111, 112]
        def callback(env):
            if (env.iteration + 1) % 25 == 0:
                observed = resource_gate(protocol, started)
                print(json.dumps({'event': 'round_progress', 'round': env.iteration + 1, **observed}), flush=True)
        for j, unit in enumerate(units):
            resource_gate(protocol, started)
            tick = time.monotonic()
            model = lgb.LGBMClassifier(**protocol['control_parameters'])
            model.fit(np.asarray(xt[j::14]), y[:, j], categorical_feature=categorical, callbacks=[callback])
            assert model.booster_.num_trees() == 350 and model.booster_.num_feature() == 115
            checkpoint = out / f'control_{unit}.txt'
            model.booster_.save_model(str(checkpoint))
            controls[:, j] = predict_replay(model, checkpoint, xv[j::14], 350, 1).ravel()
            report['models'][unit] = {'trees': 350, 'features': 115, 'checkpoint_sha256': base.sha(checkpoint),
                                      'full_screen_replay_bitwise': True, 'runtime_seconds': time.monotonic() - tick}
            print(json.dumps({'event': 'control_complete', 'unit': unit, **report['models'][unit]}), flush=True)
            del model
            gc.collect()
        resource_gate(protocol, started)
        tick = time.monotonic()
        model = lgb.LGBMClassifier(**protocol['pooled_parameters'])
        model.fit(xt, y.reshape(-1), categorical_feature=categorical, callbacks=[callback])
        assert model.booster_.num_trees() == 1400 and model.booster_.num_feature() == 115
        candidates = {}
        for rounds in ROUNDS:
            checkpoint = out / f'pooled_{rounds}.txt'
            model.booster_.save_model(str(checkpoint), num_iteration=rounds)
            candidates[rounds] = predict_replay(model, checkpoint, xv, rounds, 14).reshape(360, 168, 14)
            report['models'][f'pooled_{rounds}'] = {'trees': rounds, 'features': 115,
                'checkpoint_sha256': base.sha(checkpoint), 'full_screen_replay_bitwise': True,
                'single_shared_1400_round_trajectory': True}
            np.save(out / f'raw_pooled_{rounds}.npy', candidates[rounds], allow_pickle=False)
        report['pooled_trajectory_seconds'] = time.monotonic() - tick
        controls = controls.reshape(360, 168, 14)
        np.save(out / 'raw_control.npy', controls, allow_pickle=False)
        del model, xt, xv, y
        gc.collect()
        seal = {'status': 'locked', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                'protocol_sha256': args.protocol_sha256, 'maximum_training_target_year': 2019,
                'models': report['models'], 'raw_predictions': {name: base.sha(out / name)
                for name in ['raw_control.npy', 'raw_pooled_700.npy', 'raw_pooled_1400.npy']},
                'all_screen_prediction_replays_bitwise': True}
        (out / 'prediction_seal.json').write_text(json.dumps(seal, indent=2))
        report['prediction_seal_sha256'] = base.sha(out / 'prediction_seal.json')
        report['screen_labels_opened_utc'] = datetime.now(timezone.utc).isoformat()
        ids = metadata.iloc[screen]['Run No'].to_numpy()
        truth = read_screen_labels(ROOT / 'sintef_project/data/kernel/Unit_commitment_decisions.csv', ids, units)
        post = import_path(ROOT / 'sintef_project/postprocess.py', 'canonical_frozen_projection')
        with np.load(ROOT / protocol['raw_anchor'], allow_pickle=False) as saved:
            assert 'labels' not in saved.files and saved['units'].tolist() == units and saved['case_ids'].tolist() == ids.tolist()
            anchor = post.project_vinje(saved['probability'].reshape(-1, 14), units).reshape(360, 168, 14)
        with np.load(ROOT / protocol['projected_anchor'], allow_pickle=False) as saved:
            assert saved['units'].tolist() == units and saved['case_ids'].tolist() == ids.tolist()
            assert saved['labels'].dtype == truth.dtype and saved['labels'].tobytes() == truth.tobytes()
            assert saved['predictions'].dtype == anchor.dtype and saved['predictions'].tobytes() == anchor.tobytes()
        dates = metadata.iloc[screen].starttime.tolist()
        anchor_scores = scores(anchor, truth, dates)
        grid, control_scores = {}, {}
        for weight in WEIGHTS:
            matched = post.project_vinje(((1 - weight) * anchor + weight * controls).reshape(-1, 14), units).reshape(truth.shape)
            control_scores[str(weight)] = scores(matched, truth, dates)
            for rounds in ROUNDS:
                value = post.project_vinje(((1 - weight) * anchor + weight * candidates[rounds]).reshape(-1, 14), units).reshape(truth.shape)
                grid[f'{rounds}:{weight}'] = scores(value, truth, dates)
        decision = select_candidate(grid, control_scores, anchor_scores)
        rounds, weight = decision['selected_rounds'], decision['selected_weight']
        selected = post.project_vinje(((1 - weight) * anchor + weight * candidates[rounds]).reshape(-1, 14), units).reshape(truth.shape)
        np.savez_compressed(out / 'selected_screen_2020.npz', predictions=selected, labels=truth, case_ids=ids, units=np.asarray(units))
        lock = {'version': 'canonical_unit115_screen_lock_v1', 'locked_utc': datetime.now(timezone.utc).isoformat(),
                'protocol_sha256': args.protocol_sha256, 'prediction_seal_sha256': report['prediction_seal_sha256'],
                'weights': WEIGHTS, 'rounds': ROUNDS, 'scores': grid, 'control_scores': control_scores,
                'anchor_scores': anchor_scores, **decision}
        (out / 'selection_2020.json').write_text(json.dumps(lock, indent=2))
        report.update(status='complete', **decision, grid=grid, control_scores=control_scores, anchor_scores=anchor_scores,
                      selection_lock_sha256=base.sha(out / 'selection_2020.json'),
                      all_14_controls_and_two_pooled_checkpoints_replayed=True, no_2021_or_later_targets_decoded=True)
        print(json.dumps({'status': 'complete', **decision, 'anchor_scores': anchor_scores}), flush=True)
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report.update(runtime_seconds=time.monotonic() - started, peak_rss_bytes=base.peak_rss())
        report['artifacts'] = {p.relative_to(out).as_posix(): base.sha(p) for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        (out / 'experiment_record.md').write_text(
            f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus {report["status"]}. '
            f'Command `{report["command"]}`. Frozen canonical115 from-scratch pooled BCE, one 1,400-round trajectory '
            'with 700/1,400 checkpoints and 14 matched 350-round per-unit controls; exact pre-2020 122,304-hour sample, '
            'six-day purge and post-fit 2020 full/half-year gates. Reused development screen; no later target values, '
            f'promotion or submission. Runtime {report["runtime_seconds"]:.3f}s; peak RSS {report["peak_rss_bytes"]}. '
            f'Exact source/data/environment/checkpoints and result hashes: `artifacts/{args.run_id}/manifest.json`.\n')


if __name__ == '__main__':
    main()
