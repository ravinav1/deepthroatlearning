"""Bounded synthetic timing for canonical shared/per-unit trees; no real data."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback

for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[key] = '2'

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sintef_project.canonical_unit_features import feature_specification
from scripts.train_lgb_shop664 import environment, memory_free_percent, sha


def main() -> None:
    """Fit two fixed64-tree synthetic models and report cautious scaling estimates."""
    started = time.monotonic()
    out = ROOT / 'artifacts/canonical-pooled-timing-001'
    out.mkdir(exist_ok=False)
    p = json.loads((ROOT / 'docs/lgb_shop664_protocol.json').read_text())
    spec = feature_specification(p['feature_columns'], p['units'])
    params = {**p['model_parameters'], 'n_estimators': 64}
    report = {'status': 'initializing', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': [sys.executable, *sys.argv], 'synthetic_only': True,
              'competition_inputs_or_labels_read': False, 'environment': environment(),
              'rows': 32768, 'features': 115, 'rounds': 64, 'parameters': params,
              'source': {name: sha(ROOT / name) for name in
                         ['scripts/time_canonical_pooled.py', 'sintef_project/canonical_unit_features.py']},
              'metadata_protocol_sha256': sha(ROOT / 'docs/lgb_shop664_protocol.json'),
              'seed': 4173, 'results': {}, 'model_scores': None, 'kaggle_submission': None}
    try:
        assert memory_free_percent() >= 40
        rng = np.random.default_rng(4173)
        base = rng.standard_normal((32768, 115), dtype=np.float32)
        identities = np.asarray(spec['identity_values'], dtype=np.float32)
        for name in ['pooled', 'one_unit']:
            x = base.copy()
            groups = np.arange(len(x)) % 14 if name == 'pooled' else np.full(len(x), 10)
            x[:, 111:] = identities[groups]
            logits = x[:, 53] + .4 * x[:, 55] - .25 * x[:, 72] + .1 * (groups - 6)
            y = (rng.random(len(x)) < 1 / (1 + np.exp(-logits))).astype(np.uint8)
            np.save(out / f'{name}_x.npy', x, allow_pickle=False)
            np.save(out / f'{name}_y.npy', y, allow_pickle=False)
            tick = time.monotonic()
            model = lgb.LGBMClassifier(**params)
            model.fit(x, y, categorical_feature=[111, 112])
            seconds = time.monotonic() - tick
            checkpoint = out / f'{name}.txt'
            model.booster_.save_model(str(checkpoint))
            first = model.predict_proba(x)[:, 1]
            replay = lgb.Booster(model_file=str(checkpoint)).predict(x, num_threads=2)
            assert first.dtype == replay.dtype and first.tobytes() == replay.tobytes()
            assert model.booster_.num_feature() == 115 and model.booster_.num_trees() == 64
            np.save(out / f'{name}_prediction.npy', first, allow_pickle=False)
            report['results'][name] = {'fit_seconds': seconds, 'checkpoint_sha256': sha(checkpoint),
                                       'full_synthetic_replay_bitwise': True}
            del model, first, replay, x, y
            assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= 2 * 1024 ** 3
            assert time.monotonic() - started < 120
        pooled_seconds = report['results']['pooled']['fit_seconds']
        unit_seconds = report['results']['one_unit']['fit_seconds']
        report.update(status='complete', rough_row_round_extrapolation_seconds={
            'pooled700': pooled_seconds * (122304 * 14 / 32768) * 700 / 64,
            'pooled1400': pooled_seconds * (122304 * 14 / 32768) * 1400 / 64,
            'all14_unit350': unit_seconds * (122304 / 32768) * 350 / 64 * 14},
            limitations='Synthetic timing only; extrapolation does not model real bin counts, splits or memory.2GiB/120s checks occur after each short model, not as hard allocation/wall limits. No actual prediction gain or full-data resource guarantee.')
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        shutil.copy2(__file__, out / Path(__file__).name)
        report.update(runtime_seconds=time.monotonic() - started,
                      peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        report['artifacts'] = {q.name: sha(q) for q in out.iterdir() if q.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'status': report['status'], 'sha256': sha(out / 'manifest.json'),
                          'results': report['results'],
                          'extrapolation_seconds': report.get('rough_row_round_extrapolation_seconds')}), flush=True)


if __name__ == '__main__':
    main()
