"""Separate original-environment664 feature build; never train an XGB model."""
from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']: os.environ[key] = '2'
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
import xgboost as xgb
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.cpu_vector14_fold_common import read_protocol, review, validate_lock, artifacts, sha
from scripts.cpu_vector14_portability import write_dataframe_rows
from scripts.train_lgb_shop664 import check_closure, memory_free_percent, peak_rss, environment


def main() -> None:
    """Materialize only original sampled rows after independent review and explicit build."""
    ap = argparse.ArgumentParser()
    for name in ['protocol', 'independent-audit', 'selection-lock', 'lock-audit']: ap.add_argument('--' + name, type=Path, required=name in ['protocol', 'independent-audit'])
    for name in ['protocol-sha256', 'independent-audit-sha256', 'run-id']: ap.add_argument('--' + name, required=True)
    for name in ['selection-lock-sha256', 'lock-audit-sha256']: ap.add_argument('--' + name)
    ap.add_argument('--year', type=int, choices=[2021, 2022], required=True); ap.add_argument('--execute', action='store_true')
    a = ap.parse_args(); assert a.execute and Path(a.run_id).name == a.run_id and a.run_id not in ['', '.', '..']
    out = ROOT / 'artifacts' / a.run_id; out.mkdir(exist_ok=False); start = time.monotonic()
    m = {'status': 'initializing', 'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'year': a.year,
         'command': [sys.executable, *sys.argv], 'training': False, 'protocol_sha256': a.protocol_sha256}
    try:
        p = read_protocol(a.protocol, a.protocol_sha256); review(a.independent_audit, a.independent_audit_sha256, a.protocol_sha256)
        if a.year == 2022:
            validate_lock(a.selection_lock, a.selection_lock_sha256, p, a.protocol_sha256, a.lock_audit, a.lock_audit_sha256)
            m.update(selection_lock_sha256=a.selection_lock_sha256, lock_audit_sha256=a.lock_audit_sha256)
        else: assert a.selection_lock is None and a.lock_audit is None
        original = json.loads((ROOT / p['original_local_protocol']).read_text())
        loader_protocol, metadata, folds = check_closure(original)
        assert environment() == original['environment'] and xgb.__version__ == '3.2.0'
        assert memory_free_percent() >= p['resources']['minimum_free_percent_before_build']
        spec = importlib.util.spec_from_file_location('cpu_vector_original664_loader', ROOT / 'scripts/cuda_xgb664_loader.py')
        loader = importlib.util.module_from_spec(spec); spec.loader.exec_module(loader)
        x, labels, train, _, meta, _ = loader.build_verified_features(ROOT / original['stage'], a.year, loader_protocol)
        assert x.columns.tolist() == p['feature_columns'] and x.shape == (3647 * 168, 664)
        assert train['Run No'].tolist() == metadata['Run No'].tolist()
        split = folds[str(a.year)]; sample = split['sample']; validation = split['validation_rows']
        assert len(sample) == p['sample_rows'][str(a.year)] and len(validation) == 365 * 168
        y = np.column_stack([labels[u] for u in p['units']]).astype(np.uint8)
        assert np.isin(y[sample], [0, 1]).all() and np.isin(y[validation], [0, 1]).all()
        records = {}
        for name, rows in [('train', sample), ('validation', validation)]:
            records[name] = write_dataframe_rows(x, rows, out / f'x_{name}.npy', p['feature_columns'], xgb)
            assert peak_rss() <= p['resources']['maximum_build_peak_rss_bytes']
        np.save(out / 'y_train.npy', y[sample], allow_pickle=False)
        np.save(out / 'train_sample.npy', sample, allow_pickle=False)
        np.save(out / 'validation_rows.npy', validation, allow_pickle=False)
        ids = train.iloc[split['validation_cases']]['Run No'].to_numpy()
        np.savez_compressed(out / 'validation_targets.npz', labels=y[validation], case_ids=ids, units=np.asarray(p['units']))
        assert sha(out / 'train_sample.npy') == p['samples'][str(a.year)]['sha256']
        (out / 'schema.json').write_text(json.dumps({'feature_columns': p['feature_columns'], 'units': p['units'],
            'train_ids': p['folds'][str(a.year)]['train_ids'], 'validation_ids': ids.tolist()}, indent=2) + '\n')
        m.update(status='complete', environment=environment(), records=records, feature_spec=meta['feature_spec'],
                 exact_original_sample_bytes=True, all_dataframe_native_model_inputs_bitwise=True,
                 maximum_decoded_target_year=a.year, training_rows=len(sample), validation_rows=len(validation),
                 independent_audit_sha256=a.independent_audit_sha256)
    except BaseException:
        m.update(status='failed', error=traceback.format_exc()); raise
    finally:
        shutil.copy2(__file__, out / Path(__file__).name); shutil.copy2(a.protocol, out / 'protocol.json')
        m.update(runtime_seconds=time.monotonic() - start, peak_rss_bytes=peak_rss(), artifacts=artifacts(out))
        (out / 'manifest.json').write_text(json.dumps(m, indent=2) + '\n'); print(m['status'], sha(out / 'manifest.json'), flush=True)


if __name__ == '__main__': main()
