"""Freeze a small input-only nonlinear PQ/DP cache without fitting a learner."""
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shlex
import shutil
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sintef_project.nonlinear_pq import COLUMNS, TARGET_UNITS, VERSION, NonlinearPQInputs, digest


def main() -> None:
    """Build a new immutable cache after independent physical/DP tests."""
    out = ROOT / 'artifacts/nonlinear-pq-features-001'
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    sources = ['sintef_project/nonlinear_pq.py', 'sintef_project/features.py',
               'sintef_project/physics.py', 'sintef_project/lp_features.py',
               'scripts/build_nonlinear_pq_features.py']
    inputs = NonlinearPQInputs(ROOT, ROOT / 'artifacts/lp-full-001')
    report = {
        'run_id': out.name, 'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'version': VERSION,
        'command': shlex.join([sys.executable, *sys.argv]), 'status': 'running',
        'seed': None, 'training': 'none; legal inputs and frozen input-only LP duals',
        'environment': {'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
                        'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'pyyaml']}},
        'source': {name: digest(ROOT / name) for name in sources}, 'data': inputs.lp_manifest['data'],
        'lp_manifest_sha256': digest(ROOT / 'artifacts/lp-full-001/manifest.json'),
        'columns': COLUMNS, 'units': TARGET_UNITS, 'case_ids': inputs.cases['Run No'].tolist(),
        'case_dates': inputs.cases.starttime.astype(str).tolist(),
        'formula': 'On reward=max_q(price*original_unhulled_P(q)-.0036*q*(upstream_dual-downstream_dual)); physical on-domain only, off separate. Full168h max-sum DP with supplied start/stop costs and free boundaries. Reward and DP advantage divided by (1+mean(abs(price)))*nominalPmax. Sixfloat32 columns.',
        'curve': '257-point global scan plus efficiency knots and numerical roots of effective-head, flow and Pmin/Pmax limits; then257points over physicalonspan plus all feasibleknots/roots; recheck every retainedpoint; eightgenerator-efficiency fixedpointiterations; no convexification or roundingofhead.',
        'limitations': 'Fixed initial head and frozen LPv1 water shadow; no reoptimized water budgets, environmental constraints or initial state. Currency assumption inherited. No hydraulic-feasibility or exactSHOP reconstruction claim.',
        'cases': [], 'kaggle_submission': None,
    }
    for name in sources:
        target = out / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    (out / 'manifest.json').write_text(json.dumps(report, indent=2))
    try:
        first_start = time.monotonic()
        first, first_raw = inputs.one_case(0)
        report['one_case_seconds'] = time.monotonic() - first_start
        print('ONE_CASE', report['one_case_seconds'], 'shape', first.shape, flush=True)
        values = np.empty((len(inputs.cases), 168, 6), dtype=np.float32)
        rewards = np.empty((len(inputs.cases), 168, 3), dtype=np.float64)
        for index in range(len(inputs.cases)):
            case_started = time.monotonic()
            data, raw = (first, first_raw) if index == 0 else inputs.one_case(index)
            values[index], rewards[index] = data, raw.pop('raw_on_profit_eur')
            report['cases'].append({'case_id': int(inputs.cases.iloc[index]['Run No']),
                                    **raw, 'seconds': time.monotonic() - case_started})
            if index % 500 == 0:
                print('CASE', index + 1, '/', len(inputs.cases), flush=True)
        np.savez_compressed(out / 'features.npz', values=values, columns=np.asarray(COLUMNS),
                            case_ids=inputs.cases['Run No'].to_numpy(), dates=inputs.cases.starttime.astype(str).to_numpy(dtype=str))
        np.savez_compressed(out / 'raw_on_rewards.npz', rewards=rewards, units=np.asarray(TARGET_UNITS),
                            case_ids=inputs.cases['Run No'].to_numpy())
        report.update(status='complete', shape=list(values.shape), distinct_unit_head_curves=len(inputs.curves))
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report['runtime_seconds'] = time.monotonic() - started
        report['artifacts'] = {str(p.relative_to(out)): digest(p) for p in out.rglob('*')
                               if p.is_file() and p.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({k: report.get(k) for k in ['status', 'runtime_seconds', 'shape']}), flush=True)


if __name__ == '__main__':
    main()
