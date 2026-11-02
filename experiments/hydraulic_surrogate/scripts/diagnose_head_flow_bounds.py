"""Label-free pre2021 falsification of the frozen LP's head-dependent flow bounds."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import shlex
import shutil
import sys
import time
import traceback
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sintef_project.lp_features import HydroCaseData, UNIT_ORDER
from sintef_project.nonlinear_pq import physical_on_curve, original_power, digest
from sintef_project.physics import POOLS


def load_arrays(path: Path) -> dict:
    """Read saved numeric physical results without targets or executable objects."""
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved}


def main() -> None:
    """Measure physical capacity disagreement without solving LPs or reading labels."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    report = {'run_id': args.run_id, 'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'status': 'initializing',
              'command': shlex.join([sys.executable, *sys.argv]), 'training': False, 'solver_runs': 0,
              'labels_decoded': False, 'kaggle_submission': None, 'seed': None,
              'environment': {'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
                  'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'PyYAML']}}}
    try:
        diagnostic = ROOT / 'artifacts/lp-shop-relaxation-diagnostic-001'
        shop = ROOT / 'artifacts/lp-shop-full-001'
        base = ROOT / 'artifacts/lp-full-001'
        manifests = {name: json.loads((path / 'manifest.json').read_text()) for name, path in
                     [('diagnostic', diagnostic), ('shop', shop), ('base', base)]}
        if (digest(diagnostic / 'manifest.json') != '95ca40b0ff3b29f6cb36bb55b5539dcdcd3839fd05ca4f9a222d447a7821800e'
                or digest(shop / 'manifest.json') != 'e3d9f53ec7ac67a0afcc48c9630a215ad0a2be4abbfc07aa03cdc3a4345d8163'):
            raise ValueError('Frozen physical cache changed')
        sources = ['scripts/diagnose_head_flow_bounds.py', 'sintef_project/lp_features.py',
                   'sintef_project/nonlinear_pq.py', 'sintef_project/physics.py', 'sintef_project/features.py']
        protocol = {'hypothesis': 'Independent LP qmax can exceed head-dependent turbine endpoint limits; corrected network dispatch may differ even when physical-DP profit features failed.',
                    'case_ids': manifests['diagnostic']['case_ids'], 'units': UNIT_ORDER,
                    'data': manifests['shop']['data'], 'source': {name: digest(ROOT / name) for name in sources},
                    'cache_manifests': {name: digest(path / 'manifest.json') for name, path in [('diagnostic', diagnostic), ('shop', shop), ('base', base)]},
                    'gate': {'capacity_ratio_above': 1.01, 'minimum_case_unit_fraction': .1,
                             'or_minimum_dispatched_hour_fraction': .01, 'flow_tolerance_m3s': 1e-6},
                    'scope': '36 metadata-selected cases ending before2021, no label values or new LP solves; all14 units with own-flow losses only. Tokke/Vinje shared-loss corrected control remains a distinct model.',
                    'interpretation': 'A positive result supports a small separately frozen network re-solve diagnostic, not a trained-model gain or exact teacher reconstruction.'}
        (out / 'protocol_before_inputs.json').write_text(json.dumps(protocol, indent=2))
        report['protocol_sha256'] = digest(out / 'protocol_before_inputs.json')
        original_read = pd.read_csv
        label_path = (ROOT / 'sintef_project/data/kernel/Unit_commitment_decisions.csv').resolve()
        def guarded(path, *positional, **kwargs):
            if isinstance(path, (str, Path)) and Path(path).resolve() == label_path:
                if positional or kwargs != {'usecols': ['Run No', 'starttime']}:
                    raise ValueError('Capacity screen may read target metadata only')
            return original_read(path, *positional, **kwargs)
        rows = []
        with patch.object(pd, 'read_csv', guarded):
            inputs = HydroCaseData(ROOT)
            for case_id in protocol['case_ids']:
                index = int(np.flatnonzero(inputs.cases['Run No'].to_numpy() == case_id)[0])
                metadata = inputs.cases.iloc[index]
                if metadata.starttime + pd.Timedelta(hours=167) >= pd.Timestamp('2021-01-01', tz='UTC'):
                    raise ValueError('Case exceeds pre2021 scope')
                old_path, new_path, case_path = base / f'case_{case_id:04d}.npz', shop / f'case_{case_id:04d}_result.npz', shop / f'case_{case_id:04d}_inputs.npz'
                for path, manifest in [(old_path, manifests['base']), (new_path, manifests['shop']), (case_path, manifests['shop'])]:
                    if digest(path) != manifest['artifacts'][path.name]:
                        raise ValueError('Saved physical case changed')
                old, new, case = load_arrays(old_path), load_arrays(new_path), load_arrays(case_path)
                volume = inputs.volume.loc[metadata.starttime]
                heads = {name: float(np.interp(volume[name], record['vol_head']['x'], record['vol_head']['y']))
                         for name, record in inputs.model['reservoir'].items()}
                for j, unit in enumerate(UNIT_ORDER):
                    plant_name = unit.rsplit('_G', 1)[0]
                    plant, generator = inputs.model['plant'][plant_name], inputs.model['generator'][unit]
                    source_head = np.mean([heads[name] for name in POOLS[plant_name]]) if plant_name == 'Songa' else heads[POOLS[plant_name][0]]
                    outlet = max(plant['outlet_line'], heads.get(inputs.plant_downstream_reservoir.get(plant_name), -np.inf))
                    gross = max(float(source_head - outlet), 1.)
                    curve = physical_on_curve(generator, plant, gross)
                    qmax_old, qmax_legal = float(case['qmax'][j]/.0036), curve['qmax_m3s']
                    power, net_head, _, turbine_upper = original_power(np.asarray([qmax_old]), generator, plant, gross)
                    qbase, qshop = old['flow'][j]/.0036, new['flow'][j]/.0036
                    rows.append({'case_id': case_id, 'year': metadata.starttime.year, 'unit': unit, 'initial_gross_head_m': gross,
                        'old_qmax_m3s': qmax_old, 'legal_qmax_m3s': qmax_legal, 'capacity_ratio': qmax_old/qmax_legal,
                        'old_qmin_m3s': float(case['qmin'][j]/.0036), 'legal_qmin_m3s': curve['qmin_m3s'],
                        'net_head_at_old_qmax_m': float(net_head[0]), 'interpolated_turbine_max_at_old_qmax_m3s': float(turbine_upper[0]),
                        'old_qmax_turbine_violation_m3s': max(0., qmax_old-float(turbine_upper[0])),
                        'base_hours_above_legal_qmax': int(np.sum(qbase > qmax_legal+1e-6)),
                        'shop_hours_above_legal_qmax': int(np.sum(qshop > qmax_legal+1e-6)),
                        'base_max_flow_ratio': float(qbase.max()/qmax_legal), 'shop_max_flow_ratio': float(qshop.max()/qmax_legal),
                        'physical_best_factor_mwh_per_mm3': float(np.max(curve['power_mw']/curve['q_m3s'])/.0036),
                        'old_best_factor_mwh_per_mm3': float(case['energy'][j, 0])})
        frame = pd.DataFrame(rows)
        frame.to_csv(out / 'case_unit_bound_audit.csv', index=False)
        case_fraction = float(np.mean(frame.capacity_ratio > 1.01))
        base_fraction = float(frame.base_hours_above_legal_qmax.sum()/(len(frame)*168))
        shop_fraction = float(frame.shop_hours_above_legal_qmax.sum()/(len(frame)*168))
        report.update(status='complete', case_units=len(frame), cases=len(protocol['case_ids']),
                      capacity_ratio_above_1_01_fraction=case_fraction, base_dispatched_violation_fraction=base_fraction,
                      shop_dispatched_violation_fraction=shop_fraction, maximum_capacity_ratio=float(frame.capacity_ratio.max()),
                      screen_passed=case_fraction >= .1 or max(base_fraction, shop_fraction) >= .01,
                      per_unit=frame.groupby('unit').agg(maximum_capacity_ratio=('capacity_ratio','max'),
                          average_capacity_ratio=('capacity_ratio','mean'), base_hours=('base_hours_above_legal_qmax','sum'),
                          shop_hours=('shop_hours_above_legal_qmax','sum')).to_dict('index'))
        report['source'], report['data'] = protocol['source'], protocol['data']
        for name in sources:
            target = out / 'source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report['runtime_seconds'] = time.monotonic()-start
        report['peak_rss_bytes'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform == 'darwin' else 1024)
        report['artifacts'] = {p.relative_to(out).as_posix(): digest(p) for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        (out / 'experiment_record.md').write_text(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus {report["status"]}. Label-free head-dependent turbine-bound diagnostic on36metadata-selected pre2021cases, all14units. No LPsolve/fit/targetvalues. Command `{report["command"]}`. Screen {report.get("screen_passed")}; original capacityratio>1.01 in {report.get("capacity_ratio_above_1_01_fraction")} ofcase-unitpairs, dispatchedviolationfractions baseline {report.get("base_dispatched_violation_fraction")}, SHOP {report.get("shop_dispatched_violation_fraction")}. Full source/data/cache/formula/gate/environment/CSVproof `artifacts/{args.run_id}/manifest.json`; runtime {report["runtime_seconds"]:.3f}s. No validation/leaderboard claim.\n')
        print(json.dumps({key: report.get(key) for key in ['status', 'screen_passed', 'maximum_capacity_ratio', 'capacity_ratio_above_1_01_fraction', 'base_dispatched_violation_fraction', 'shop_dispatched_violation_fraction', 'runtime_seconds']}), flush=True)


if __name__ == '__main__':
    main()
