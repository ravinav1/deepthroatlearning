"""Twelve-case pre2021 screen of integer states coupled to the full water budget."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from unittest.mock import patch

for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[variable] = '1'

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sintef_project.lp_features import HydroCaseData, UNIT_ORDER
from sintef_project.integer_water_budget import IntegerWaterBudgetLP
from scripts.diagnose_lp_shop_relaxation import digest, metrics

TARGETS = ['Hogga_G1', 'Lio_G1', 'Haukeli_G1']
SOURCES = ['scripts/diagnose_integer_water_budget.py', 'sintef_project/integer_water_budget.py',
           'sintef_project/lp_shop_relaxation.py', 'sintef_project/lp_features.py', 'sintef_project/physics.py',
           'sintef_project/features.py', 'scripts/diagnose_lp_shop_relaxation.py',
           'tests/test_integer_water_budget.py', 'tests/test_lp_shop_relaxation.py', 'docs/integer_water_budget.md']


def arrays(path: Path) -> dict:
    """Read only numeric saved LP data."""
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z}


def freeze(path: Path) -> dict:
    """Freeze twelve dates, source/data identities, fixed costs and every screen gate."""
    cache = ROOT / 'artifacts/lp-shop-full-001'
    m = json.loads((cache / 'manifest.json').read_text())
    diagnostic = json.loads((ROOT / 'artifacts/lp-shop-relaxation-diagnostic-001/manifest.json').read_text())
    positions = [0, 4, 7, 11, 12, 16, 19, 23, 24, 28, 31, 35]
    protocol = {'version': 'integer_water_budget_screen_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'case_ids': [diagnostic['case_ids'][i] for i in positions], 'blocks': ['2015_2018']*4 + ['2019']*4 + ['2020']*4,
        'targets': TARGETS, 'source': {name: digest(ROOT / name) for name in SOURCES}, 'data': m['data'],
        'cache': 'artifacts/lp-shop-full-001', 'cache_manifest_sha256': digest(cache / 'manifest.json'),
        'milp_seconds': 8., 'conditional_lp_seconds_each': 1., 'mip_rel_gap': 0., 'threads': 1,
        'formulation': 'Exact saved linked supplied-cost network LP. Only three selected unit-state vectors become binary. All other states, constraints, segment curves, initial heads, terminal values and costs unchanged. No extra runtime/ramp constraints.',
        'replay': 'After one MILP, fix chosen binary states and solve identical LP twice with1second caps. Require bitwise solution and conditional-dual agreement. These duals are conditional, not global MILP shadow prices.',
        'time_limit': '8seconds MILP plus two1second LP caps percase; measured wall time includes Python/HiGHS overhead and is separately reported. Timeouts/unproven incumbents are retained and never pass promotion.',
        'gate': 'All12 cases must have a proven optimum and exact fixed-state replay, all slacks zero, physical feasibility/objective subset check. Weak3 cutoff errors reduce at least10%, no dateblock weak3 deterioration, no targetunit error rise over5%, all14 cutoff errors do not rise, and all14 Q-proxy AUC does not fall.',
        'selection': 'No parameter/threshold/unit/weight search. Metadata-selected cases only. Labels decoded only after all physical solves; no2021/2022 labels.'}
    with path.open('x') as stream:
        json.dump(protocol, stream, indent=2)
    return protocol


def main() -> None:
    """Run only the explicitly hash-pinned small diagnostic, preserving every failure."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze-protocol', action='store_true')
    parser.add_argument('--protocol', type=Path, default=ROOT / 'docs/integer_water_budget_protocol.json')
    parser.add_argument('--protocol-sha256')
    parser.add_argument('--run-id')
    args = parser.parse_args()
    if args.freeze_protocol:
        freeze(args.protocol)
        print(digest(args.protocol))
        return
    if args.run_id is None or args.protocol_sha256 is None:
        raise ValueError('Run ID and reviewed protocol SHA required')
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {'run_id': args.run_id, 'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'status': 'initializing',
        'command': shlex.join([sys.executable, *sys.argv]), 'training': False, 'kaggle_submission': None,
        'labels_decoded_utc': None, 'cases': [], 'environment': {'python': sys.version, 'platform': platform.platform(),
            'machine': platform.machine(), 'packages': {p: importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','PyYAML']}}}
    try:
        if digest(args.protocol) != args.protocol_sha256:
            raise ValueError('Protocol hash changed')
        protocol = json.loads(args.protocol.read_text())
        report['protocol_sha256'] = args.protocol_sha256
        if protocol['targets'] != TARGETS or len(protocol['case_ids']) != 12:
            raise ValueError('Frozen scope changed')
        for group in ['source', 'data']:
            for name, expected in protocol[group].items():
                if digest(ROOT / name) != expected:
                    raise ValueError(f'Frozen {group} changed: {name}')
        for name in SOURCES:
            target = out / 'source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        cache = ROOT / protocol['cache']
        if digest(cache / 'manifest.json') != protocol['cache_manifest_sha256']:
            raise ValueError('Original control cache changed')
        cached = json.loads((cache / 'manifest.json').read_text())
        original_read = pd.read_csv
        target_path = (ROOT / 'sintef_project/data/kernel/Unit_commitment_decisions.csv').resolve()
        def guarded(path, *positional, **kwargs):
            if isinstance(path, (str, Path)) and Path(path).resolve() == target_path:
                if positional or kwargs != {'usecols': ['Run No', 'starttime']}:
                    raise ValueError('Input-only solve stage may read metadata only')
            return original_read(path, *positional, **kwargs)
        candidates, controls, selected_cases, selected_blocks = [], [], [], []
        weak = [UNIT_ORDER.index(unit) for unit in TARGETS]
        with patch.object(pd, 'read_csv', guarded):
            inputs = HydroCaseData(ROOT)
            for case_id, block in zip(protocol['case_ids'], protocol['blocks']):
                tick = time.monotonic()
                index = int(np.flatnonzero(inputs.cases['Run No'].to_numpy() == case_id)[0])
                if inputs.cases.iloc[index].starttime + pd.Timedelta(hours=167) >= pd.Timestamp('2021-01-01', tz='UTC'):
                    raise ValueError('Case window exceeds pre2021 scope')
                paths = {kind: cache / f'case_{case_id:04d}_{kind}.npz' for kind in ['inputs','result']}
                for path in paths.values():
                    if digest(path) != cached['artifacts'][path.name]:
                        raise ValueError('Saved physical case changed')
                case, control = arrays(paths['inputs']), arrays(paths['result'])
                solver = IntegerWaterBudgetLP(inputs.network)
                record = {'case_id': case_id, 'block': block, 'input_sha256': digest(paths['inputs']),
                          'control_sha256': digest(paths['result'])}
                try:
                    result, proof = solver.solve_integer(case, weak, protocol['milp_seconds'], protocol['conditional_lp_seconds_each'])
                    if result['objective'] > control['objective'] + 1e-4:
                        raise ValueError('Integer subset objective exceeds original relaxed optimum')
                    np.savez_compressed(out / f'case_{case_id:04d}_integer.npz', **result)
                    record.update(status='complete', proof=proof, relaxed_objective=float(control['objective']),
                        integer_objective=float(result['objective']), objective_loss_eur=float(control['objective']-result['objective']),
                        all_slacks_zero=all(np.max(result[key]) == 0 for key in ['minimum_slack','maximum_slack','bypass_slack','vest_slack']))
                    candidates.append((result['flow']/case['qmin'][:,None], result['power']/case['pmin'][:,None]))
                    controls.append((control['flow']/case['qmin'][:,None], control['power']/case['pmin'][:,None]))
                    selected_cases.append(case_id)
                    selected_blocks.append(block)
                except Exception:
                    record.update(status='failed', error=traceback.format_exc(), proof=solver.last_integer_metadata)
                    np.savez_compressed(out / f'case_{case_id:04d}_failed_incumbent.npz', **solver.last_integer_arrays)
                record['runtime_seconds'] = time.monotonic()-tick
                report['cases'].append(record)
                (out / f'case_{case_id:04d}_proof.json').write_text(json.dumps(record, indent=2))
                print(json.dumps({'case_id': case_id, 'status': record['status'], 'mip_status': record['proof'].get('mip_status'),
                                  'proven_optimal': record['proof'].get('proven_optimal'), 'seconds': record['runtime_seconds']}), flush=True)
        (out / 'physical_results_before_labels.json').write_text(json.dumps(report, indent=2))
        if selected_cases:
            meta = pd.read_csv(target_path, usecols=['Run No','starttime'])
            keep = meta['Run No'].isin(selected_cases).to_numpy()
            columns = [f'result_committed_{unit}_t{hour}' for unit in UNIT_ORDER for hour in range(168)]
            truth_frame = pd.read_csv(target_path, skiprows=(np.flatnonzero(~keep)+1).tolist(),
                usecols=['Run No',*columns], dtype={column:'uint8' for column in columns})
            if truth_frame['Run No'].tolist() != selected_cases:
                raise ValueError('Selected target rows misaligned')
            truth = truth_frame[columns].to_numpy().reshape(len(selected_cases),14,168)
            cq, cp = np.asarray([v[0] for v in controls]), np.asarray([v[1] for v in controls])
            nq, npower = np.asarray([v[0] for v in candidates]), np.asarray([v[1] for v in candidates])
            baseline_metrics, candidate_metrics = metrics(truth,cq,cp,selected_blocks), metrics(truth,nq,npower,selected_blocks)
            be, ne = (cp >= .5) != truth, (npower >= .5) != truth
            weak_old, weak_new = int(be[:,weak].sum()), int(ne[:,weak].sum())
            block_gates = {block: int(ne[np.asarray(selected_blocks)==block][:,weak].sum()) <= int(be[np.asarray(selected_blocks)==block][:,weak].sum()) for block in set(selected_blocks)}
            unit_gates = {UNIT_ORDER[j]: int(ne[:,j].sum()) <= 1.05*int(be[:,j].sum()) for j in weak}
            proof_gate = len(report['cases']) == len(selected_cases) == 12 and all(
                r['proof']['proven_optimal'] and r['proof']['fixed_lp_replay_bitwise'] and r['all_slacks_zero'] for r in report['cases'])
            gates = {'all_12_proven_and_feasible': proof_gate, 'weak3_error_reduction_at_least_10percent': weak_new <= .9*weak_old,
                'no_block_weak3_deterioration': all(block_gates.values()), 'no_target_error_increase_above_5percent': all(unit_gates.values()),
                'all14_error_nonincrease': int(ne.sum()) <= int(be.sum()),
                'all14_qproxy_auc_nonincrease_in_error': candidate_metrics['micro_auc_qratio'] >= baseline_metrics['micro_auc_qratio']}
            report.update(labels_decoded_utc=datetime.now(timezone.utc).isoformat(), metrics={'control':baseline_metrics,'integer':candidate_metrics},
                weak3_errors={'control':weak_old,'integer':weak_new}, gates=gates, block_gates=block_gates, unit_gates=unit_gates,
                screen_passed=all(gates.values()), scored_case_ids=selected_cases)
            np.savez_compressed(out / 'diagnostic.npz', labels=truth, control_qratio=cq, control_pratio=cp,
                                integer_qratio=nq, integer_pratio=npower, case_ids=np.asarray(selected_cases), units=np.asarray(UNIT_ORDER))
        else:
            report.update(screen_passed=False, scored_case_ids=[], reason='No retained feasible incumbent; no target values read')
        report['status'] = 'complete'
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report['runtime_seconds'] = time.monotonic()-started
        report['peak_rss_bytes'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform == 'darwin' else 1024)
        if args.protocol.is_file():
            shutil.copy2(args.protocol, out / 'protocol.json')
        report['artifacts'] = {p.relative_to(out).as_posix(): digest(p) for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        (out / 'experiment_record.md').write_text(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus {report["status"]}. Fixed12-case pre2021 integer water-budget diagnostic, three prescribed binary unit states, same physical/economic LP control. Command `{report["command"]}`. Per-case8-second MILP + two1-second fixed-state LP caps; timeouts and unproven incumbents retained. No model fitting. Screen {report.get("screen_passed")}; weak3 errors {report.get("weak3_errors")}. Full protocol/source/data/inputs/results/solver bounds/statuses/labels/environment/runtime proof `artifacts/{args.run_id}/manifest.json`; runtime {report["runtime_seconds"]:.3f}s. No validation or Kaggle claim.\n')


if __name__ == '__main__':
    main()
