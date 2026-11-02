"""Finish only previously timed-out MILPs, preserving the original frozen screen."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
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
from scripts.diagnose_integer_water_budget import arrays, digest, metrics, TARGETS
from sintef_project.integer_water_budget import IntegerWaterBudgetLP
from sintef_project.lp_features import HydroCaseData, UNIT_ORDER

BASE = ROOT / 'artifacts/integer-water-budget-screen-001'
NEW_SOURCE = ['scripts/complete_integer_water_budget.py', 'docs/integer_water_budget_completion.md']


def freeze(path: Path) -> None:
    """Pin status-selected retries and every byte of the original completed screen."""
    m = json.loads((BASE / 'manifest.json').read_text())
    base_protocol = json.loads((BASE / 'protocol.json').read_text())
    assert m['status'] == 'complete' and not m['screen_passed']
    retry = [r['case_id'] for r in m['cases'] if r['proof']['mip_status'] == 1]
    reuse = [r['case_id'] for r in m['cases'] if r['proof']['proven_optimal']]
    assert retry == [1826, 1827, 1958, 2055] and len(reuse) == 8
    all_inputs = {'manifest.json': digest(BASE / 'manifest.json'), 'protocol.json': digest(BASE / 'protocol.json')}
    all_inputs.update(m['artifacts'])
    p = {'version': 'integer_water_budget_completion_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
         'base': str(BASE.relative_to(ROOT)), 'base_files': all_inputs,
         'base_protocol_sha256': m['protocol_sha256'], 'retry_case_ids': retry, 'reuse_case_ids': reuse,
         'case_ids': base_protocol['case_ids'], 'blocks': base_protocol['blocks'],
         'source': {name: digest(ROOT / name) for name in NEW_SOURCE},
         'milp_seconds': 60., 'conditional_lp_seconds_each': 1.,
         'selection': 'Only the four original solver-status-1 cases. Eight proven results reused byte-for-byte.',
         'scope': 'Additional solver effort only. Identical model, input, targets, gap, threads, gates and cutoffs. No automatic promotion or full cache.'}
    with path.open('x') as stream:
        json.dump(p, stream, indent=2)
    print(digest(path))


def validate(path: Path, expected: str) -> tuple[dict, dict, dict]:
    """Check immutable closure and the solver-status-only case partition."""
    if digest(path) != expected:
        raise ValueError('Completion protocol changed')
    p = json.loads(path.read_text())
    for name, sha in p['source'].items():
        assert digest(ROOT / name) == sha, name
    for name, sha in p['base_files'].items():
        assert digest(BASE / name) == sha, name
    m = json.loads((BASE / 'manifest.json').read_text())
    q = json.loads((BASE / 'protocol.json').read_text())
    assert digest(BASE / 'protocol.json') == p['base_protocol_sha256'] == m['protocol_sha256']
    for group in ['source', 'data']:
        for name, sha in q[group].items():
            assert digest(ROOT / name) == sha, name
    assert [r['case_id'] for r in m['cases'] if r['proof']['mip_status'] == 1] == p['retry_case_ids']
    assert [r['case_id'] for r in m['cases'] if r['proof']['proven_optimal']] == p['reuse_case_ids']
    assert len(p['retry_case_ids']) == 4 and len(p['reuse_case_ids']) == 8
    assert p['case_ids'] == q['case_ids'] and p['blocks'] == q['blocks']
    assert p['milp_seconds'] == 60. and p['conditional_lp_seconds_each'] == q['conditional_lp_seconds_each']
    return p, q, m


def main() -> None:
    """Save separate completion evidence without changing the initial diagnostic."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze-protocol', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--protocol', type=Path, default=ROOT / 'docs/integer_water_budget_completion_protocol.json')
    parser.add_argument('--protocol-sha256')
    parser.add_argument('--run-id')
    args = parser.parse_args()
    if args.freeze_protocol:
        freeze(args.protocol)
        return
    if args.protocol_sha256 is None or args.run_id is None:
        raise ValueError('Explicit frozen protocol hash and unique run ID are required')
    p, q, original = validate(args.protocol, args.protocol_sha256)
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {'run_id': args.run_id, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'initializing', 'command': shlex.join([sys.executable, *sys.argv]),
              'protocol_sha256': args.protocol_sha256, 'base_manifest_sha256': digest(BASE/'manifest.json'),
              'training': False, 'kaggle_submission': None, 'environment': original['environment'],
              'cases': [], 'reused_case_ids': p['reuse_case_ids'], 'retried_case_ids': p['retry_case_ids']}
    try:
        shutil.copy2(args.protocol, out / 'protocol.json')
        for name in list(p['source']) + list(q['source']):
            dest = out / 'source' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
        if args.preflight:
            report.update(status='passed_preflight_only', check='All source/data/base-artifact hashes and status-only 8/4 partition verified. No labels or solves.')
            return
        cache = ROOT / q['cache']
        assert digest(cache/'manifest.json') == q['cache_manifest_sha256']
        cached = json.loads((cache/'manifest.json').read_text())
        old_records = {r['case_id']: r for r in original['cases']}
        target_path = (ROOT / 'sintef_project/data/kernel/Unit_commitment_decisions.csv').resolve()
        old_read = pd.read_csv
        def guarded(path, *positional, **kwargs):
            if isinstance(path, (str, Path)) and Path(path).resolve() == target_path:
                if positional or kwargs != {'usecols': ['Run No', 'starttime']}:
                    raise ValueError('Physical continuation may read target metadata only')
            return old_read(path, *positional, **kwargs)
        with patch.object(pd, 'read_csv', guarded):
            data = HydroCaseData(ROOT)
            for case_id in p['case_ids']:
                if case_id in p['reuse_case_ids']:
                    for suffix in ['integer.npz', 'proof.json']:
                        name = f'case_{case_id:04d}_{suffix}'
                        shutil.copy2(BASE/name, out/name)
                        assert digest(out/name) == p['base_files'][name]
                    record = dict(old_records[case_id], continuation_action='reused_exact_bytes')
                else:
                    tick = time.monotonic()
                    input_path = cache / f'case_{case_id:04d}_inputs.npz'
                    control_path = cache / f'case_{case_id:04d}_result.npz'
                    for path in [input_path, control_path]:
                        assert digest(path) == cached['artifacts'][path.name]
                    case, control = arrays(input_path), arrays(control_path)
                    solver = IntegerWaterBudgetLP(data.network)
                    record = {'case_id': case_id, 'block': old_records[case_id]['block'],
                              'input_sha256': digest(input_path), 'control_sha256': digest(control_path),
                              'continuation_action': 'rerun_original_timeout_only'}
                    try:
                        result, proof = solver.solve_integer(case, [UNIT_ORDER.index(u) for u in TARGETS],
                                                             p['milp_seconds'], p['conditional_lp_seconds_each'])
                        assert result['objective'] <= control['objective'] + 1e-4
                        np.savez_compressed(out/f'case_{case_id:04d}_integer.npz', **result)
                        record.update(status='complete', proof=proof, relaxed_objective=float(control['objective']),
                                      integer_objective=float(result['objective']),
                                      objective_loss_eur=float(control['objective']-result['objective']),
                                      all_slacks_zero=all(np.max(result[k]) == 0 for k in ['minimum_slack','maximum_slack','bypass_slack','vest_slack']))
                    except Exception:
                        record.update(status='failed', error=traceback.format_exc(), proof=solver.last_integer_metadata)
                        np.savez_compressed(out/f'case_{case_id:04d}_failed_incumbent.npz', **solver.last_integer_arrays)
                    record['runtime_seconds'] = time.monotonic()-tick
                    (out/f'case_{case_id:04d}_proof.json').write_text(json.dumps(record, indent=2))
                report['cases'].append(record)
                print(json.dumps({'case_id': case_id, 'action': record['continuation_action'],
                                  'status': record['status'], 'proven_optimal': record['proof'].get('proven_optimal'),
                                  'mip_gap': record['proof'].get('mip_gap'), 'seconds': record['runtime_seconds']}), flush=True)
        (out/'physical_results_before_labels.json').write_text(json.dumps(report, indent=2))
        proof_gate = len(report['cases']) == 12 and all(r['status'] == 'complete' and r['proof']['proven_optimal'] and r['proof']['fixed_lp_replay_bitwise'] and r['all_slacks_zero'] for r in report['cases'])
        report['screen_passed'] = False
        if all(r['status'] == 'complete' for r in report['cases']):
            with np.load(BASE/'diagnostic.npz', allow_pickle=False) as z:
                truth, cq, cp, ids, units = [z[k].copy() for k in ['labels','control_qratio','control_pratio','case_ids','units']]
            assert ids.tolist() == p['case_ids'] and units.tolist() == UNIT_ORDER
            nq, npower = [], []
            for case_id in p['case_ids']:
                result, case = arrays(out/f'case_{case_id:04d}_integer.npz'), arrays(cache/f'case_{case_id:04d}_inputs.npz')
                nq.append(result['flow']/case['qmin'][:,None])
                npower.append(result['power']/case['pmin'][:,None])
            nq, npower = np.asarray(nq), np.asarray(npower)
            bm, nm = metrics(truth,cq,cp,p['blocks']), metrics(truth,nq,npower,p['blocks'])
            be, ne = (cp >= .5) != truth, (npower >= .5) != truth
            weak = [UNIT_ORDER.index(u) for u in TARGETS]
            weak_old, weak_new = int(be[:,weak].sum()), int(ne[:,weak].sum())
            block_gates = {block: int(ne[np.asarray(p['blocks']) == block][:,weak].sum()) <= int(be[np.asarray(p['blocks']) == block][:,weak].sum()) for block in sorted(set(p['blocks']))}
            unit_gates = {UNIT_ORDER[j]: int(ne[:,j].sum()) <= 1.05*int(be[:,j].sum()) for j in weak}
            gates = {'all_12_proven_and_feasible': proof_gate, 'weak3_error_reduction_at_least_10percent': weak_new <= .9*weak_old,
                     'no_block_weak3_deterioration': all(block_gates.values()), 'no_target_error_increase_above_5percent': all(unit_gates.values()),
                     'all14_error_nonincrease': int(ne.sum()) <= int(be.sum()), 'all14_qproxy_auc_nonincrease_in_error': nm['micro_auc_qratio'] >= bm['micro_auc_qratio']}
            report.update(labels_loaded_from_original_diagnostic_utc=datetime.now(timezone.utc).isoformat(),
                          metrics={'control':bm,'integer':nm}, weak3_errors={'control':weak_old,'integer':weak_new},
                          gates=gates, block_gates=block_gates, unit_gates=unit_gates, screen_passed=all(gates.values()))
            np.savez_compressed(out/'diagnostic.npz', labels=truth, control_qratio=cq, control_pratio=cp,
                                integer_qratio=nq, integer_pratio=npower, case_ids=ids, units=units)
        report['status'] = 'complete'
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report['runtime_seconds'] = time.monotonic()-started
        report['peak_rss_bytes'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform == 'darwin' else 1024)
        report['artifacts'] = {p.relative_to(out).as_posix(): digest(p) for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json'}
        (out/'manifest.json').write_text(json.dumps(report, indent=2))
        (out/'experiment_record.md').write_text(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus {report["status"]}. Separate solver-completion continuation of the failed original 12-case screen. Four original timeout cases receive 60-second MILP limits; eight proven result/proof files are reused byte-for-byte. Identical formulation, inputs, targets and gates. Command `{report["command"]}`. Screen passed: {report.get("screen_passed")}. Runtime {report["runtime_seconds"]:.3f} s. Every hash, status, bound, runtime and metric is recorded in `artifacts/{args.run_id}/manifest.json`. No model fit, new validation year, promotion, full cache or submission.\n')


if __name__ == '__main__':
    main()
