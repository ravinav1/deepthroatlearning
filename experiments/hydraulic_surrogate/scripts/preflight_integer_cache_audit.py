"""Exercise the full-cache auditor on the separately completed 24-case pilot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_integer_water_budget_cache import (
    ROOT, arrays, digest, same, require, physical_certificate, independent_features,
    verify_hashes, PROTOCOL_SHA, HydroCaseData, ShopRelaxationLP, UNIT_ORDER, np, pd,
)
from sintef_project.integer_water_budget_features import compute_features


def main() -> None:
    """Check existing finalized pilot evidence; never access the live cache."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    require(Path(args.run_id).name == args.run_id and args.run_id not in ['', '.', '..'], 'Invalid ID')
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    report = {'status': 'initializing', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': shlex.join([sys.executable, *sys.argv]), 'optimization_calls': 0,
              'labels_decoded': False, 'live_cache_read': False, 'partial_feature_cache_read': False,
              'full_cache_approval': False, 'cases': [], 'corruption_checks': []}
    try:
        protocol_path = ROOT / 'docs/integer_water_budget_four_worker_protocol_v2.json'
        require(digest(protocol_path) == PROTOCOL_SHA, 'Frozen protocol changed')
        p = json.loads(protocol_path.read_text())
        pilot = ROOT / p['pilot_directory']
        require(digest(pilot / 'manifest.json') == p['pilot_manifest_sha256'], 'Pilot changed')
        require(digest(pilot / 'protocol.json') == p['pilot_protocol_sha256'], 'Pilot protocol changed')
        pm = json.loads((pilot / 'manifest.json').read_text())
        require(pm['status'] == 'complete' and pm['all_attempts_proven']
                and pm['all_case_replays_bitwise'] and len(pm['case_ids']) == 24, 'Pilot incomplete')
        verify_hashes(pilot, pm['artifacts'])
        continuous = ROOT / p['continuous_cache']
        require(digest(continuous / 'manifest.json') == p['continuous_cache_manifest_sha256'], 'Control changed')
        cm = json.loads((continuous / 'manifest.json').read_text())
        target = (ROOT / 'sintef_project/data/kernel/Unit_commitment_decisions.csv').resolve()
        original_read = pd.read_csv

        def metadata(path, *a, **kw):
            if isinstance(path, (str, Path)) and Path(path).resolve() == target:
                require(not a and kw == {'usecols': ['Run No', 'starttime']}, 'Labels forbidden')
            return original_read(path, *a, **kw)

        with patch.object(pd, 'read_csv', metadata):
            data = HydroCaseData(ROOT)
            solver = ShopRelaxationLP(data.network)
            for case_id in pm['case_ids']:
                case = arrays(pilot / f'case_{case_id:04d}_inputs.npz')
                index = p['case_ids'].index(case_id)
                fresh = data.case(index)
                fresh['natural'] = np.asarray(fresh['natural'], dtype=np.float64)
                for key, field in [('pmin', 'p_min'), ('start_cost', 'startcost_const'), ('stop_cost', 'stopcost_const')]:
                    fresh[key] = np.asarray([data.model['generator'][u][field] for u in UNIT_ORDER], dtype=np.float64)
                same(case, fresh)
                cp = continuous / f'case_{case_id:04d}_result.npz'
                require(digest(cp) == cm['artifacts'][cp.name], 'Control payload changed')
                control = arrays(cp)
                results = []
                checks = []
                for replica in [1, 2]:
                    folder = pilot / f'case_{case_id:04d}' / f'attempt_{replica}'
                    record = json.loads((folder / 'proof.json').read_text())
                    result = arrays(folder / 'result.npz')
                    lp = arrays(folder / 'fixed_lp_1.npz')
                    same(lp, arrays(folder / 'fixed_lp_2.npz'))
                    checks.append(physical_certificate(solver, case, result, lp, record['proof'], control))
                    independent = independent_features(case, result, control, data.network)
                    produced = compute_features(case, result, control, data.network, record['proof'])
                    same({'features': independent}, {'features': produced})
                    results.append(result)
                    if case_id == pm['case_ids'][0] and replica == 1:
                        corruptions = []
                        changed_proof = dict(record['proof'], mip_status=1, proven_optimal=False)
                        corruptions.append(('unproven_native', result, lp, changed_proof))
                        changed_lp = {key: value.copy() for key, value in lp.items()}
                        changed_lp['lower_marginals'][0] += 1
                        corruptions.append(('altered_dual', result, changed_lp, record['proof']))
                        changed_result = {key: value.copy() for key, value in result.items()}
                        changed_result['integer_solution'][0] += 1
                        corruptions.append(('altered_incumbent', changed_result, lp, record['proof']))
                        for name, r, l, proof in corruptions:
                            try:
                                physical_certificate(solver, case, r, l, proof, control)
                            except (ValueError, AssertionError, RuntimeError):
                                report['corruption_checks'].append({'name': name, 'rejected': True})
                            else:
                                raise ValueError('Corrupted evidence accepted: ' + name)
                same(results[0], results[1])
                report['cases'].append({'case_id': case_id, 'checks': checks, 'independent_feature_bytes_equal': True})
        report.update(status='passed', pilot_manifest_sha256=p['pilot_manifest_sha256'],
                      protocol_sha256=PROTOCOL_SHA, all_24_cases_physical_and_replayed=True,
                      caveat='Auditor preflight only. Actual full-cache audit still requires finalized3647manifest and independent source review.')
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        sources = ['scripts/audit_integer_water_budget_cache.py', 'scripts/preflight_integer_cache_audit.py',
                   'tests/test_integer_cache_physical_audit.py']
        report['source'] = {name: digest(ROOT / name) for name in sources}
        for name in sources:
            dest = out / 'source' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
        report['runtime_seconds'] = time.monotonic() - start
        (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
        with (ROOT / 'docs/experiments.md').open('a') as stream:
            stream.write(f'\n\n## {args.run_id} — {report["timestamp_utc"]}\n\nInteger-cache auditor preflight {report["status"]}, completed24case pilot only; {len(report["cases"])} checked. No current-cache or partial-feature reads, optimizations, labels or fits. Exact command `{report["command"]}`, source and input pins, physical/KKT/replay/independent84feature reconstruction and corrupted-proof checks retained in `artifacts/{args.run_id}/manifest.json`. Runtime {report["runtime_seconds"]:.3f}s. Does not approve an incomplete cache.\n')
        print(json.dumps({'status': report['status'], 'sha256': digest(out / 'manifest.json'),
                          'cases': len(report['cases']), 'seconds': report['runtime_seconds']}))


if __name__ == '__main__':
    main()
