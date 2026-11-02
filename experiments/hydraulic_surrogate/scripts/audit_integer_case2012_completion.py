"""Independent source/protocol review and no-solve rehearsal of the one-case exception."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import complete_integer_case2012 as candidate


def main() -> None:
    """Check frozen closure and exercise only read-only saved-evidence validation."""
    out = ROOT / 'artifacts/integer-case2012-completion-source-independent-audit-001'
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {'status': 'initializing', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'optimization_calls': 0, 'labels_decoded': False, 'eligibility_changed': False,
              'protocol_sha256': '6499a285d267e8ccab0cb2063e682880c8837423fdd8606d9c83d11063cff989',
              'source_sha256': '5a739e4e78546098fb46751f7f81d84012d5bd71390700cd23ebba081b25638d'}
    try:
        assert candidate.digest(ROOT / 'scripts/complete_integer_case2012.py') == report['source_sha256']
        p = candidate.check_protocol(ROOT / 'docs/integer_case2012_completion_protocol.json', report['protocol_sha256'])
        assert p['allowed_case_ids'] == [2012] and p['replicas'] == p['workers'] == 2
        assert p['milp_seconds'] == 60 and p['conditional_lp_seconds_each'] == p['solver_threads'] == 1
        assert not p['retry'] and not p['aggregate_eligible'] and not p['old_v2_eligible']
        assert p['acceptance_basis'] == 'case2012_dual_feasibility_certificate'
        assert p['absolute_eur_bound'] == {'numerator': '1', 'denominator': '100000'}
        proof = ROOT / 'artifacts/integer-case2012-completion-preflight-001/manifest.json'
        assert candidate.digest(proof) == 'c231d938940e8d29cef2db0717fa7a7c65ffede725a16cd0450a9c41b0c424cf'
        pre = json.loads(proof.read_text())
        assert pre['status'] == 'passed_preparation_only' and pre['optimization_calls'] == 0
        assert pre['protocol_sha256'] == report['protocol_sha256'] and pre['source_sha256'] == report['source_sha256']
        test = subprocess.run([sys.executable, '-m', 'unittest', 'tests.test_integer_case2012_completion'],
                              cwd=ROOT, text=True, capture_output=True)
        (out / 'tests.txt').write_text(test.stdout + test.stderr)
        assert test.returncode == 0
        certificate = json.loads((ROOT / p['certificate']).read_text())
        # Reconstruct inputs with the same metadata-only guard, then execute
        # only saved-payload decoding and certificate routines, never workers.
        target = (ROOT / 'sintef_project/data/kernel/Unit_commitment_decisions.csv').resolve()
        original = candidate.pd.read_csv

        def metadata(path, *args, **kwargs):
            if isinstance(path, (str, Path)) and Path(path).resolve() == target:
                assert not args and kwargs == {'usecols': ['Run No', 'starttime']}
            return original(path, *args, **kwargs)

        with candidate.patch.object(candidate.pd, 'read_csv', metadata):
            data = candidate.HydroCaseData(ROOT)
        case = candidate.arrays(ROOT / p['cache'] / 'case_2012_inputs.npz')
        control = candidate.arrays(ROOT / p['cache'] / 'case_2012_result.npz')
        checked = []
        for replica, old in enumerate(p['original_attempts'], 1):
            folder = ROOT / old['directory']
            record = json.loads((folder / 'proof.json').read_text())
            candidate.verify_fresh(folder, record, p)
            result, checks = candidate.verify_physics(data.network, case, control, folder, record,
                0, certificate['replicas'][replica - 1]['certificate'])
            assert checks['all_other_original_physical_checks_passed']
            checked.append({'replica': replica, 'checks': checks,
                            'decode_iteration_metadata': 'Synthetic zero for no-solve rehearsal only; actual execution must observe both LP iteration counts.'})
        candidate.check_protocol(ROOT / 'docs/integer_case2012_completion_protocol.json', report['protocol_sha256'])
        report.update(status='passed', focused_tests_passed=4, saved_evidence_rehearsal=checked,
            review=['Unchanged native MILP and two conditional LP calls per fresh process; no alternative retries.',
                    'Native status0, same seeds/bounds/nodes/objectives and every NPZ file plus array must match both old failed replicas.',
                    'Exact positive-scale/dimensional checks and rational bound must match independently reviewed original certificate.',
                    'Unchanged physical auditor reaches only the precise objective-guard exception after native/conditional/KKT/zero-slack checks; subsequent subset check explicitly applied.',
                    'Two fresh spawned workers exit through ProcessPoolExecutor context before acceptance.',
                    'Separate exception result/manifest; no ordinary complete receipt, feature cache or original status rewrite.'],
            caveat='Source review and stored-evidence rehearsal only. No fresh optimization executed or completion result accepted by this audit.')
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        shutil.copy2(__file__, out / Path(__file__).name)
        report.update(runtime_seconds=time.monotonic() - started,
                      reviewer_source_sha256=candidate.digest(Path(__file__)), environment=candidate.environment())
        (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
        with (ROOT / 'docs/experiments.md').open('a') as stream:
            stream.write(f'\n\n## integer-case2012-completion-source-independent-audit-001 — {report["timestamp_utc"]}\n\n'
                f'Status {report["status"]}. Source/protocol review and no-solve saved-evidence rehearsal, four rejection tests. '
                f'Command `{sys.executable} scripts/audit_integer_case2012_completion.py`. Runtime {report["runtime_seconds"]:.3f}s. '
                'No target decoding or eligibility mutation. Full source/data/protocol checks and reasoning in the audit manifest.\n')
        print(json.dumps({'status': report['status'], 'manifest_sha256': candidate.digest(out / 'manifest.json')}), flush=True)


if __name__ == '__main__':
    main()
