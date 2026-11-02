"""Review the frozen aggregate builder/loader without accepting a partial cache."""
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
from scripts import build_integer_water_budget_aggregate as builder


def main() -> None:
    """Verify exact preparation, typed receipt examples and immutable source closure."""
    out = ROOT / 'artifacts/integer-water-budget-aggregate-source-independent-audit-001'
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    protocol_sha = 'aacf3ee1492fbb48f1d62827cfc7306bad20841c48088cdb1b68842b94fe0c48'
    preflight_sha = '0563e518a505935f26f18dcfa9d49955a375bc1620ae064f2c0e4a5fa809d902'
    report = {'status': 'initializing', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'protocol_sha256': protocol_sha, 'optimization_calls': 0, 'labels_decoded': False,
              'aggregate_accepted': False, 'command': [sys.executable, *sys.argv]}
    try:
        protocol = json.loads(builder.PROTOCOL.read_text())
        builder.verify_protocol(protocol, protocol_sha)
        assert protocol['version'] == 'integer_water_budget_aggregate_v1'
        assert protocol['expected_counts'] == {'ordinary': 3010, 'imported': 636, 'exception': 1}
        assert protocol['exception_case_ids'] == [2012] and protocol['preflight_case_ids'] == [1, 1000, 2012]
        old = json.loads((ROOT / protocol['original_protocol']).read_text())
        for name, expected in old['source'].items():
            assert protocol['source'][name] == expected
        assert len(set(protocol['case_ids'])) == len(protocol['case_ids']) == 3647
        imported = json.loads((ROOT / protocol['imported_manifest']).read_text())
        assert protocol['imported_case_ids'] == sorted(map(int, imported['completed_cases']))
        preflight = ROOT / 'artifacts/integer-water-budget-aggregate-preflight-001'
        assert builder.digest(preflight / 'manifest.json') == preflight_sha
        pre = json.loads((preflight / 'manifest.json').read_text())
        assert pre['status'] == 'passed_preflight_only' and not pre['aggregate_eligible']
        assert pre['optimization_calls'] == 0 and not pre['labels_decoded'] and not pre['training']
        assert pre['protocol_sha256'] == protocol_sha and pre['preflight_case_ids'] == [1, 1000, 2012]
        assert set(pre['completed_cases']) == {'1', '1000', '2012'}
        builder.verify_hashes(preflight, pre['artifacts'])
        for cid, kind in [(1, 'imported'), (1000, 'ordinary'), (2012, 'exception')]:
            folder = preflight / 'cases' / f'case_{cid:04d}'
            receipt_path = folder / 'aggregate_receipt.json'
            assert builder.digest(receipt_path) == pre['completed_cases'][str(cid)]
            receipt = json.loads(receipt_path.read_text())
            assert receipt['status'] == 'accepted' and receipt['kind'] == kind and receipt['case_id'] == cid
            assert receipt['aggregate_protocol_sha256'] == protocol_sha
            assert receipt['original_cache_manifest_sha256'] is None
            builder.verify_hashes(folder, receipt['artifacts'])
            original = ROOT / protocol['original_cache'] / 'cases' / f'case_{cid:04d}'
            assert builder.inventory(original) == builder.inventory(folder / 'original_case') == receipt['original_case_inventory']
            assert builder.digest(folder / receipt['input_path']) == receipt['input_sha256']
            if cid == 2012:
                assert receipt['original_complete_path'] is None and not (folder / 'original_case/complete.json').exists()
                assert receipt['exception']['acceptance_basis'] == 'case2012_dual_feasibility_certificate'
                assert receipt['exception']['original_status_preserved'] == 'failed'
                assert receipt['exception']['completion_manifest_sha256'] == protocol['completion_manifest_sha256']
                assert receipt['exception']['actual_audit_sha256'] == protocol['exception_actual_audit_sha256']
                for replica in [1, 2]:
                    proof = json.loads((folder / 'original_case' / f'replica_{replica}_60s/proof.json').read_text())
                    assert proof['status'] == 'failed'
            else:
                assert receipt['exception'] is None
                assert builder.digest(folder / receipt['original_complete_path']) == receipt['original_complete_sha256']
        # A preparation with three cases must remain unusable by the compact loader.
        try:
            builder.validate_aggregate_manifest(pre, protocol)
        except ValueError:
            pass
        else:
            raise AssertionError('Partial preparation was accepted as a full aggregate')
        command = [sys.executable, '-m', 'unittest', 'tests.test_integer_water_budget_aggregate']
        test = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        (out / 'tests.txt').write_text(test.stdout + test.stderr)
        assert test.returncode == 0
        for name in builder.NEW:
            destination = out / 'source' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
        report.update(status='passed', source_sha256=builder.digest(ROOT / 'scripts/build_integer_water_budget_aggregate.py'),
            loader_sha256=builder.digest(ROOT / 'sintef_project/integer_water_budget_aggregate_features.py'),
            preflight_manifest_sha256=preflight_sha, focused_tests_passed=5, tests_command=command,
            exact_preflight_case_kinds={'1': 'imported', '1000': 'ordinary', '2012': 'exception'},
            source_review=['Old cache must be finalized with no queued work, exactly3646 complete cases and only the two declared2012 failures.',
                'Externally pinned scoped coordinator/all-worker exit observation precedes full assembly; all original payload inventory is preserved.',
                'Each original_case directory and original receipt remains byte exact; distinct aggregate receipts identify ordinary/imported/exception lineage.',
                'The sole2012 exception uses separately reviewed fresh completion and exactcertificate; original/fresh failed proof statuses stayfailed.',
                'Only a full3647 typed population can publish compact84 features; strict separate loader requires externalactualphysicalaudit and exactcase/date/hour/prefix bytes.',
                'Raw labels are not decoded, no optimization occurs, and no existing integer source or receipt is modified.'],
            limits='Source/preparation approval only. Full assembly still requires finalized original manifest, actual exit evidence, and later independent full aggregate audit before any training.')
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        shutil.copy2(__file__, out / Path(__file__).name)
        report.update(runtime_seconds=time.monotonic()-started, reviewer_source_sha256=builder.digest(Path(__file__)),
                      environment=builder.environment())
        (out / 'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
        with (ROOT / 'docs/experiments.md').open('a') as stream:
            stream.write(f'\n\n## integer-water-budget-aggregate-source-independent-audit-001 — {report["timestamp_utc"]}\n\n'
                f'Status {report["status"]}; root review of frozen separate aggregate builder/loader and three actualpreflight receipts. '
                f'Command `{sys.executable} scripts/audit_integer_aggregate_preparation.py`. Five eligibility/preservation tests, '
                f'zero solves/target decoding/fullaggregate approval. Runtime {report["runtime_seconds"]:.3f}s. Full evidence/source snapshots in manifest directory.\n')
        print(json.dumps({'status': report['status'], 'manifest_sha256': builder.digest(out / 'manifest.json')}), flush=True)


if __name__ == '__main__':
    main()
