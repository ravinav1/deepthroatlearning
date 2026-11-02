"""Independently check v2 continuation evidence without solving or signaling."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path: Path) -> str:
    """Hash exact saved bytes without decoding target labels."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checked_json(path: Path, expected: str) -> dict:
    """Require an externally supplied hash before decoding a manifest."""
    assert sha(path) == expected, path
    return json.loads(path.read_text())


def closure(root: Path, mapping: dict) -> None:
    """Check every recorded artifact, including the preserved source snapshot."""
    for name, expected in mapping.items():
        assert sha(root / name) == expected, root / name


def main() -> None:
    """Approve source and existing no-solve evidence, leaving live gates pending."""
    out = ROOT / 'artifacts/integer-water-budget-four-worker-v2-independent-audit-001'
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    protocol_sha = '7571930990cadc542c7ad4f47e7f1195d7a239885c887ad45b051afd9b9377d9'
    base_sha = 'f8aaa986c7afdaa802101f84c9cc33ce89309339efcfd321930431b035e8df9a'
    preflight_sha = '5ab737c0239d821c6b86128593431d532e834a604ba55a40e922265f09ff1e5a'
    harness_sha = '38b518cd5600c777012fcab39a6dd6e0c105231bc9ba5aa819543daf04593635'
    report = {'status': 'initializing', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'protocol_sha256': protocol_sha, 'base_protocol_sha256': base_sha,
              'preflight_manifest_sha256': preflight_sha, 'harness_manifest_sha256': harness_sha,
              'command': [sys.executable, *sys.argv],
              'environment': {'python': sys.version, 'platform': platform.platform()},
              'training': False, 'labels_decoded': False, 'solves': 0,
              'live_cache_interrupted': False, 'headroom_probe_run': False}
    try:
        p = checked_json(ROOT / 'docs/integer_water_budget_four_worker_protocol_v2.json', protocol_sha)
        base = checked_json(ROOT / 'docs/integer_water_budget_full_cache_protocol.json', base_sha)
        # Independently enumerate precisely the changed fields, rather than
        # accepting the continuation runner's own compatibility verdict alone.
        changed = {key for key in base if p[key] != base[key]}
        assert changed == {'version', 'created_utc', 'workers', 'source', 'resume_policy'}
        assert set(p) - set(base) == {'execution', 'compatible_base'}
        assert base['workers'] == 2 and p['workers'] == 4
        expected_extra = {'scripts/continue_integer_water_budget_cache_v2.py',
                          'scripts/check_integer_cache_transition_resources_v2.py',
                          'tests/test_integer_cache_continuation_v2.py',
                          'docs/integer_water_budget_four_worker_protocol_v2.md'}
        assert set(p['source']) - set(base['source']) == expected_extra
        assert all(p['source'][name] == value for name, value in base['source'].items())
        for group in ['source', 'data', 'audits']:
            closure(ROOT, p[group])
        assert p['execution']['headroom_probe_bytes'] == 3 * 1024**3
        assert p['execution']['headroom_minimum_free_percent'] == 30
        assert p['execution']['runtime_minimum_free_percent'] == 20
        assert p['execution']['resource_check_interval_seconds'] == 20
        assert p['execution']['maximum_resource_proof_age_seconds'] == 900
        from scripts import build_integer_water_budget_cache as original
        from scripts import continue_integer_water_budget_cache_v2 as candidate
        for name in ['run_attempt', 'finish_case', 'verify_complete_case', 'validate_attempt',
                     'eligible_attempt', 'retry_allowed', 'same_arrays', 'checked_copy']:
            assert getattr(candidate, name) is getattr(original, name), name
        candidate.check_compatibility(base, p)
        assert p['environment'] == candidate.environment()
        assert p['feature_specification'] == candidate.feature_specification()

        preflight_root = ROOT / 'artifacts/integer-water-budget-four-worker-v2-preflight-001'
        pf = checked_json(preflight_root / 'manifest.json', preflight_sha)
        assert pf['status'] == 'passed_preflight_only' and pf['protocol_sha256'] == protocol_sha
        assert not pf['labels_decoded'] and not pf['training'] and not pf['attempts']
        assert pf['source'] == p['source'] and pf['data'] == p['data']
        assert pf['case_ids'] == p['case_ids'] and pf['case_dates'] == p['case_dates']
        assert len(pf['case_ids']) == 3647 and len(pf['completed_cases']) == 24
        closure(preflight_root, pf['artifacts'])
        for case_id, receipt_sha in pf['completed_cases'].items():
            case_root = preflight_root / 'cases' / f'case_{int(case_id):04d}'
            assert sha(case_root / 'complete.json') == receipt_sha
            original.verify_complete_case(case_root, protocol_sha)

        harness_root = ROOT / 'artifacts/integer-water-budget-session-interruption-harness-v2-001'
        h = checked_json(harness_root / 'manifest.json', harness_sha)
        assert h['status'] == 'passed' and h['protocol_sha256'] == protocol_sha
        assert h['source_sha256'] == p['source']['scripts/check_integer_cache_transition_resources_v2.py']
        for name in ['interrupted', 'coordinator_finalized', 'all_children_exited', 'complete_case_payloads_unchanged']:
            assert h[name] is True, name
        assert h['live_cache_interrupted'] is False and h['partial_imported'] is False
        assert h['complete_case_count'] == 1 and h['partial_attempt_count'] == 2
        assert len(h['worker_exit_codes']) == 2 and all(x is not None for x in h['worker_exit_codes'])
        closure(harness_root, h['artifacts'])
        old = checked_json(harness_root / 'finalized_manifest.json', h['old_manifest_sha256'])
        assert old['status'] == 'failed' and old['protocol_sha256'] == base_sha
        assert len(old['completed_cases']) == 1
        for case_id, receipt_sha in old['completed_cases'].items():
            name = f'case_{int(case_id):04d}'
            source = harness_root / 'cases' / name
            target = harness_root / 'resume_simulation' / name
            receipt = checked_json(source / 'complete.json', receipt_sha)
            original.verify_complete_case(source, base_sha)
            imported = original.verify_complete_case(target, protocol_sha)
            assert (target / 'compatible_original_complete.json').read_bytes() == (source / 'complete.json').read_bytes()
            closure(target, receipt['artifacts'])
            assert imported['compatibility']['original_manifest_sha256'] == h['old_manifest_sha256']
            assert imported['compatibility']['original_complete_sha256'] == receipt_sha
        partials = list((harness_root / 'cases').glob('partial_case_*'))
        assert len(partials) == 2 and all(not (x / 'complete.json').exists() for x in partials)
        assert len(list((harness_root / 'resume_simulation').iterdir())) == 1
        tests = subprocess.run([sys.executable, '-m', 'unittest', 'tests.test_integer_cache_continuation_v2', '-v'],
                               cwd=ROOT, text=True, capture_output=True)
        (out / 'tests.txt').write_text(tests.stdout + tests.stderr)
        assert tests.returncode == 0
        report.update(status='passed', changed_base_fields=sorted(changed), preserved_original_source_files=len(base['source']),
                      preflight_artifact_count=len(pf['artifacts']), harness_artifact_count=len(h['artifacts']),
                      verified_complete_pilot_cases=24, independent_tests_passed=2,
                      interruption_harness_passed=True, physical_recipe_unchanged=True,
                      required_before_live_transition=['Fresh exact-protocol 3 GiB headroom proof passing the frozen limits.',
                        'Supported session interruption only, followed by observed final coordinator exit and child exit.',
                        'Externally hash finalized original manifest; verify every completed receipt before compatible reuse.'],
                      limitation='Saved native-session harness evidence checked; this audit did not signal any live process or run the headroom allocation.')
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report['runtime_seconds'] = time.monotonic() - started
        report['audit_source_sha256'] = sha(Path(__file__))
        report['artifacts'] = {x.name: sha(x) for x in out.iterdir() if x.is_file() and x.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'status': report['status'], 'manifest_sha256': sha(out / 'manifest.json')}))


if __name__ == '__main__':
    main()
