"""Prepare fresh 2022 GPU execution only from independently verified 2021 locks."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import pprint
import sys

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'artifacts/cuda-two-gpu-2021-notebook-002/run_two_gpu_2021.py'
TEMPLATE_SHA = 'ff52ce56947e9fccc2e11104333c84ac1f8a2018830062c61cdde33827499e5f'


def sha(path: Path) -> str:
    """Hash a small pinned source or manifest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source: str, old: str, new: str) -> str:
    """Fail closed if the reviewed template no longer matches an exact edit."""
    if source.count(old) != 1:
        raise ValueError(f'Template replacement is not unique: {old[:80]}')
    return source.replace(old, new, 1)


def main() -> None:
    """Create a clean notebook; this preparation never starts any remote job."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--eligible-lock', nargs=5, action='append', required=True,
                        metavar=('EXPERIMENT', 'LOCK_PATH', 'LOCK_SHA', 'AUDIT_PATH', 'AUDIT_SHA'))
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in ['', '.', '..']:
        raise ValueError('Invalid output run ID')
    if sha(TEMPLATE) != TEMPLATE_SHA:
        raise ValueError('Reviewed cgroup-aware2021 template changed')
    source = TEMPLATE.read_text()
    tree = ast.parse(source)
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign) and
                      any(isinstance(t, ast.Name) and t.id == 'JOBS' for t in node.targets))
    original_jobs = ast.literal_eval(assignment.value)
    jobs, input_records = [], []
    supplied = {item[0]: item for item in args.eligible_lock}
    if len(supplied) != len(args.eligible_lock) or not set(supplied) <= {'shop664', 'capacity608'}:
        raise ValueError('Each eligible experiment must appear once')
    for job in original_jobs:
        if job['name'] not in supplied:
            continue
        name, lock_name, lock_sha, audit_name, audit_sha = supplied[job['name']]
        lock_path, audit_path = Path(lock_name), Path(audit_name)
        if sha(lock_path) != lock_sha or sha(audit_path) != audit_sha:
            raise ValueError('Explicit2021 lock or independent audit hash differs')
        lock_bytes, audit_bytes = lock_path.read_bytes(), audit_path.read_bytes()
        lock, audit = json.loads(lock_bytes), json.loads(audit_bytes)
        if (audit['status'] != 'passed' or audit['selection_lock_sha256'] != lock_sha or
                not audit['eligible_for_2022']):
            raise ValueError('Require an independent passing audit of this eligible lock')
        if (lock['training_protocol_sha256'] != job['protocol_sha'] or not lock['eligible_for_2022'] or
                lock['selection_year'] != 2021 or lock['selected_weight'] <= 0 or
                lock['selected_2021_micro_auc'] <= max(lock['anchor_2021_micro_auc'], lock['control_2021_micro_auc'])):
            raise ValueError('Frozen2021 selection does not authorize2022')
        job.update(smoke_id=job['smoke_id'].replace('-001', '-002'),
                   fit_id=job['fit_id'].replace('-2021-', '-2022-'),
                   lock_sha=lock_sha, lock_text=lock_bytes.decode('utf-8'),
                   audit_sha=audit_sha, audit_text=audit_bytes.decode('utf-8'))
        jobs.append(job)
        input_records.append({'experiment': name, 'lock_path': str(lock_path), 'lock_sha256': lock_sha,
                              'audit_path': str(audit_path), 'audit_sha256': audit_sha,
                              'selected_weight': lock['selected_weight']})
    if not jobs:
        raise ValueError('No eligible experiment to run')
    lines = source.splitlines(keepends=True)
    source = ''.join(lines[:assignment.lineno - 1]) + 'JOBS = ' + pprint.pformat(jobs, width=120, sort_dicts=False) + '\n' + ''.join(lines[assignment.end_lineno:])
    source = replace_once(source, '"""Two separately reviewed2021 experiments; sequential smokes, staggered GPU fits."""',
                          '"""Frozen2022 confirmation for independently accepted2021 choices; fresh smokes and guarded fits."""')
    source = replace_once(source, "result += ['--year', '2021', '--smoke-manifest', str(job['smoke_path'])]",
                          "result += ['--year', '2022', '--smoke-manifest', str(job['smoke_path']),\n"
                          "                   '--selection-lock', str(job['lock_path']), '--selection-lock-sha256', job['lock_sha']]")
    source = replace_once(source, "m['mode'] == 'fit' and m['year'] == 2021", "m['mode'] == 'fit' and m['year'] == 2022")
    source = replace_once(source, "assert m['completed_units'] == 14 and m['completed_rounds_per_unit'] == job['trees']",
                          "assert m['completed_units'] == 14 and m['completed_rounds_per_unit'] == job['trees']\n"
                          "        assert m['selection_lock_sha256'] == job['lock_sha']")
    source = replace_once(source, "job['smoke_path'] = ROOT / job['run_root'] / job['smoke_id'] / 'manifest.json'",
                          """job['smoke_path'] = ROOT / job['run_root'] / job['smoke_id'] / 'manifest.json'
    lock_bytes, audit_bytes = job['lock_text'].encode('utf-8'), job['audit_text'].encode('utf-8')
    assert hashlib.sha256(lock_bytes).hexdigest() == job['lock_sha']
    assert hashlib.sha256(audit_bytes).hexdigest() == job['audit_sha']
    lock, audit = json.loads(lock_bytes), json.loads(audit_bytes)
    assert audit['status'] == 'passed' and audit['selection_lock_sha256'] == job['lock_sha'] and audit['eligible_for_2022']
    assert lock['training_protocol_sha256'] == job['protocol_sha'] and lock['selection_year'] == 2021
    assert lock['eligible_for_2022'] and lock['selected_weight'] > 0
    assert lock['selected_2021_micro_auc'] > max(lock['anchor_2021_micro_auc'], lock['control_2021_micro_auc'])
    lock_root = ROOT / 'independently-verified-2021-locks'
    lock_root.mkdir(exist_ok=True)
    job['lock_path'] = lock_root / f'{job["name"]}-selection_2021.json'
    for target, payload, expected in [(job['lock_path'], lock_bytes, job['lock_sha']),
                                      (lock_root / f'{job["name"]}-audit.json', audit_bytes, job['audit_sha'])]:
        if target.exists():
            assert target.is_file() and not target.is_symlink() and sha(target) == expected
        else:
            with target.open('xb') as stream:
                stream.write(payload)
        assert sha(target) == expected""")
    source = replace_once(source, "print('BOTH_PROTOCOLS_AND_RUNTIME_VERIFIED'", "print('ELIGIBLE_PROTOCOLS_LOCKS_AND_RUNTIME_VERIFIED'")
    source = replace_once(source, "print('TWO_GPU2021_ALL_FINISHED'", "print('FROZEN2022_ALL_FINISHED'")
    source = replace_once(source, "assert len(results) == 2, 'Each experiment must have its own archived result'",
                          "assert len(results) == len(JOBS), 'Each eligible experiment must have its own archived result'")
    ast.parse(source)
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    source_path = out / 'run_frozen_2022.py'
    source_path.write_text(source)
    notebook = {'cells': [
        {'cell_type': 'markdown', 'metadata': {}, 'source': [
            '# Frozen2022 confirmation\n',
            'Only independently accepted2021 experiments are included. Fresh sequential CUDA smokes, then guarded GPU fits. Original protocols and selected weights stay fixed. Each archive is retained independently. No final fit or submission.\n']},
        {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': source.splitlines(keepends=True)}],
        'metadata': {'kernelspec': {'display_name': 'Python3', 'language': 'python', 'name': 'python3'},
                     'language_info': {'name': 'python', 'version': '3.13.15'}},
        'nbformat': 4, 'nbformat_minor': 5}
    notebook_path = out / 'tdt4173-frozen-2022.ipynb'
    notebook_path.write_text(json.dumps(notebook, indent=2))
    report = {'run_id': args.run_id, 'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'status': 'prepared',
              'training': False, 'remote_action': False, 'template_path': str(TEMPLATE.relative_to(ROOT)),
              'template_sha256': TEMPLATE_SHA, 'builder_sha256': sha(Path(__file__)),
              'inputs': input_records, 'experiments': [job['name'] for job in jobs],
              'source_sha256': sha(source_path), 'notebook_sha256': sha(notebook_path),
              'environment': {'python': sys.version, 'platform': platform.platform()},
              'scope': 'Fresh2022 smoke/folds only, no selection change, final fit or submission',
              'command': sys.argv}
    (out / 'manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
