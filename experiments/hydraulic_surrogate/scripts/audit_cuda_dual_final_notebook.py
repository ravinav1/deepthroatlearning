"""Verify the immutable final notebook and both exact child code packages."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/cuda-two-gpu-final-notebook-independent-audit-001'
OUT.mkdir(exist_ok=False)
START = time.monotonic()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
          'command': f'{sys.executable} scripts/audit_cuda_dual_final_notebook.py',
          'environment': {'python': sys.version, 'platform': platform.platform()},
          'training': False, 'upload': False, 'kaggle_submission': None}
try:
    folder = ROOT / 'artifacts/cuda-two-gpu-final-notebook-001'
    source_path = folder / 'run_dual_final.py'
    book_path = folder / 'tdt4173-accepted-dual-final.ipynb'
    assert sha(source_path) == '67681a6464f00b16e091d0a88c39595abce89ed90b5ab54d4c54929fc8dfac59'
    assert sha(book_path) == '5ca58debfe2967ad8c92b8b6c8dba8b7fdc83272b65ba87477cdc4d7ec3bc37b'
    source = source_path.read_text(); tree = ast.parse(source)
    book = json.loads(book_path.read_text())
    cells = [cell for cell in book['cells'] if cell['cell_type'] == 'code']
    assert len(cells) == 1 and ''.join(cells[0]['source']) == source
    assert cells[0]['outputs'] == [] and cells[0]['execution_count'] is None
    assignments = {target.id: node.value for node in tree.body if isinstance(node, ast.Assign)
                   for target in node.targets if isinstance(target, ast.Name)}
    jobs = ast.literal_eval(assignments['JOBS'])
    parent_sha = ast.literal_eval(assignments['PARENT_SHA'])
    assert parent_sha == '585c6ae68a62e6f118c8f676f232b452a4821a8e8ab3d8f91bef6119acdfdbc2'
    package_dir = ROOT / 'artifacts/cuda-dual-final-package-001'
    dataset = package_dir / 'dataset'
    parent_path = dataset / 'dual_final_package_manifest.json'
    assert sha(parent_path) == parent_sha
    parent = json.loads(parent_path.read_text())
    assert parent['visibility'] == 'PRIVATE' and parent['training'] is False
    assert len(parent['children']) == len(jobs) == 2
    assert [job['device'] for job in jobs] == ['cuda:1', 'cuda:0']
    assert [job['trees'] for job in jobs] == [600, 1200]
    assert [job['features'] for job in jobs] == [664, 608]
    assert len({job['run_root'] for job in jobs}) == len({job['smoke_root'] for job in jobs}) == 2
    for entry in parent['files']:
        path = dataset / entry['path']
        assert not path.is_symlink() and path.stat().st_size == entry['size_bytes'] and sha(path) == entry['sha256']
    assert {path.relative_to(dataset).as_posix() for path in dataset.rglob('*') if path.is_file()} == {entry['path'] for entry in parent['files']} | {parent_path.name}
    for child, job in zip(parent['children'], jobs):
        assert child['name'] == job['name'] and child['protocol_sha'] == job['protocol_sha']
        original = ROOT / 'artifacts' / child['package_run'] / 'dataset/code'
        copied = dataset / child['name'] / 'code'
        assert {path.name for path in original.iterdir()} == {path.name for path in copied.iterdir()}
        for path in original.iterdir():
            assert path.read_bytes() == (copied / path.name).read_bytes()
        assert sha(copied / 'package_manifest.json') == child['package_sha'] == job['package_sha']
        assert sha(copied / job['protocol_file']) == job['protocol_sha']
        assert sha(copied / job['launcher_file']) == job['launcher_sha']
        audit_path = ROOT / 'artifacts' / child['audit'] / 'manifest.json'
        assert sha(audit_path) == child['audit_sha'] and json.loads(audit_path.read_text())['status'] == 'passed'
        protocol = json.loads((copied / job['protocol_file']).read_text())
        assert protocol['accepted_validation']['accepted'] and protocol['accepted_validation']['selected_weight'] == .25
        assert protocol['model_parameters']['device'] == job['device'] and protocol['model_parameters']['n_estimators'] == job['trees']
        assert job['smoke_id'].endswith('smoke-003') and job['fit_id'].endswith('final-001')
    archive_path = package_dir / 'cuda-dual-final-private-dataset.zip'
    assert sha(archive_path) == 'e3d6878960c983c96e76daa7c3150b3dcff2bba61a75ec923b3f0f08ea3eecd3'
    with zipfile.ZipFile(archive_path) as zipped:
        files = {path.relative_to(dataset).as_posix(): path for path in dataset.rglob('*') if path.is_file()}
        assert len(zipped.namelist()) == len(set(zipped.namelist())) and set(zipped.namelist()) == set(files)
        for name, path in files.items(): assert zipped.read(name) == path.read_bytes()
    template_path = ROOT / 'artifacts/cuda-two-gpu-2021-notebook-002/run_two_gpu_2021.py'
    assert sha(template_path) == 'ff52ce56947e9fccc2e11104333c84ac1f8a2018830062c61cdde33827499e5f'
    original_tree = ast.parse(template_path.read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    original_functions = {node.name: node for node in original_tree.body if isinstance(node, ast.FunctionDef)}
    for name in ['available_gib', 'start', 'heartbeat', 'sha', 'stamp']:
        assert ast.dump(functions[name]) == ast.dump(original_functions[name]), name
    runtime_start = source.index('# Exact previously verified runtime;')
    runtime_end = source.index('PARENT_SHA =', runtime_start)
    original_source = template_path.read_text()
    assert source[runtime_start:runtime_end].strip() == original_source[original_source.index('# Exact previously verified runtime;'):original_source.index('for job in JOBS:\n    matches =')].strip()
    assert "feature_phase_done = '\\nFINAL '" in source
    assert 'memory >= 12' in source and "first['started'] >= 20" in source
    assert "'--year'" not in source and "start(ready.pop(0), 'fit')" not in source
    assert "assert len(results) == 2" in source
    # Execute only isolated pure wrapper helpers against synthetic temporary files.
    namespace = {'Path': Path, 'hashlib': hashlib, 'json': json, 'zipfile': zipfile,
                 'sys': sys, 'datetime': datetime, 'timezone': timezone,
                 'FileLink': str, 'display': lambda _: None}
    helper_tree = ast.Module(body=[functions[name] for name in ['sha', 'stamp', 'command', 'archive']], type_ignores=[])
    exec(compile(helper_tree, '<isolated-wrapper-helpers>', 'exec'), namespace)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); namespace['ROOT'] = root
        for original_job in jobs:
            job = dict(original_job); job['launcher'] = Path('/reviewed') / job['launcher_file']
            for name, output_root in [(job['smoke_id'], job['smoke_root']), (job['fit_id'], job['run_root'])]:
                target = root / output_root / name; target.mkdir(parents=True)
                (target / 'manifest.json').write_text(json.dumps({'synthetic_review_fixture': name}))
            job['smoke_path'] = root / job['smoke_root'] / job['smoke_id'] / 'manifest.json'
            final_command = namespace['command'](job, 'final')
            assert final_command[final_command.index('--mode')+1] == 'final'
            assert final_command[final_command.index('--smoke-manifest')+1] == str(job['smoke_path'])
            assert final_command[final_command.index('--smoke-manifest-sha256')+1] == sha(job['smoke_path'])
            assert final_command[final_command.index('--package-sha256')+1] == job['package_sha']
            smoke_command = namespace['command'](job, 'smoke')
            assert '--smoke-manifest' not in smoke_command and smoke_command[-1] == job['smoke_id']
            record = namespace['archive'](job, False, 'synthetic reviewer-only failure')
            assert record['status'] == 'failed' and record['manifest_sha256'] and record['smoke_manifest_sha256']
            with zipfile.ZipFile(root / f'{job["fit_id"]}-download.zip') as zipped:
                assert set(zipped.namelist()) == {f'{job["fit_id"]}/manifest.json', f'{job["smoke_id"]}/manifest.json'}
    report.update(status='passed', notebook_sha256=sha(book_path), source_sha256=sha(source_path),
                  parent_manifest_sha256=parent_sha, dataset_archive_sha256=sha(archive_path),
                  exact_child_package_bytes=True, inherited_memory_runtime_guards_ast_identical=True,
                  independent_command_and_separate_archive_fixture_passed=True,
                  review='Sequential genuine smokes; original device and source/runtime pins; fresh IDs; no year/selection action; first FINAL/20second/effective12GiB gate; independent smoke+final archives and strict completion checks.')
except BaseException:
    report.update(status='failed', error=traceback.format_exc())
    raise
finally:
    shutil.copy2(__file__, OUT / Path(__file__).name)
    report['runtime_seconds'] = time.monotonic()-START
    report['artifacts'] = {path.name: sha(path) for path in OUT.iterdir() if path.is_file() and path.name != 'manifest.json'}
    (OUT / 'manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'status': report['status'], 'manifest_sha256': sha(OUT / 'manifest.json'),
                      'runtime_seconds': report['runtime_seconds']}), flush=True)
