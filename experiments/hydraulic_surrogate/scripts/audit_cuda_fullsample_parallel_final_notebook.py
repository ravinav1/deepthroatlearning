"""Independently verify combined child bytes and the one-job fullsample final wrapper."""
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
OUT = ROOT / 'artifacts/cuda-fullsample-parallel-final-notebook-independent-audit-001'
OUT.mkdir(exist_ok=False)
START = time.monotonic()
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
          'command': f'{sys.executable} scripts/audit_cuda_fullsample_parallel_final_notebook.py',
          'environment': {'python': sys.version, 'platform': platform.platform()},
          'scope': 'Both exact child datasets and fullsample608 final notebook only; SHOP2021 wrapper separately audited by shop.',
          'training': False, 'upload': False, 'kaggle_submission': None}
try:
    folder = ROOT / 'artifacts/cuda-fullsample-parallel-notebooks-001'
    source_path = folder / 'tdt4173-fullsample608-final.py'
    book_path = folder / 'tdt4173-fullsample608-final.ipynb'
    assert sha(source_path) == 'ca1f485399dc51eb06895e64ef6c3c31be811c1a28690e29afd4ed7e5eef4cdb'
    assert sha(book_path) == '97058cea23ac76d5e9e8c6757881fdf8ad39eb01f05458c6c5563521f265be96'
    source = source_path.read_text(); tree = ast.parse(source)
    cells = [c for c in json.loads(book_path.read_text())['cells'] if c['cell_type'] == 'code']
    assert len(cells) == 1 and ''.join(cells[0]['source']) == source
    assert not cells[0]['outputs'] and cells[0]['execution_count'] is None
    assignments = {t.id: n.value for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    jobs = ast.literal_eval(assignments['JOBS'])
    assert len(jobs) == 1
    job = jobs[0]
    assert job['name'] == 'fullsample608final' and job['device'] == 'cuda:1'
    assert job['trees'] == 600 and job['features'] == 608
    assert job['smoke_id'] == 'cuda-xgb608-fullsample-smoke-003'
    assert job['fit_id'] == 'cuda-xgb608-fullsample-final-001'
    assert job['smoke_root'] != job['run_root']
    parent_sha = ast.literal_eval(assignments['PARENT_SHA'])
    assert parent_sha == 'ae1a7b7af811e919ab0af6233e6456536a8f2eb5f964ea8f9af715247bf89617'
    package_dir = ROOT / 'artifacts/cuda-fullsample-parallel-package-001'; dataset = package_dir / 'dataset'
    parent_path = dataset / 'fullsample_parallel_package_manifest.json'
    assert sha(parent_path) == parent_sha
    parent = json.loads(parent_path.read_text())
    assert parent['visibility'] == 'PRIVATE' and parent['training'] is False and len(parent['children']) == 2
    assert 'Two independent saved notebook VMs' in parent['execution']
    for entry in parent['files']:
        p = dataset / entry['path']
        assert not p.is_symlink() and p.stat().st_size == entry['size_bytes'] and sha(p) == entry['sha256']
    assert {p.relative_to(dataset).as_posix() for p in dataset.rglob('*') if p.is_file()} == {e['path'] for e in parent['files']} | {parent_path.name}
    for child in parent['children']:
        original = ROOT / 'artifacts' / child['package_run']
        assert sha(original / 'manifest.json') == child['package_run_manifest_sha256']
        m = json.loads((original / 'manifest.json').read_text())
        assert m['status'] == 'complete' and m['dataset_archive_sha256'] == child['original_dataset_archive_sha256']
        assert m['protocol_sha256'] == child['protocol_sha256']
        for name, expected in m['artifacts'].items(): assert sha(original / name) == expected
        proof = ROOT / child['independent_audit_path']
        assert sha(proof) == child['independent_audit_sha256']
        audit = json.loads(proof.read_text())
        assert audit['status'] == 'passed' and audit['protocol_sha256'] == child['protocol_sha256']
        original_files = {p.relative_to(original / 'dataset').as_posix(): p for p in (original / 'dataset').rglob('*') if p.is_file()}
        copied_files = {p.relative_to(dataset / child['name']).as_posix(): p for p in (dataset / child['name']).rglob('*') if p.is_file()}
        assert set(original_files) == set(copied_files)
        for name in original_files: assert original_files[name].read_bytes() == copied_files[name].read_bytes()
    code = dataset / job['name'] / 'code'
    assert sha(code / 'package_manifest.json') == job['package_sha']
    assert sha(code / job['protocol_file']) == job['protocol_sha']
    assert sha(code / job['launcher_file']) == job['launcher_sha']
    protocol = json.loads((code / job['protocol_file']).read_text())
    assert protocol['accepted_validation']['accepted'] and protocol['accepted_validation']['selected_weight'] == .25
    assert protocol['model_parameters']['device'] == 'cuda:1' and protocol['model_parameters']['n_estimators'] == 600
    assert sha(code / job['validation_file']) == job['validation_sha'] == protocol['validation_protocol_sha256']
    assert sha(code / job['runner_file']) == protocol['trainer_sha256']
    archive_path = package_dir / 'cuda-fullsample-parallel-private-dataset.zip'
    assert sha(archive_path) == '3931c540ea611b28c4913f817f545a07228761247bb6685dc2a8aacad65e66dc' and archive_path.stat().st_size == 142881
    with zipfile.ZipFile(archive_path) as z:
        files = {p.relative_to(dataset).as_posix(): p for p in dataset.rglob('*') if p.is_file()}
        assert len(z.namelist()) == len(set(z.namelist())) and set(z.namelist()) == set(files)
        for name, p in files.items(): assert z.read(name) == p.read_bytes()
    template_path = ROOT / 'artifacts/cuda-two-gpu-final-notebook-001/run_dual_final.py'
    assert sha(template_path) == '67681a6464f00b16e091d0a88c39595abce89ed90b5ab54d4c54929fc8dfac59'
    old_tree = ast.parse(template_path.read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    old_functions = {n.name: n for n in old_tree.body if isinstance(n, ast.FunctionDef)}
    assert set(functions) == set(old_functions)
    for name in functions:
        previous = ast.dump(old_functions[name])
        if name == 'finish': previous = previous.replace('value=196358', 'value=490896')
        assert ast.dump(functions[name]) == previous, name
    # All executable module statements are inherited except exact job/parent IDs,
    # one result count, and the visible completion marker. Comments are irrelevant.
    def normalize(tree, old):
        for n in tree.body:
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {'JOBS','PARENT_SHA'} for t in n.targets): n.value = ast.Constant(None)
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant):
                if isinstance(n.value,str): n.value = n.value.replace('dual_final_package_manifest.json', 'fullsample_parallel_package_manifest.json').replace('TWO_GPU_FINAL_ALL_FINISHED','FULLSAMPLE608_FINAL_ALL_FINISHED')
                if old and n.value == 196358: n.value = 490896
            if old and isinstance(n, ast.Assert) and isinstance(n.test, ast.Compare) and ast.unparse(n.test.left) == 'len(results)': n.test.comparators = [ast.Constant(1)]
        if isinstance(tree.body[0],ast.Expr) and isinstance(tree.body[0].value,ast.Constant): tree.body.pop(0)
        return ast.dump(tree)
    assert normalize(old_tree,True) == normalize(ast.parse(source),False)
    namespace = {'Path': Path, 'hashlib': hashlib, 'json': json, 'zipfile': zipfile,
                 'sys': sys, 'datetime': datetime, 'timezone': timezone, 'FileLink': str, 'display': lambda _: None}
    exec(compile(ast.Module(body=[functions[n] for n in ['sha','stamp','command','archive']],type_ignores=[]),'<pure-wrapper-helpers>','exec'),namespace)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);namespace['ROOT']=root;fixture=dict(job);fixture['launcher']=Path('/reviewed')/job['launcher_file']
        for name, output_root in [(job['smoke_id'],job['smoke_root']),(job['fit_id'],job['run_root'])]:
            d=root/output_root/name;d.mkdir(parents=True);(d/'manifest.json').write_text(json.dumps({'synthetic_review_fixture':name}))
        fixture['smoke_path']=root/job['smoke_root']/job['smoke_id']/'manifest.json'
        cmd=namespace['command'](fixture,'final')
        assert cmd[cmd.index('--mode')+1]=='final' and '--year' not in cmd
        assert cmd[cmd.index('--smoke-manifest-sha256')+1]==sha(fixture['smoke_path'])
        assert cmd[cmd.index('--package-sha256')+1]==job['package_sha']
        assert '--smoke-manifest' not in namespace['command'](fixture,'smoke')
        record=namespace['archive'](fixture,False,'synthetic reviewer-only failure')
        assert record['status']=='failed' and record['manifest_sha256'] and record['smoke_manifest_sha256']
        with zipfile.ZipFile(root/f'{job["fit_id"]}-download.zip')as z:assert set(z.namelist())=={f'{job["fit_id"]}/manifest.json',f'{job["smoke_id"]}/manifest.json'}
    report.update(status='passed',notebook_sha256=sha(book_path),source_sha256=sha(source_path),parent_manifest_sha256=parent_sha,dataset_archive_sha256=sha(archive_path),exact_child_dataset_bytes=True,inherited_executable_AST_exact_after_enumerated_changes=True,synthetic_command_and_failure_archive_passed=True,review='One GPU1 final job, all490896 rows, original600 trees; fresh genuine smoke003; fixed independently accepted .25; pinned runtime/source; foreground20second progress; strict replay/hash checks and both smoke/final failure archives. Separate VM from SHOP2021 required.')
except BaseException:
    report.update(status='failed',error=traceback.format_exc());raise
finally:
    shutil.copy2(__file__,OUT/Path(__file__).name)
    report['runtime_seconds']=time.monotonic()-START
    report['artifacts']={p.name:sha(p)for p in OUT.iterdir()if p.is_file()and p.name!='manifest.json'}
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'status':report['status'],'manifest_sha256':sha(OUT/'manifest.json'),'runtime_seconds':report['runtime_seconds']}),flush=True)
