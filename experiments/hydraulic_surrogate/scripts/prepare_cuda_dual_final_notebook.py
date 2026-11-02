"""Combine independently audited final code without changing child bytes; prepare guarded notebook."""
from __future__ import annotations
import ast
from datetime import datetime,timezone
import hashlib,json,pprint,shutil,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_cuda_dual_2022_notebook import replace_once

def sha(path:Path)->str:
    """Hash a pinned artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main()->None:
    """Prepare code and notebook only; do not train, upload or submit."""
    out=ROOT/'artifacts/cuda-dual-final-package-001';out.mkdir(exist_ok=False)
    notebook=ROOT/'artifacts/cuda-two-gpu-final-notebook-001';notebook.mkdir(exist_ok=False)
    children=[
      {'name':'shop664','prefix':'cuda-xgb664','package_run':'cuda-xgb664-final-package-001','protocol_sha':'e571c6607bd797f65c76db5d7617dec7b031774a41af44e65283ccf1ad964ab7','package_sha':'90bad8da07acf42d710c87fa4ceccd0f0006354f6e901aff467f137ec02fcd16','audit':'cuda-xgb664-final-independent-audit-001','audit_sha':'bafb124da77c9bd7436985387dacba0e414faa6c3e5457f4477287856440fffc','device':'cuda:1','trees':600,'features':664,'loader_file':'cuda_xgb664_loader.py'},
      {'name':'capacity608','prefix':'cuda-xgb608-capacity','package_run':'cuda-xgb608-capacity-final-package-001','protocol_sha':'840879d339732b14349d5355a886ef2e1b8e31101aa482ec8472df56a7be20d3','package_sha':'9e8893ac7d46494a5b4001c68ef9090e365637436f2a456062479943cb489944','audit':'cuda-xgb608-capacity-final-independent-audit-001','audit_sha':'c509f81241eee3234ef053030903f1b15808ea090259c1b0cdb907758471d5a0','device':'cuda:0','trees':1200,'features':608,'loader_file':'cuda_xgb608_loader.py'},
    ]
    jobs=[];parent={'created_utc':datetime.now(timezone.utc).isoformat(),'visibility':'PRIVATE','training':False,'children':[],'files':[]}
    payload=out/'dataset';payload.mkdir()
    for c in children:
        source=ROOT/'artifacts'/c['package_run'];audit_path=ROOT/'artifacts'/c['audit']/'manifest.json'
        assert sha(audit_path)==c['audit_sha'] and json.loads(audit_path.read_text())['status']=='passed'
        m=json.loads((source/'manifest.json').read_text());assert m['status']=='complete' and m['protocol_sha256']==c['protocol_sha'] and m['package_manifest_sha256']==c['package_sha']
        for n,h in m['artifacts'].items():assert sha(source/n)==h
        code=source/'dataset/code';assert sha(code/'package_manifest.json')==c['package_sha']
        target=payload/c['name']/'code';shutil.copytree(code,target)
        for p in code.iterdir():assert (target/p.name).read_bytes()==p.read_bytes()
        stem=c['prefix'].replace('-','_');protocol_file=stem+'_final_protocol.json';protocol=json.loads((code/protocol_file).read_text());assert sha(code/protocol_file)==c['protocol_sha']
        assert protocol['accepted_validation']['accepted'] and protocol['accepted_validation']['selected_weight']==.25
        parent['children'].append({**c,'package_run_manifest_sha256':sha(source/'manifest.json'),'original_dataset_archive_sha256':m['dataset_archive_sha256']})
        jobs.append({'name':c['name'],'protocol_file':protocol_file,'protocol_sha':c['protocol_sha'],'package_sha':c['package_sha'],
          'validation_file':stem+'_protocol.json','validation_sha':protocol['validation_protocol_sha256'],
          'launcher_file':'launch_'+stem+'_final_dataset.py','launcher_sha':protocol['dataset_launcher_sha256'],
          'runner_file':'train_'+stem+'_final.py','validation_runner_file':'train_'+stem+'.py','loader_file':c['loader_file'],
          'smoke_root':c['prefix'][5:]+'-runs','run_root':c['prefix'][5:]+'-final-runs','smoke_id':c['prefix']+'-smoke-003','fit_id':c['prefix']+'-final-001',
          'device':c['device'],'trees':c['trees'],'features':c['features']})
    for p in sorted(payload.rglob('*')):
        if p.is_file():parent['files'].append({'path':p.relative_to(payload).as_posix(),'size_bytes':p.stat().st_size,'sha256':sha(p)})
    parent_path=payload/'dual_final_package_manifest.json';parent_path.write_text(json.dumps(parent,indent=2))
    archive=out/'cuda-dual-final-private-dataset.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED)as z:
        for p in sorted(payload.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(payload).as_posix())
    with zipfile.ZipFile(archive)as z:
        assert set(z.namelist())=={p.relative_to(payload).as_posix()for p in payload.rglob('*')if p.is_file()}
        for name in z.namelist():assert z.read(name)==(payload/name).read_bytes()
    template=ROOT/'artifacts/cuda-two-gpu-2021-notebook-002/run_two_gpu_2021.py';assert sha(template)=='ff52ce56947e9fccc2e11104333c84ac1f8a2018830062c61cdde33827499e5f'
    source=template.read_text();tree=ast.parse(source);assignment=next(n for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='JOBS'for t in n.targets));lines=source.splitlines(keepends=True)
    source=''.join(lines[:assignment.lineno-1])+'JOBS = '+pprint.pformat(jobs,width=120,sort_dicts=False)+'\n'+''.join(lines[assignment.end_lineno:])
    source=replace_once(source,'"""Two separately reviewed2021 experiments; sequential smokes, staggered GPU fits."""','"""Two independently accepted final recipes; genuine sequential smokes and guarded full fits."""')
    source=replace_once(source,"'--mode', mode, '--run-id', job['smoke_id'] if mode == 'smoke' else job['fit_id']]","'--package-sha256', job['package_sha'], '--mode', mode, '--run-id', job['smoke_id'] if mode == 'smoke' else job['fit_id']]")
    source=replace_once(source,"if mode == 'fit':\n        result += ['--year', '2021', '--smoke-manifest', str(job['smoke_path'])]","if mode == 'final':\n        result += ['--smoke-manifest', str(job['smoke_path']), '--smoke-manifest-sha256', sha(job['smoke_path'])]")
    source=replace_once(source,"folder = run_root / name","folder = ROOT / (job['smoke_root'] if name == job['smoke_id'] else job['run_root']) / name")
    source=replace_once(source,"path = run_root / name / 'manifest.json'","path = ROOT / (job['smoke_root'] if name == job['smoke_id'] else job['run_root']) / name / 'manifest.json'")
    source=replace_once(source,"m['mode'] == 'fit' and m['year'] == 2021","m['mode'] == 'final'")
    source=replace_once(source,"assert len(m['features']) == job['features'] and m['cuda']['requested_device'] == job['device']","assert len(m['features']) == job['features'] and m['cuda']['requested_device'] == job['device']\n        assert m['checkpoint_replay_bitwise'] is True and m['sampled_training_rows'] == 196358\n        assert m['sample_sha256'] == job['protocol']['train_sample_sha256']\n        assert m['accepted_validation'] == job['protocol']['accepted_validation']\n        assert m['model_parameters_requested'] == job['protocol']['model_parameters']")
    start=source.index('for job in JOBS:\n    matches =');end=source.index("print('BOTH_PROTOCOLS_AND_RUNTIME_VERIFIED'",start)
    guard='''parent_matches = [p for p in Path('/kaggle/input').rglob('dual_final_package_manifest.json') if p.is_file() and not p.is_symlink() and sha(p) == PARENT_SHA]
assert len(parent_matches) == 1, 'Attach exactly one audited PRIVATE dual final package'
parent_path = parent_matches[0]
parent = json.loads(parent_path.read_text())
for entry in parent['files']:
    relative = Path(entry['path'])
    assert not relative.is_absolute() and '..' not in relative.parts
    member = parent_path.parent / relative
    assert member.is_file() and not member.is_symlink() and member.stat().st_size == entry['size_bytes'] and sha(member) == entry['sha256']
for job in JOBS:
    job['code'] = parent_path.parent / job['name'] / 'code'
    protocol_path = job['code'] / job['protocol_file']
    assert sha(protocol_path) == job['protocol_sha'] and sha(job['code'] / 'package_manifest.json') == job['package_sha']
    job['launcher'] = job['code'] / job['launcher_file']
    assert sha(job['launcher']) == job['launcher_sha']
    job['protocol'] = protocol = json.loads(protocol_path.read_text())
    assert sha(job['code'] / job['validation_file']) == job['validation_sha'] == protocol['validation_protocol_sha256']
    validation = json.loads((job['code'] / job['validation_file']).read_text())
    assert sha(job['code'] / job['runner_file']) == protocol['trainer_sha256']
    for name, key in [(job['validation_runner_file'], 'trainer_sha256'), (job['loader_file'], 'shared_loader_sha256'),
                      ('train_cuda_tcn.py', 'core_loader_sha256'), ('launch_cuda_xgb.py', 'launcher_sha256')]:
        assert sha(job['code'] / name) == validation[key]
    if job['name'] == 'shop664':
        assert sha(job['code'] / 'cuda_xgb608_loader.py') == validation['prior_loader_sha256']
    assert protocol['accepted_validation']['accepted'] and protocol['accepted_validation']['selected_weight'] == .25
    assert protocol['model_parameters'] == validation['model_parameters']
    for name in [job['smoke_id'], job['fit_id']]:
        output_root = job['smoke_root'] if name == job['smoke_id'] else job['run_root']
        assert not (ROOT / output_root / name).exists(), 'Run ID exists; inspect instead of rerunning'
        assert not (ROOT / f'{name}.log').exists(), 'Log exists; no overwrite'
    assert not (ROOT / f'{job["fit_id"]}-download.zip').exists(), 'Archive exists; no overwrite'
    job['smoke_path'] = ROOT / job['smoke_root'] / job['smoke_id'] / 'manifest.json'
'''
    source=source[:start]+"PARENT_SHA = "+repr(sha(parent_path))+"\n"+guard+source[end:]
    source=replace_once(source,"smoke['protocol_sha256'] == job['protocol_sha']","smoke['protocol_sha256'] == job['validation_sha']")
    source=source.replace("start(ready.pop(0), 'fit')","start(ready.pop(0), 'final')")
    source=replace_once(source,"'\\nMODEL '","'\\nFINAL '")
    source=replace_once(source,"'TWO_GPU2021_ALL_FINISHED'","'TWO_GPU_FINAL_ALL_FINISHED'")
    compile(source,'run_dual_final.py','exec')
    source_path=notebook/'run_dual_final.py';source_path.write_text(source)
    book={'cells':[{'cell_type':'markdown','metadata':{},'source':['# Accepted final GPU recipes\n','Both models passed their fixed 2021 and 2022 gates. Fresh sequential smokes, guarded full training, complete checkpoint replay and independent return archives. No local submission or weight search.\n']},{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':source.splitlines(keepends=True)}],'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    book_path=notebook/'tdt4173-accepted-dual-final.ipynb';book_path.write_text(json.dumps(book,indent=2))
    report={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'status':'prepared','training':False,'upload':False,'source_sha256':sha(Path(__file__)),'parent_manifest_sha256':sha(parent_path),'dataset_archive_sha256':sha(archive),'dataset_archive_bytes':archive.stat().st_size,'notebook_sha256':sha(book_path),'notebook_source_sha256':sha(source_path),'children':parent['children']}
    shutil.copy2(__file__,out/Path(__file__).name)
    (out/'manifest.json').write_text(json.dumps(report,indent=2));(notebook/'manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
