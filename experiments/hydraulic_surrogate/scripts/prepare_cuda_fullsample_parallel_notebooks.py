"""Preserve two audited child datasets and adapt reviewed wrappers for separate saved VMs."""
from __future__ import annotations
import argparse,ast
from datetime import datetime,timezone
import hashlib,json,pprint,shutil,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_cuda_dual_2022_notebook import replace_once

def sha(p:Path)->str:
    """Hash an immutable code artifact."""
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save_notebook(out:Path,name:str,source:str,description:str)->dict:
    """Store identical inspectable Python and one executable notebook cell."""
    compile(source,name,'exec');sp=out/(name+'.py');sp.write_text(source)
    book={'cells':[{'cell_type':'markdown','metadata':{},'source':[description+'\n']},{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':source.splitlines(keepends=True)}],'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5};bp=out/(name+'.ipynb');bp.write_text(json.dumps(book,indent=2));return {'source':str(sp.relative_to(ROOT)),'source_sha256':sha(sp),'notebook':str(bp.relative_to(ROOT)),'notebook_sha256':sha(bp)}

def main()->None:
    """Prepare only; both frozen GPU1 jobs require separate saved sessions."""
    ap=argparse.ArgumentParser();ap.add_argument('--final-audit',type=Path,required=True);ap.add_argument('--final-audit-sha256',required=True);ap.add_argument('--interaction-audit',type=Path,required=True);ap.add_argument('--interaction-audit-sha256',required=True);a=ap.parse_args()
    children=[('fullsample608final','cuda-xgb608-fullsample-final-package-001','4bf91a4ceb200a884802f40af82e3f1f186f50d647c4d44f7bd4d6f601b6714e',a.final_audit,a.final_audit_sha256),('shop664fullsample2021','cuda-xgb664-fullsample-package-001','2a5a114ae75d9a34d4a1f580ff7c3a781332fa92120605501b5b76bc72f31c5d',a.interaction_audit,a.interaction_audit_sha256)]
    out=ROOT/'artifacts/cuda-fullsample-parallel-package-001';out.mkdir(exist_ok=False);payload=out/'dataset';payload.mkdir();parent={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'visibility':'PRIVATE','training':False,'execution':'Two independent saved notebook VMs; both original CUDA1 devices remain unchanged.','children':[],'files':[]}
    for name,run,protocol_sha,audit_path,audit_sha in children:
        assert sha(audit_path)==audit_sha;proof=json.loads(audit_path.read_text());assert proof['status']=='passed' and proof['protocol_sha256']==protocol_sha
        d=ROOT/'artifacts'/run;m=json.loads((d/'manifest.json').read_text());assert m['status']=='complete' and m['protocol_sha256']==protocol_sha
        for n,h in m['artifacts'].items():assert sha(d/n)==h,n
        shutil.copytree(d/'dataset',payload/name)
        for p in (d/'dataset').rglob('*'):
            if p.is_file():assert (payload/name/p.relative_to(d/'dataset')).read_bytes()==p.read_bytes()
        parent['children'].append({'name':name,'package_run':run,'package_run_manifest_sha256':sha(d/'manifest.json'),'protocol_sha256':protocol_sha,'original_dataset_archive_sha256':m['dataset_archive_sha256'],'independent_audit_sha256':audit_sha,'independent_audit_path':str(audit_path)})
    for p in sorted(payload.rglob('*')):
        if p.is_file():parent['files'].append({'path':p.relative_to(payload).as_posix(),'sha256':sha(p),'size_bytes':p.stat().st_size})
    pp=payload/'fullsample_parallel_package_manifest.json';pp.write_text(json.dumps(parent,indent=2));archive=out/'cuda-fullsample-parallel-private-dataset.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED)as z:
        for p in sorted(payload.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(payload).as_posix())
    with zipfile.ZipFile(archive)as z:
        assert set(z.namelist())=={p.relative_to(payload).as_posix()for p in payload.rglob('*')if p.is_file()}
        for n in z.namelist():assert z.read(n)==(payload/n).read_bytes()
    notebooks=ROOT/'artifacts/cuda-fullsample-parallel-notebooks-001';notebooks.mkdir(exist_ok=False)
    template=ROOT/'artifacts/cuda-two-gpu-final-notebook-001/run_dual_final.py';assert sha(template)=='67681a6464f00b16e091d0a88c39595abce89ed90b5ab54d4c54929fc8dfac59';s=template.read_text();p=json.loads((payload/'fullsample608final/code/cuda_xgb608_fullsample_final_protocol.json').read_text());package_sha=sha(payload/'fullsample608final/code/package_manifest.json')
    jobs=[{'name':'fullsample608final','protocol_file':'cuda_xgb608_fullsample_final_protocol.json','protocol_sha':children[0][2],'package_sha':package_sha,'validation_file':'cuda_xgb608_fullsample_protocol.json','validation_sha':p['validation_protocol_sha256'],'launcher_file':'launch_cuda_xgb608_fullsample_final_dataset.py','launcher_sha':p['dataset_launcher_sha256'],'runner_file':'train_cuda_xgb608_fullsample_final.py','validation_runner_file':'train_cuda_xgb608_fullsample.py','loader_file':'cuda_xgb608_loader.py','smoke_root':'xgb608-fullsample-runs','run_root':'xgb608-fullsample-final-runs','smoke_id':'cuda-xgb608-fullsample-smoke-003','fit_id':'cuda-xgb608-fullsample-final-001','device':'cuda:1','trees':600,'features':608}]
    tree=ast.parse(s);node=next(n for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='JOBS'for t in n.targets));lines=s.splitlines(keepends=True);s=''.join(lines[:node.lineno-1])+'JOBS = '+pprint.pformat(jobs,width=120,sort_dicts=False)+'\n'+''.join(lines[node.end_lineno:]);s=s.replace('dual_final_package_manifest.json','fullsample_parallel_package_manifest.json').replace('585c6ae68a62e6f118c8f676f232b452a4821a8e8ab3d8f91bef6119acdfdbc2',sha(pp));s=replace_once(s,"m['sampled_training_rows'] == 196358","m['sampled_training_rows'] == 490896");s=replace_once(s,'assert len(results) == 2','assert len(results) == 1');s=s.replace('TWO_GPU_FINAL_ALL_FINISHED','FULLSAMPLE608_FINAL_ALL_FINISHED');s=s.replace('Two independently accepted final recipes; genuine sequential smokes and guarded full fits.','One independently accepted fullsample608 final recipe; genuine smoke and full fit.');s=s.replace('# Start shorter664 first. Launch capacity only after its feature assembly and first\n# model complete, with >=12GiB available; otherwise wait and run sequentially.','# A single reviewed final job runs in this saved VM; no concurrent companion here.')
    final_record=save_notebook(notebooks,'tdt4173-fullsample608-final',s,'# Accepted fullsample608 final\nOne fresh smoke003 and one all490896-row600-tree final fit on frozen GPU1. Run in a separate saved VM from the SHOP664 fullsample2021 experiment.')
    template2=ROOT/'artifacts/cuda-xgb608-fullsample-notebook-001/run_fullsample_2021.py';assert sha(template2)=='ac4f0901fadc47653f53d19063f0a82a35210381c113d0a5b831bf25127bbd6b';s=template2.read_text().replace('xgb608_fullsample','xgb664_fullsample').replace('xgb608-fullsample','xgb664-fullsample').replace('cuda_xgb608_loader.py','cuda_xgb664_loader.py');vp=json.loads((payload/'shop664fullsample2021/code/cuda_xgb664_fullsample_protocol.json').read_text());s=s.replace('d5add5941279d980f156015b4b5ca70da2af4b6d8b99c4841050569b02b7255e',children[1][2]).replace('f2eeab5c856ae2848820f5b6c4a8e5d593606a96475f4f2c67f3632029cc33c9',vp['dataset_launcher_sha256'])
    s=replace_once(s,"('train_cuda_tcn.py', 'core_loader_sha256'), ('launch_cuda_xgb.py', 'launcher_sha256')]:","('train_cuda_tcn.py', 'core_loader_sha256'), ('cuda_xgb608_loader.py', 'prior_loader_sha256'), ('launch_cuda_xgb.py', 'launcher_sha256')]:")
    guard="PARENT_SHA = "+repr(sha(pp))+"\n"+'''parent_matches = [p for p in Path('/kaggle/input').rglob('fullsample_parallel_package_manifest.json') if p.is_file() and not p.is_symlink() and sha(p) == PARENT_SHA]
assert len(parent_matches) == 1, 'Require one exact reviewed PRIVATE combined code dataset'
parent_path = parent_matches[0]
for entry in json.loads(parent_path.read_text())['files']:
    rel = Path(entry['path'])
    assert not rel.is_absolute() and '..' not in rel.parts
    member = parent_path.parent / rel
    assert member.is_file() and not member.is_symlink() and member.stat().st_size == entry['size_bytes'] and sha(member) == entry['sha256']
'''
    s=replace_once(s,"protocols = [p for p in Path('/kaggle/input').rglob('cuda_xgb664_fullsample_protocol.json')",guard+"protocols = [p for p in Path('/kaggle/input').rglob('cuda_xgb664_fullsample_protocol.json')")
    s=replace_once(s,"code = protocols[0].parent","code = protocols[0].parent\nassert code == parent_path.parent / 'shop664fullsample2021' / 'code'")
    s=replace_once(s,"print('FULLSAMPLE_SMOKE_PASSED'","assert smoke['smoke']['shape'] == [2048,664] and smoke['smoke']['exact_replay_rows'] == 2048\nprint('FULLSAMPLE_SMOKE_PASSED'")
    interaction_record=save_notebook(notebooks,'tdt4173-shop664-fullsample-2021',s,'# Frozen SHOP664 fullsample2021 interaction\nOne fresh smoke001 and one all367248-row600-tree2021 fold on unchanged GPU1. AcceptedSHOP.25 anchor, original40%rawSHOP control, no2022 command. Separate saved VM from fullsample608 final.')
    report={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'status':'prepared','training':False,'parent_manifest_sha256':sha(pp),'dataset_archive_sha256':sha(archive),'dataset_archive_bytes':archive.stat().st_size,'children':parent['children'],'notebooks':[final_record,interaction_record],'builder_sha256':sha(Path(__file__))};shutil.copy2(__file__,out/Path(__file__).name);(out/'manifest.json').write_text(json.dumps(report,indent=2));(notebooks/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
