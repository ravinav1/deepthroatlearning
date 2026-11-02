"""Prepare a standalone synthetic smoke notebook; no competition-fold launcher."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def sha(path:Path)->str:
    """Digest prepared bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main()->None:
    """Embed exact synthetic worker source in an immutable one-cell notebook."""
    worker=ROOT/'scripts/smoke_cuda_vector14.py';source=worker.read_text()
    compile(source,str(worker),'exec')
    out=ROOT/'artifacts/cuda-vector14-smoke-notebook-001';out.mkdir(exist_ok=False)
    wrapper='''"""Run only a synthetic14-label vector-leaf CUDA/serialization smoke."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib, importlib.metadata, json, shutil, subprocess, sys, time, traceback, zipfile
from IPython.display import FileLink, display

WORKER_SOURCE = __WORKER_SOURCE__
WORKER_SHA = __WORKER_SHA__
PACKAGES = {'numpy':'2.1.3','pandas':'2.3.3','xgboost':'3.4.1','scipy':'1.16.3','scikit-learn':'1.6.1','pyyaml':'6.0.3'}
ROOT = Path('/kaggle/working')
RUN_ID = 'cuda-vector14-smoke-001'
OUT = ROOT / RUN_ID
VENV = ROOT / '.venv-cuda-vector14-smoke-001'
ARCHIVE = ROOT / (RUN_ID + '-download.zip')

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def stamp():
    return datetime.now(timezone.utc).isoformat()

def phase(command, name):
    log = OUT / (name + '.log')
    deadline = time.monotonic() + 120
    print('START_SYNTHETIC_PHASE', name, stamp(), command, flush=True)
    with log.open('xb') as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        while True:
            try:
                code = child.wait(timeout=min(20,max(.1,deadline-time.monotonic())))
                break
            except subprocess.TimeoutExpired:
                print('SYNTHETIC_HEARTBEAT',name,stamp(),log.read_text(errors='replace').splitlines()[-2:],flush=True)
                if time.monotonic() >= deadline:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill(); child.wait()
                    raise RuntimeError('Synthetic phase exceeded120seconds: ' + name)
    if code:
        print(log.read_text(errors='replace')[-20000:],flush=True)
    assert code == 0, 'Synthetic phase failed: ' + name

assert not OUT.exists() and not VENV.exists() and not ARCHIVE.exists(), 'Immutable synthetic run already exists'
OUT.mkdir()
report={'status':'initializing','timestamp_utc':stamp(),'synthetic_only':True,'competition_data_read':False,
        'future_fold_authorized':False,'per_phase_timeout_seconds':120,'worker_sha256':WORKER_SHA}
started=time.monotonic()
try:
    assert hashlib.sha256(WORKER_SOURCE.encode()).hexdigest()==WORKER_SHA, 'Embedded worker changed'
    worker=OUT/'smoke_cuda_vector14.py';worker.write_text(WORKER_SOURCE)
    assert sha(worker)==WORKER_SHA
    assert sys.version.split()[0]=='3.13.15', 'Pinned Python runtime changed'
    assert {name:importlib.metadata.version(name) for name in PACKAGES}==PACKAGES, 'Pinned packages changed'
    uv=shutil.which('uv');assert uv is not None, 'Preinstalled uv required'
    assert subprocess.check_output([uv,'--version'],text=True).strip()=='uv 0.13.5 (x86_64-unknown-linux-gnu)'
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip()
    assert gpu=='Tesla T4, 15360 MiB, 580.178.04\\nTesla T4, 15360 MiB, 580.178.04', 'Pinned2T4 runtime changed'
    print('VECTOR14_SYNTHETIC_RUNTIME_VERIFIED',stamp(),gpu,PACKAGES,flush=True)
    subprocess.run([uv,'venv','--offline','--python',sys.executable,'--system-site-packages',str(VENV)],check=True,timeout=30)
    executable=str(VENV/'bin/python')
    phase([executable,str(worker),'--phase','fit','--directory',str(OUT)],'fit')
    fit_sha=sha(OUT/'fit_manifest.json');fit=json.loads((OUT/'fit_manifest.json').read_text())
    assert fit['status']=='complete' and fit['trees']==8 and fit['vector_leaf_size']==14
    assert fit['target_shape']==[2048,14] and fit['cpu_fallback'] is False
    phase([executable,str(worker),'--phase','replay','--directory',str(OUT),'--fit-manifest-sha256',fit_sha],'replay')
    replay=json.loads((OUT/'replay_manifest.json').read_text())
    assert replay['status']=='complete' and replay['fit_manifest_sha256']==fit_sha
    assert replay['ubj_and_json_replays_bitwise'] and replay['exact_replay_rows']==2048 and replay['output_count']==14
    assert fit['pid']!=replay['pid'] and replay['cpu_fallback'] is False
    for manifest in [fit,replay]:
        for name,h in manifest['artifacts'].items():
            assert sha(OUT/name)==h, name
    report.update(status='complete',fit_manifest_sha256=fit_sha,replay_manifest_sha256=sha(OUT/'replay_manifest.json'),
                  separate_process_replay=True,all14_outputs_verified=True,actual_vector_leaf_size=14,actual_tree_count=8)
except BaseException:
    report.update(status='failed',error=traceback.format_exc())
    raise
finally:
    report['runtime_seconds']=time.monotonic()-started
    report['artifacts']={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'}
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\\n')
    with zipfile.ZipFile(ARCHIVE,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(OUT.iterdir()):
            assert p.is_file() and not p.is_symlink(), 'Unexpected output payload'
            archive.write(p,RUN_ID+'/'+p.name)
    result={'status':report['status'],'archive_size_bytes':ARCHIVE.stat().st_size,
            'archive_sha256':sha(ARCHIVE),'manifest_sha256':sha(OUT/'manifest.json'),
            'fit_manifest_sha256':report.get('fit_manifest_sha256'),'replay_manifest_sha256':report.get('replay_manifest_sha256')}
    print('VECTOR14_SYNTHETIC_RETURN',json.dumps(result),stamp(),flush=True)
    display(FileLink(str(ARCHIVE)))
assert report['status']=='complete'
print('VECTOR14_SYNTHETIC_ONLY_COMPLETE',stamp(),flush=True)
'''.replace('__WORKER_SOURCE__',repr(source)).replace('__WORKER_SHA__',repr(sha(worker)))
    compile(wrapper,'run_vector14_smoke.py','exec')
    path=out/'run_vector14_smoke.py';path.write_text(wrapper)
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},
              'cells':[{'cell_type':'code','execution_count':None,'id':'vector14-synthetic-only','metadata':{},'outputs':[],
                        'source':wrapper.splitlines(keepends=True)}]}
    nbpath=out/'tdt4173-vector14-synthetic-smoke.ipynb';nbpath.write_text(json.dumps(notebook,indent=2)+'\n')
    shutil.copy2(worker,out/worker.name);shutil.copy2(__file__,out/Path(__file__).name)
    report={'status':'prepared_not_executed','created_utc':datetime.now(timezone.utc).isoformat(),
            'training':False,'synthetic_smoke_executed':False,'competition_data_read':False,'full_fold_protocol_exists':False,
            'run_id':'cuda-vector14-smoke-001','worker_sha256':sha(worker),'notebook_sha256':sha(nbpath),'wrapper_sha256':sha(path),
            'hypothesis':'Shared tree partitions with14vector leaf outputs may pool commitment structure; this artifact only checks CUDA binary-multilabel and serialization support.',
            'scope':'2048synthetic rows,664features,14binary labels,8depth3trees; fresh fit/replay processes,120second caps each. No dataset attachment, labels, folds, metrics, selection or submission.',
            'official_sources':['https://xgboost.ai/2026/08/25/introducing-the-xgboost-vector-leaf-model',
                'https://xgboost.readthedocs.io/en/stable/tutorials/multioutput.html',
                'https://raw.githubusercontent.com/dmlc/xgboost/v3.4.1/src/tree/multi_target_tree_model.cc'],
            'future_fold':'Not authorized by this notebook. Any later protocol must freeze600actualvector trees/14outputs, same664features/samples/parameters, strict2021selection thenindependentlocked2022.',
            'artifacts':{p.name:sha(p)for p in out.iterdir()if p.is_file()}}
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'manifest_sha256':sha(out/'manifest.json'),'notebook_sha256':sha(nbpath)}))


if __name__=='__main__':main()
