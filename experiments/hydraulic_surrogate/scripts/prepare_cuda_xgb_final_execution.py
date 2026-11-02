"""Prepare explicit private-notebook transfer/launch/download cells, without running them."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import py_compile
import textwrap

from train_cuda_xgb_final import digest, require

ROOT=Path(__file__).resolve().parents[1]


def main()->None:
    """Write cells for root to use only after independent audit and GPU queue checks."""
    package=ROOT/'artifacts/cuda-xgb-final-package-001'
    manifest=json.loads((package/'manifest.json').read_text())
    require(manifest['status']=='complete','Final package is not complete')
    archive=package/'cuda-xgb-final-code.zip'
    require(digest(archive)==manifest['archive_sha256'],'Code archive changed')
    out=ROOT/'artifacts/cuda-xgb-final-execution-001';out.mkdir(exist_ok=False)
    code_path='/kaggle/working/tdt4173-xgb-final-code-'+manifest['archive_sha256'][:12]
    encoded='\n'.join(textwrap.wrap(base64.b64encode(archive.read_bytes()).decode(),100))
    transfer=f'''# Private reviewed code only; no GPU fit is launched by this cell.
from pathlib import Path
import base64, hashlib, io, json, zipfile
blob=base64.b64decode("""{encoded}""")
assert hashlib.sha256(blob).hexdigest()=={manifest['archive_sha256']!r}
code=Path({code_path!r})
with zipfile.ZipFile(io.BytesIO(blob)) as archive:
    package=json.loads(archive.read('package_manifest.json'))
    assert set(archive.namelist())=={{p['path'] for p in package['files']}}|{{'package_manifest.json'}}
    for record in package['files']:
        payload=archive.read(record['path'])
        assert len(payload)==record['size_bytes'] and hashlib.sha256(payload).hexdigest()==record['sha256']
        assert Path(record['path']).name==record['path']
    code.mkdir(exist_ok=True)
    for name in archive.namelist():
        target=code/name
        payload=archive.read(name)
        if target.exists():assert target.read_bytes()==payload
        else:
            with target.open('xb') as stream:stream.write(payload)
assert hashlib.sha256((code/'cuda_xgb_final_protocol.json').read_bytes()).hexdigest()=={manifest['protocol_sha256']!r}
print('FINAL_CODE_VERIFIED',code)
'''
    launch=f'''# Run only after independent review and the cuda:1 training queue is clear.
from pathlib import Path
import hashlib, json, subprocess, sys
code=Path({code_path!r})
package=json.loads((code/'package_manifest.json').read_text())
for record in package['files']:
    assert hashlib.sha256((code/record['path']).read_bytes()).hexdigest()==record['sha256']
assert hashlib.sha256((code/'cuda_xgb_final_protocol.json').read_bytes()).hexdigest()=={manifest['protocol_sha256']!r}
smoke=Path('/kaggle/working/xgb-runs/cuda-xgb-smoke-001/manifest.json')
assert hashlib.sha256(smoke.read_bytes()).hexdigest()=='f31b812df3cdea770b5f04861608d884faa8bbcaccebbe078a6e95c6736e7e4f'
bundle=Path('/kaggle/input/datasets/ravinav/tdt4173-private-hydraulic-training-inputs')
assert bundle.is_dir()
assert not Path('/kaggle/working/xgb-final-runs/cuda-xgb-final-001').exists()
command=[sys.executable,str(code/'launch_cuda_xgb.py'),'--venv','/kaggle/working/tdt4173-cuda-system-env',
    '--runner',str(code/'train_cuda_xgb_final.py'),'--','--protocol',str(code/'cuda_xgb_final_protocol.json'),
    '--validation-protocol',str(code/'cuda_xgb_protocol.json'),'--validation-runner',str(code/'train_cuda_xgb.py'),
    '--shared-loader',str(code/'train_cuda_tcn.py'),'--bundle',str(bundle),'--smoke-manifest',str(smoke),'--run-id','cuda-xgb-final-001']
with open('/kaggle/working/cuda-xgb-final-001.log','xb') as log:
    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
print('FINAL_PID',process.pid)
'''
    download='''# Run only once the final manifest reports complete.
from pathlib import Path
import hashlib, json, zipfile
from IPython.display import FileLink, display
run=Path('/kaggle/working/xgb-final-runs/cuda-xgb-final-001')
manifest=json.loads((run/'manifest.json').read_text())
assert manifest['status']=='complete' and manifest['completed_units']==14 and manifest['checkpoint_replay_bitwise'] is True
for name,expected in manifest['artifacts'].items():
    assert hashlib.sha256((run/name).read_bytes()).hexdigest()==expected
smoke=Path('/kaggle/working/xgb-runs/cuda-xgb-smoke-001/manifest.json')
archive=Path('/kaggle/working/cuda-xgb-final-001-download.zip')
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as saved:
    for path in sorted(run.rglob('*')):
        if path.is_file():saved.write(path,'cuda-xgb-final-001/'+path.relative_to(run).as_posix())
    saved.write(smoke,'cuda-xgb-smoke-001/manifest.json')
    saved.write('/kaggle/working/cuda-xgb-final-001.log','cuda-xgb-final-001.log')
print('ZIP_SHA',hashlib.sha256(archive.read_bytes()).hexdigest(),'BYTES',archive.stat().st_size)
print('MANIFEST_SHA',hashlib.sha256((run/'manifest.json').read_bytes()).hexdigest())
print('SMOKE_SHA',hashlib.sha256(smoke.read_bytes()).hexdigest())
display(FileLink(str(archive)))
'''
    for name,content in [('transfer_code_cell.py',transfer),('launch_final_cell.py',launch),('download_final_cell.py',download)]:
        (out/name).write_text(content)
        py_compile.compile(str(out/name),cfile=str(out/(name+'.pyc')),doraise=True)
    report={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'status':'complete','purpose':'Prepare cells only; no upload or GPU launch',
            'source_sha256':digest(Path(__file__)),'code_directory':code_path,
            'package_manifest_sha256':digest(package/'manifest.json') if isinstance(package,Path) else digest(ROOT/'artifacts/cuda-xgb-final-package-001/manifest.json'),
            'files':{p.name:digest(p) for p in out.glob('*.py')}}
    (out/'manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    with (ROOT/'docs/experiments.md').open('a') as stream:
        stream.write(f'\n## cuda-xgb-final-execution-001 — {report["timestamp_utc"]}\n\nPrepared explicit private-notebook code transfer, final-fit launch and complete-artifact download cells. All three compile; archive/protocol/smoke hashes are pinned. No upload, model fit or Kaggle submission performed. Root owns queue checks and launch after independent package review. Cell hashes and package identity: `artifacts/cuda-xgb-final-execution-001/manifest.json`.\n')


if __name__=='__main__':main()
