"""Review the unsealed SHOP fullsample2022 builder without inventing actual lock bytes."""
import ast
from datetime import datetime,timezone
import hashlib,json,platform,shutil,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/cuda-xgb664-fullsample-2022-builder-independent-audit-001';OUT.mkdir(exist_ok=False)
start=time.monotonic()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
report={'status':'running','timestamp_utc':datetime.now(timezone.utc).isoformat(),'training':False,'notebook_generated':False,'actual_lock_verified':False,'command':f'{sys.executable} scripts/audit_cuda_shop_fullsample_2022_builder.py','environment':{'python':sys.version,'platform':platform.platform()}}
try:
    path=ROOT/'scripts/prepare_cuda_shop_fullsample_2022_notebook.py'
    assert sha(path)=='b882678fdb347c3a07f6871c9155945d3b337df1eb50ea3a3407bb9a00513ec6'
    source=path.read_text();tree=ast.parse(source);compile(source,str(path),'exec')
    original=ROOT/'scripts/prepare_cuda_fullsample_2022_notebook.py'
    assert original.read_bytes()==(ROOT/'artifacts/cuda-xgb608-fullsample-2022-notebook-001/prepare_cuda_fullsample_2022_notebook.py').read_bytes()
    changes={
      'fixed fullsample2022':'fixed SHOP664 fullsample2022',
      'd5add5941279d980f156015b4b5ca70da2af4b6d8b99c4841050569b02b7255e':'2a5a114ae75d9a34d4a1f580ff7c3a781332fa92120605501b5b76bc72f31c5d',
      'artifacts/cuda-xgb608-fullsample-notebook-001/run_fullsample_2021.py':'artifacts/cuda-fullsample-parallel-notebooks-001/tdt4173-shop664-fullsample-2021.py',
      'ac4f0901fadc47653f53d19063f0a82a35210381c113d0a5b831bf25127bbd6b':'859829b44090a5970007f66ab77e8708235de81ae1ecdf167d6c41affb9083bf',
      'independently-verified-fullsample2021-lock':'independently-verified-shop664-fullsample2021-lock',
      'run_fullsample_2022.py':'run_shop664_fullsample_2022.py',
      '# Frozen fullsample2022 confirmation':'# Frozen SHOP664 fullsample2022 confirmation',
      'tdt4173-fullsample608-frozen-2022.ipynb':'tdt4173-shop664-fullsample-frozen-2022.ipynb'}
    expected=original.read_text()
    for before,after in changes.items():assert before in expected;expected=expected.replace(before,after)
    assert source==expected
    template=ROOT/'artifacts/cuda-fullsample-parallel-notebooks-001/tdt4173-shop664-fullsample-2021.py'
    assert sha(template)=='859829b44090a5970007f66ab77e8708235de81ae1ecdf167d6c41affb9083bf'
    proof=ROOT/'artifacts/cuda-xgb664-fullsample-2021-notebook-independent-audit-001/manifest.json'
    assert sha(proof)=='e592f527e9b37613267640658b92ac601c0f0e67e9c82734b41f2a175efdbf37' and json.loads(proof.read_text())['status']=='passed'
    synthetic=template.read_text().replace('smoke-001','smoke-002').replace('fullsample-2021-001','fullsample-2022-001').replace('FULLSAMPLE2021','FULLSAMPLE2022').replace('smoke and2021 fold','smoke and locked2022 fold')
    substitutions=[]
    for call in ast.walk(tree):
        if isinstance(call,ast.Call)and isinstance(call.func,ast.Name)and call.func.id=='replace_once'and len(call.args)==3 and isinstance(call.args[1],ast.Constant):
            if isinstance(call.args[2],ast.Constant):substitutions.append((call.args[1].value,call.args[2].value))
    assert len(substitutions)==4
    for before,after in substitutions:
        assert synthetic.count(before)==1,before
        synthetic=synthetic.replace(before,after)
    assert synthetic.count("print('PINNED_RUNTIME_VERIFIED', stamp(), gpu, PACKAGES, flush=True)")==1
    compile(synthetic,'<review-only-transformed-source-no-execution>','exec')
    assert "'--year', '2021'" not in synthetic and "'--year', '2022'" in synthetic
    assert "protocol['sample_rows']['2022'] == 428568" in synthetic
    assert "manifest['selection_lock_sha256'] == LOCK_SHA" in synthetic
    report.update(status='passed',builder_sha256=sha(path),template_sha256=sha(template),template_independent_audit_sha256=sha(proof),original_builder_sha256=sha(original),checks=['Exact original audited builder except enumerated protocol/template/namespace/name replacements','Four unique transformations found and compiled against actual audited SHOP2021 template','Genuine eligible external lock+audit required before actual generation; exact embedded lock/hash and immutable2022 args','Fresh smoke002/fold2022-001, full428568-row check, original runtime/source/parent/archive guards inherited'],limitation='Actual sealed notebook/lock byte audit still required after independently eligible2021 result; no invented lock or candidate score.')
except BaseException:
    report.update(status='failed',error=traceback.format_exc());raise
finally:
    shutil.copy2(__file__,OUT/Path(__file__).name)
    report['runtime_seconds']=time.monotonic()-start
    report['artifacts']={p.name:sha(p)for p in OUT.iterdir()if p.is_file()and p.name!='manifest.json'}
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps({'status':report['status'],'manifest_sha256':sha(OUT/'manifest.json'),'runtime_seconds':report['runtime_seconds']}))
