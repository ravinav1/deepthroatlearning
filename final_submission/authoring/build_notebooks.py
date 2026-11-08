from pathlib import Path
import ast, json, textwrap, hashlib
BASE=Path('/Users/heap/deepthroatlearning')
OUT=Path(__file__).resolve().parents[1]
SRC=BASE/'experiments/hydraulic_surrogate/sintef_project'

def selected_source(path,names,prelude):
    source=path.read_text(); tree=ast.parse(source)
    return prelude+'\n\n'+'\n\n'.join(ast.get_source_segment(source,n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.Assign)) and ((isinstance(n,ast.FunctionDef) and n.name in names) or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets))))+'\n'
startup=selected_source(BASE/'scripts/supplied_only/startup_like.py',{'PARAMS','STARTUP_COLUMNS','startup_fields','build_rows'},'import numpy as np\nimport pandas as pd\nfrom sintef_project.canonical_unit_features import canonical_block\nfrom sintef_project.lp_features import UNIT_ORDER as UNITS')
projection=selected_source(BASE/'scripts/supplied_only/blend.py',{'TOKKE','project_tokke'},'import numpy as np\nfrom sintef_project.lp_features import UNIT_ORDER as UNITS')
modules=['features','physics','context_features','normalized','lp_features','lp_head_update','lp_shared_loss_update','lp_joint_plant','lp_shop_relaxation','lp_shop_features','lp_context','canonical_unit_features','postprocess']
# Only package code is embedded. No model files, arrays, or training-data payloads.
module_sources={n:(SRC/f'{n}.py').read_text() for n in modules}
module_sources['startup']=startup
module_sources['tokke']=projection
cfg=json.load(open(BASE/'configs/hydraulic_surrogate_best.json'))
component_params={}
columns={}
for c in cfg['components']:
    m=json.load(open(BASE/'models/hydraulic_surrogate'/c['directory']/'manifest.json'))
    columns[c['name']]=m['features']
    component_params[c['name']]=m.get('model_parameters',m.get('model_parameters_requested'))
# Column names describe the feature schema; numerical predictions are not embedded.
(OUT/'evidence/component_parameters.json').write_text(json.dumps(component_params,indent=2))

setup='''from pathlib import Path
import os, sys, json, shutil, time, hashlib, tempfile
# Set this to the supplied sintef_project directory (contains data/, mapping and template).
RAW_DATA = Path("sintef_project").resolve()
THREADS = 4
# The historical XGBoost members used one Tesla T4 at a time; CPU retraining is NOT equivalent.
XGB_DEVICE = "cuda:0"
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = str(THREADS)
assert RAW_DATA.is_dir(), f"Set RAW_DATA to the supplied project directory: {RAW_DATA}"
WORK = Path(tempfile.mkdtemp(prefix="tdt4173_reproduction_", dir=Path.cwd()))
PKG = WORK / "sintef_project"
PKG.mkdir()
(PKG / "__init__.py").write_text("")
STARTED = time.monotonic()
RAW_FILES = [
    "data/kernel/Historical_day_ahead_price_2015_2025.csv",
    "data/kernel/Historical_inflow_1958_2025.csv",
    "data/kernel/Historical_volume_2015_2024.csv",
    "data/kernel/Synthetic_water_value_2015_2024.csv",
    "data/kernel/Unit_commitment_decisions.csv",
    "data/extended/Constraint_min_flow.csv", "data/extended/Constraint_min_volume.csv",
    "data/extended/Tokke_Vinje_topology.yaml", "prediction_mapping.csv", "sample_submission.csv",
]
raw_hashes = {}
for relative in RAW_FILES:
    source = RAW_DATA / relative
    assert source.is_file(), source
    target = PKG / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    raw_hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
(WORK / "raw_hashes.json").write_text(json.dumps(raw_hashes, indent=2))
sys.path.insert(0, str(WORK))
# Prevent accidental reuse of modules from a preceding notebook in the same kernel.
for name in list(sys.modules):
    if name == "sintef_project" or name.startswith("sintef_project."):
        del sys.modules[name]
print("Fresh offline working directory:", WORK)
'''
imports='''import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import yaml
from threadpoolctl import threadpool_limits
threadpool_limits(THREADS)
from sintef_project.features import build_features
from sintef_project.physics import add_physics_features
from sintef_project.context_features import add_context_features
from sintef_project.normalized import add_normalized_features
from sintef_project.lp_context import add_lp_context_features
from sintef_project.lp_features import HydroCaseData, HydraulicLP, result_features, UNIT_ORDER
from sintef_project.lp_joint_plant import JointPlantLP, build_joint_curve, result_joint_features
from sintef_project.lp_shop_relaxation import ShopRelaxationLP
from sintef_project.lp_shop_features import compute_features, COLUMNS as SHOP_COLUMNS
from sintef_project.canonical_unit_features import feature_specification
from sintef_project.startup import PARAMS, build_rows
from sintef_project.postprocess import project_vinje
from sintef_project.tokke import project_tokke
print("Versions:", np.__version__, pd.__version__, lgb.__version__, xgb.__version__)
assert xgb.__version__ == "3.4.1", "Use the archived CUDA XGBoost version, 3.4.1."
assert xgb.build_info().get("USE_CUDA"), "A CUDA-enabled XGBoost build and one GPU are required."
'''
lp_code='''# Solve all three approximations afresh, using case inputs and no commitment targets.
inputs = HydroCaseData(WORK)
base_solver = HydraulicLP(inputs.network)
joint_solver = JointPlantLP(inputs.network)
shop_solver = ShopRelaxationLP(inputs.network)
lp_rows, joint_rows, shop_rows = [], [], []
for i in range(len(inputs.cases)):
    case = inputs.case(i)
    base_result = base_solver.solve(case, time_limit=20.0)
    base_frame = result_features(case, base_result, inputs.network)
    curves = [build_joint_curve(inputs, i, plant) for plant in ("Tokke", "Vinje")]
    joint_result = joint_solver.solve(case, curves, time_limit=20.0)
    joint_frame = result_joint_features(case, joint_result, inputs.network)
    # Preserve the original baseline arithmetic, then use the corrected RHS in the linked-state LP.
    case["natural"] = np.asarray(case["natural"], dtype=np.float64)
    for name, field in (("pmin", "p_min"), ("start_cost", "startcost_const"), ("stop_cost", "stopcost_const")):
        case[name] = np.asarray([inputs.model["generator"][u][field] for u in UNIT_ORDER], dtype=np.float64)
    shop_result = shop_solver.solve(case, 1.0)
    lp_rows.append(base_frame.to_numpy(dtype=np.float32))
    joint_rows.append(joint_frame.to_numpy(dtype=np.float32))
    shop_rows.append(compute_features(case, shop_result, base_result, inputs.network))
    if (i+1) % 100 == 0:
        print("Hydraulic cases:", i+1, "/", len(inputs.cases), flush=True)
lp_frame = pd.DataFrame(np.concatenate(lp_rows), columns=base_frame.columns)
joint_frame = pd.DataFrame(np.concatenate(joint_rows), columns=joint_frame.columns)
shop_frame = pd.DataFrame(np.concatenate(shop_rows), columns=SHOP_COLUMNS)
assert len(lp_frame) == len(joint_frame) == len(shop_frame) == 3647*168
assert all(np.isfinite(f.to_numpy()).all() for f in (lp_frame, joint_frame, shop_frame))
del lp_rows, joint_rows, shop_rows, inputs, base_solver, joint_solver, shop_solver
'''
features='''base, labels, train, mapping, meta = build_features(WORK)
base, _ = add_physics_features(base, WORK)
base, _ = add_context_features(base, WORK)
UNITS = meta["units"]
assert UNITS == UNIT_ORDER
Y = np.column_stack([labels[u] for u in UNITS]).astype(np.uint8)
NTRAIN = len(train)*168
assert (len(train), len(mapping), len(UNITS)) == (2922, 725, 14)
sample = np.sort(np.random.default_rng(4173).choice(NTRAIN, size=int(NTRAIN*.4), replace=False))
assert len(sample) == 196358
np.save(WORK / "training_sample.npy", sample)
# The archived 395/521/404/608 feature sets use different normalization order.
def feature_table(name):
    if name == "joint":
        f = pd.concat([base, joint_frame], axis=1)
        return add_normalized_features(f, WORK)[0]
    f = pd.concat([base, lp_frame], axis=1)
    if name != "lp":
        f = add_lp_context_features(f, WORK)[0]
    f = add_normalized_features(f, WORK)[0]
    if name in ("context_joint", "gpu608", "gpu664", "lgb664"):
        f = pd.concat([f, joint_frame], axis=1)
    if name in ("gpu664", "lgb664"):
        f = pd.concat([f, shop_frame], axis=1)
    return f
'''
traincode='''predictions = {}
for name in ("lp", "context", "joint", "context_joint", "gpu608", "gpu664", "lgb664"):
    frame = feature_table(name)
    assert frame.columns.tolist() == EXPECTED_COLUMNS[name], f"Feature order differs: {name}"
    assert np.isfinite(frame.to_numpy()).all()
    xt, xv = frame.iloc[sample], frame.iloc[NTRAIN:]
    params = dict(COMPONENT_PARAMS[name])
    params["n_jobs"] = min(THREADS, 2) if name.startswith("gpu") or name == "lgb664" else min(THREADS, 3)
    if name.startswith("gpu"):
        params["device"] = XGB_DEVICE
    pred = np.empty((len(xv), 14), dtype=np.float32)
    for j, unit in enumerate(UNITS):
        model = xgb.XGBClassifier(**params) if name.startswith("gpu") else lgb.LGBMClassifier(**params)
        model.fit(xt, Y[sample, j])
        if name.startswith("gpu"):
            actual = json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]
            assert actual.startswith("cuda"), "XGBoost fell back to CPU; stop instead of changing the recipe."
        pred[:, j] = model.predict_proba(xv)[:, 1]
        if name.startswith("gpu"):
            model.save_model(WORK / f"{name}_{unit}.ubj")
        else:
            model.booster_.save_model(str(WORK / f"{name}_{unit}.txt"))
        print(name, j+1, "/14", unit, flush=True)
        del model
    predictions[name] = pred.astype(np.float64)
    np.save(WORK / f"predictions_{name}.npy", pred)
    del frame, xt, xv, pred
anchor = project_vinje(sum(w * predictions[name] for name, w in
    {"lp":.140625, "context":.421875, "joint":.1875, "context_joint":.25}.items()))
for name, weight in (("gpu608", .5), ("gpu664", .25), ("lgb664", .1)):
    anchor = project_vinje((1-weight)*anchor + weight*predictions[name])
np.save(WORK / "anchor.npy", anchor)
del predictions
'''
poolcode='''frame = feature_table("lgb664")
# startup16 was trained from the float32 rebuilt table, preserving that conversion.
frame = frame.astype(np.float32)
spec = feature_specification(frame.columns.tolist(), UNITS)
topology = yaml.safe_load((PKG / "data/extended/Tokke_Vinje_topology.yaml").read_text())["model"]["generator"]
ref = max(float(topology[u]["p_max"]) for u in UNITS)
constants = {u:{"pmin":float(topology[u]["p_min"]), "pmax":float(topology[u]["p_max"]),
    "start":float(topology[u]["startcost_const"]), "stop":float(topology[u]["stopcost_const"]), "ref":ref} for u in UNITS}
xt = build_rows(frame, sample, spec, constants)
xv = build_rows(frame, np.arange(NTRAIN,len(frame)), spec, constants)
yt = Y[sample].ravel()
assert xt.shape == (2749012,131) and xv.shape == (1705200,131)
startup_predictions = []
for seed in STARTUP_SEEDS:
    params = dict(PARAMS, random_state=seed, n_jobs=THREADS,
        subsample=1.0 if seed == 4173 else .8, subsample_freq=0 if seed == 4173 else 1)
    model = lgb.LGBMClassifier(**params)
    model.fit(xt, yt, categorical_feature=[111,112])
    model.booster_.save_model(str(WORK / f"startup16_{seed}.txt"))
    startup_predictions.append(model.booster_.predict(xv, num_threads=THREADS).reshape(-1,14))
    print("Completed startup16 seed",seed,flush=True)
    del model
probabilities = project_tokke(project_vinje(.75*anchor + .25*np.mean(startup_predictions,axis=0)))
'''
export='''template = pd.read_csv(PKG / "sample_submission.csv")
expected = [f"result_committed_{u}_t{h}" for u in UNITS for h in range(168)]
assert template.columns.tolist() == ["Run No"] + expected
assert mapping["Run No"].tolist() == template["Run No"].tolist()
assert probabilities.shape == (725*168,14)
assert np.isfinite(probabilities).all() and probabilities.min() >= 0 and probabilities.max() <= 1
wide = probabilities.reshape(725,168,14).transpose(0,2,1).reshape(725,-1)
result = pd.DataFrame(wide,columns=expected)
result.insert(0,"Run No",mapping["Run No"].to_numpy())
OUTPUT = Path.cwd() / OUTPUT_NAME
result.to_csv(OUTPUT,index=False,float_format="%.9g")
readback = pd.read_csv(OUTPUT)
assert readback.shape == (725,2353) and readback.columns.tolist() == template.columns.tolist()
assert readback["Run No"].tolist() == template["Run No"].tolist()
assert np.max(np.abs(readback.iloc[:,1:].to_numpy()-wide)) <= 5.000001e-10
actual_sha = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
receipt = {"file": str(OUTPUT), "sha256": actual_sha, "historical_sha256": HISTORICAL_SHA,
    "historical_byte_match": actual_sha == HISTORICAL_SHA, "elapsed_seconds":time.monotonic()-STARTED,
    "packages":{"numpy":np.__version__,"pandas":pd.__version__,"lightgbm":lgb.__version__,"xgboost":xgb.__version__},
    "raw_hashes":raw_hashes,"seeds":STARTUP_SEEDS,"device":XGB_DEVICE}
(WORK / "reproduction_receipt.json").write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2))
if actual_sha != HISTORICAL_SHA:
    print("Numerical reproduction differs from the historical CSV. Score equivalence is not established by this run.")
assert receipt["elapsed_seconds"] < 12*3600, "Runtime exceeded the course's 12-hour limit."
'''
def md(t):return {'cell_type':'markdown','metadata':{},'source':t.splitlines(True)}
def code(t):return {'cell_type':'code','execution_count':None,'outputs':[],'metadata':{},'source':t.splitlines(True)}
for number,score,seeds,sha in [(1,'0.99657',[4173,1],'f7b0b52fc54e93e1db6171fdf85a7955486c3439897911291eef98c70a4f716a'),(2,'0.99656',[4173],'a45f8001a87afbc67b17d95fb39c9f1982593d24f91887d5747fec1f3c367ceb')]:
    cells=[md(f'''# Short notebook {number}: hydropower commitment prediction
**Group:** 167{'  '}
**Kaggle team:** deepthroatlearning (ensure the Kaggle display name includes 167){'  '}
**Full names and student IDs:** TO BE COMPLETED BY GROUP MEMBERS

Historical public ROC-AUC: **{score}**. The report gives the full experiment history.

## Reproduction instructions
Run all cells in a fresh kernel after setting `RAW_DATA`. Everything needed to construct features and fit the models is contained in this notebook. Only the ten supplied raw files and preinstalled packages are read. There are no network calls, downloaded weights or embedded predictions. Temporary Python modules and checkpoints are created during execution, as permitted by the grading guide.

Use Python 3.11, the attached requirements, at most four CPU threads and **one NVIDIA Tesla T4 (16 GB)** or a comparable CUDA GPU. The historical XGBoost model fits used one T4 each; the notebook runs the two groups sequentially on the same GPU. Original LightGBM and XGBoost fits used different Python environments, so this consolidated environment still needs an end-to-end raw-data rerun. CPU XGBoost retraining is not an interchangeable substitute.

**Verification status:** the saved final composition reproduces the historical CSV exactly. The complete raw-data training notebook has not yet been executed end to end on the required GPU, and the 12-hour/32-GB limits are not yet measured for this combined workflow. No unexecuted output below is presented as a measured result.

## 1. Raw inputs and a fresh working directory
'''),code(setup),code(f'STARTUP_SEEDS = {seeds!r}\nHISTORICAL_SHA = {sha!r}\nOUTPUT_NAME = "submission_{score}.csv"\n')]
    cells.append(md('## 2. Self-contained feature and hydraulic solver code\nThe following cells store readable implementation code in the fresh working directory. These are generated intermediate files, not dependencies distributed alongside the notebook.'))
    for name,source in module_sources.items():
        # Plain source strings remain readable/editable; no compressed or encoded code payload.
        cells.append(md(f'### {name.replace("_"," ").title()}'))
        cells.append(code(f'(PKG / "{name}.py").write_text('+repr(source)+', encoding="utf-8")\n'))
    cells += [code(imports),md('## 3. Compute the three input-only hydraulic feature blocks'),code(lp_code),md('## 4. Build chronological case-hour features and the fixed training sample'),code(features),code('COMPONENT_PARAMS = '+repr(component_params)+'\nEXPECTED_COLUMNS = '+repr(columns)+'\n'),md('## 5. Fit the seven component groups and preserve each nested projection'),code(traincode),md('## 6. Fit the pooled startup16 model'+('s and average their probabilities' if number==1 else '')),code(poolcode),md('## 7. Export and verify the submission'),code(export)]
    # Store long embedded module sources as readable multiline literals, not escaped one-line blobs.
    for c in cells:
        if c['cell_type']=='code' and ''.join(c['source']).startswith('(PKG / '):
            line=''.join(c['source']); name=line.split('"')[1].removesuffix('.py');source=module_sources[name]
            c['source']=(f'module_source = r\'\'\'\n{source}\'\'\'\n(PKG / "{name}.py").write_text(module_source.lstrip("\\n"), encoding="utf-8")\n').splitlines(True)
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11.15'}},'cells':cells}
    for i,c in enumerate(cells):c['id']=f'cell-{i:03d}'
    path=OUT/'canvas'/f'Short notebook {number}.ipynb';path.write_text(json.dumps(nb,indent=1))
    for i,c in enumerate(cells):
        if c['cell_type']=='code':compile(''.join(c['source']),f'{path.name} cell {i}','exec')
(OUT/'canvas/requirements.txt').write_text('''# Install before the offline grading run; Python 3.11.15.
# Historical LightGBM/solver packages below; CUDA-enabled XGBoost 3.4.1.
# The consolidated environment needs an end-to-end reproduction run.
numpy==2.4.6
pandas==3.0.6
scipy==1.17.1
PyYAML==6.0.3
lightgbm==4.7.0
xgboost==3.4.1
scikit-learn==1.9.1
threadpoolctl==3.7.0
''')
# Helpers for local validation only; never required by the submitted notebooks.
(OUT/'authoring/runtime_sections.json').write_text(json.dumps(dict(setup=setup,imports=imports,lp=lp_code,features=features,train=traincode,pool=poolcode,export=export,module_sources=module_sources)))
print('Wrote and syntax-checked both standalone notebooks')
