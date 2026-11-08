from pathlib import Path
import ast, json, hashlib, subprocess
BASE = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
OUT=Path(__file__).resolve().parents[1]
SOURCE_REVISION = subprocess.check_output(['git', 'rev-parse', '6ccab5c'], cwd=BASE, text=True).strip()
source_hashes = {}

def source_text(path):
    relative = str(Path(path).relative_to(BASE)) if Path(path).is_absolute() else str(path)
    source = subprocess.check_output(['git', 'show', f'{SOURCE_REVISION}:{relative}'], cwd=BASE, text=True)
    source_hashes[relative] = hashlib.sha256(source.encode()).hexdigest()
    return source

SRC=BASE/'experiments/hydraulic_surrogate/sintef_project'

def selected_source(path,names,prelude):
    source=source_text(path); tree=ast.parse(source)
    return prelude+'\n\n'+'\n\n'.join(ast.get_source_segment(source,n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.Assign)) and ((isinstance(n,ast.FunctionDef) and n.name in names) or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets))))+'\n'
startup=selected_source(BASE/'scripts/supplied_only/startup_like.py',{'PARAMS','STARTUP_COLUMNS','startup_fields','build_rows'},'import numpy as np\nimport pandas as pd\nfrom sintef_project.canonical_unit_features import canonical_block\nfrom sintef_project.lp_features import UNIT_ORDER as UNITS')
projection=selected_source(BASE/'scripts/supplied_only/blend.py',{'TOKKE','project_tokke'},'import numpy as np\nfrom sintef_project.lp_features import UNIT_ORDER as UNITS')
projection = projection.replace(
    'Sum of the four probabilities is not exactly preserved; values stay in [0, 1].',
    'Pair averaging preserves the four-probability sum up to roundoff; values stay in [0, 1].')
modules=['features','physics','context_features','normalized','lp_features','lp_head_update','lp_shared_loss_update','lp_joint_plant','lp_shop_relaxation','lp_shop_features','lp_context','canonical_unit_features','postprocess']
# Only package code is embedded. No model files, arrays, or training-data payloads.
module_sources={n:source_text(SRC/f'{n}.py') for n in modules}
module_sources['startup']=startup
module_sources['tokke']=projection
cfg=json.loads(source_text('configs/hydraulic_surrogate_best.json'))
component_params={}
columns={}
manifest_hashes={}
for c in cfg['components']:
    manifest = BASE/'models/hydraulic_surrogate'/c['directory']/'manifest.json'
    m=json.loads(manifest.read_text())
    manifest_hashes[str(manifest.relative_to(BASE))] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    columns[c['name']]=m['features']
    component_params[c['name']]=m.get('model_parameters',m.get('model_parameters_requested'))
# Column names describe the feature schema; numerical predictions are not embedded.
legacy_modules = ['features', 'topology_features', 'price_features', 'operating_features',
                  'hydraulic_features', 'water_budget_features', 'temporal_features', 'tree_preprocessing']
legacy_sources = {}
for name in legacy_modules:
    source = source_text(f'src/{name}.py')
    for dependency in legacy_modules:
        source = source.replace(f'from {dependency} import ', f'from sintef_project.legacy_{dependency} import ')
    if name == 'features':
        source = source.replace('Path(__file__).resolve().parents[1]', 'Path(__file__).resolve().parent')
        source = source.replace('data/raw/', 'data/').replace('configs/Tokke_Vinje_topology.yaml', 'data/extended/Tokke_Vinje_topology.yaml')
        source = source.replace('data/sample_submission.csv', 'sample_submission.csv')
    legacy_sources[f'legacy_{name}'] = source
legacy_sources['legacy_training'] = selected_source(BASE/'src/round5_lightgbm.py',
    {'OPTIONS', 'generator_features', 'training_rows'},
    'import numpy as np\nfrom sintef_project.legacy_features import HOURS, build_features\nfrom sintef_project.legacy_tree_preprocessing import rescale_tree_features')
legacy_configs = {}
legacy_columns = {}
legacy_historical_hashes = {}
for seed in (299, 673):
    config = json.loads(source_text(f'configs/round10_legacy_so_seed{seed}_final.json'))
    legacy_configs[seed] = {key: value for key, value in config.items()
                            if key.startswith('use_') or key in ('model', 'hour_stride', 'trees_by_generator')}
    report = json.loads(source_text(f'reports/round10_legacy_so_seed{seed}_final_run.json'))
    legacy_columns[seed] = {unit: record['feature_names'] for unit, record in report['generator_models'].items()}
    legacy_historical_hashes[seed] = report['prediction_sha256']
    assert all(len(names) == 115 for names in legacy_columns[seed].values())
source_text('scripts/supplied_only/export_candidate.py')
source_text('configs/supplied_only/secondary_anchor_startup16_legacy.json')
source_text('scripts/supplied_only/legacy_member.py')
(OUT/'evidence/component_parameters.json').write_text(json.dumps(
    {'hydraulic': component_params, 'legacy': legacy_configs}, indent=2))

setup='''from pathlib import Path
import os, sys, json, shutil, time, hashlib, tempfile, gc, resource
# Set this to the supplied sintef_project directory (contains data/, mapping and template).
RAW_DATA = Path("sintef_project").resolve()
THREADS = 4
# The historical XGBoost members used one Tesla T4 at a time; CPU retraining is NOT equivalent.
XGB_DEVICE = "cuda:0"
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
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
def file_sha256(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

def peak_memory_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak*1024)

for relative in RAW_FILES:
    source = RAW_DATA / relative
    assert source.is_file(), source
    target = PKG / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    raw_hashes[relative] = file_sha256(source)
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
from importlib.metadata import version
from threadpoolctl import threadpool_limits
threadpool_limits(THREADS)
EXPECTED_PACKAGES = {"numpy":"2.4.6", "pandas":"3.0.6", "scipy":"1.17.1",
    "PyYAML":"6.0.3", "lightgbm":"4.7.0", "xgboost":"3.4.1",
    "scikit-learn":"1.9.1", "threadpoolctl":"3.7.0"}
package_versions = {name:version(name) for name in EXPECTED_PACKAGES}
assert package_versions == EXPECTED_PACKAGES, f"Install the pinned requirements before running: {package_versions}"
assert sys.version_info[:2] == (3,11), "Use the documented Python 3.11 environment."
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
del case, curves, base_result, joint_result, shop_result, base_frame
gc.collect()
'''
features='''base, labels, train, mapping, meta = build_features(WORK)
base, _ = add_physics_features(base, WORK)
base, _ = add_context_features(base, WORK)
UNITS = meta["units"]
assert UNITS == UNIT_ORDER
Y = np.column_stack([labels[u] for u in UNITS]).astype(np.uint8)
NTRAIN = len(train)*168
assert (len(train), len(mapping), len(UNITS)) == (2922, 725, 14)
del labels, train, meta
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
    gc.collect()
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
# Release the full feature blocks before fitting the large pooled table.
del frame, base, lp_frame, joint_frame, shop_frame, Y, sample
gc.collect()
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
probabilities = project_vinje(.75*anchor + .25*np.mean(startup_predictions,axis=0))
del xt, xv, yt, anchor, startup_predictions, spec, constants, topology
gc.collect()
'''
legacy_code='''from sintef_project.legacy_features import load_data as load_legacy_data, prediction_frame
from sintef_project.legacy_training import generator_features, training_rows

legacy_train, legacy_labels, legacy_units, legacy_inputs, legacy_topology = load_legacy_data(use_extended=True)
assert legacy_units == UNITS and len(legacy_train) == 2922
legacy_ids = pd.read_csv(PKG / "sample_submission.csv", usecols=["Run No"])["Run No"]
legacy_test = pd.read_csv(PKG / "prediction_mapping.csv")
legacy_test["starttime"] = pd.to_datetime(legacy_test["starttime"], utc=True)
assert legacy_ids.is_unique and legacy_test["Run No"].is_unique
assert set(legacy_ids) == set(legacy_test["Run No"])
assert not legacy_train["Run No"].isin(legacy_test["Run No"]).any()
legacy_test = legacy_test.set_index("Run No").loc[legacy_ids].reset_index()
legacy_arrays, legacy_receipts = [], []
for seed in (299, 673):
    config = LEGACY_CONFIGS[seed]
    rows = training_rows(legacy_train, config["hour_stride"], seed)
    assert len(rows) == 122724
    per_unit = []
    for index, unit in enumerate(legacy_units):
        target_columns = legacy_labels[index*168:(index+1)*168]
        # Complete weeks first; construct only one generator's table at a time.
        x_test = generator_features(legacy_test, unit, index, legacy_inputs, legacy_topology, config)
        x_train = generator_features(legacy_train, unit, index, legacy_inputs, legacy_topology, config)
        assert x_test.columns.tolist() == x_train.columns.tolist() == LEGACY_COLUMNS[seed][unit]
        assert all(dtype == np.dtype("float32") for dtype in x_train.dtypes)
        y_train = legacy_train[target_columns].to_numpy(dtype=np.uint8).ravel()
        params = dict(config["model"], n_estimators=config["trees_by_generator"][unit])
        assert params["n_jobs"] <= THREADS
        model = lgb.LGBMClassifier(objective="binary", metric="auc", **params)
        model.fit(x_train.iloc[rows], y_train[rows], categorical_feature=["generator_id"])
        booster = model.booster_
        assert booster.current_iteration() == params["n_estimators"]
        booster.save_model(str(WORK / f"legacy_{seed}_{unit.replace(' ', '_')}.txt"))
        del x_train, y_train, model
        prediction = booster.predict(x_test, num_threads=1)
        assert prediction.shape == (725*168,) and np.isfinite(prediction).all()
        assert prediction.min() >= 0 and prediction.max() <= 1
        per_unit.append(prediction)
        del booster, x_test, prediction
        gc.collect()
        print("Legacy", seed, index+1, "/14", unit, flush=True)
    legacy_csv = WORK / f"round10_legacy_so_seed{seed}_final_predictions.csv"
    legacy_frame = prediction_frame(np.concatenate(per_unit), legacy_test, legacy_labels, legacy_units)
    # Preserve the historical default CSV export and pandas default parser round-trip.
    legacy_frame.to_csv(legacy_csv, index=False)
    reread = pd.read_csv(legacy_csv).set_index("Run No").loc[legacy_ids, legacy_labels]
    legacy_arrays.append(reread.to_numpy(dtype=np.float64).reshape(725,14,168).transpose(0,2,1).reshape(-1,14))
    digest = file_sha256(legacy_csv)
    legacy_receipts.append({"seed":seed, "sha256":digest,
        "historical_sha256":LEGACY_HISTORICAL_HASHES[seed],
        "historical_byte_match":digest == LEGACY_HISTORICAL_HASHES[seed]})
    del per_unit, legacy_frame, reread
    gc.collect()
probabilities = project_vinje(.95*probabilities + .05*np.mean(legacy_arrays, axis=0))
del legacy_arrays, legacy_train, legacy_inputs, legacy_topology, legacy_test
gc.collect()
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
actual_sha = file_sha256(OUTPUT)
receipt = {"file": str(OUTPUT), "sha256": actual_sha, "historical_sha256": HISTORICAL_SHA,
    "historical_byte_match": actual_sha == HISTORICAL_SHA, "elapsed_seconds":time.monotonic()-STARTED,
    "packages":package_versions,"python":sys.version,"platform":sys.platform,"xgboost_build":xgb.build_info(),
    "raw_hashes":raw_hashes,"seeds":STARTUP_SEEDS,"device":XGB_DEVICE,
    "catalogue_number":CATALOGUE_NUMBER,"threads":THREADS,"peak_memory_bytes":peak_memory_bytes(),
    "legacy_members":legacy_receipts if USE_LEGACY else [],
    "validation_scores_computed":False}
RECEIPT = OUTPUT.with_suffix(".receipt.json")
RECEIPT.write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2))
if actual_sha != HISTORICAL_SHA:
    print("Numerical reproduction differs from the historical CSV. Score equivalence is not established by this run.")
assert receipt["elapsed_seconds"] < 12*3600, "Runtime exceeded the course's 12-hour limit."
assert receipt["peak_memory_bytes"] <= 32*1024**3, "Peak process memory exceeded 32 GiB."
assert actual_sha == HISTORICAL_SHA, "Historical byte match failed; inspect the saved receipt before using this output."
'''
def md(t):return {'cell_type':'markdown','metadata':{},'source':t.splitlines(True)}
def code(t):return {'cell_type':'code','execution_count':None,'outputs':[],'metadata':{},'source':t.splitlines(True)}
notebook_records = []
for number,score,seeds,sha in [(1,'0.99657',[4173,1],'f7b0b52fc54e93e1db6171fdf85a7955486c3439897911291eef98c70a4f716a'),(2,'0.99654',[4173],'38cf96b1165324ac6e63c77d5284e15d499dda534f3871ebea82f11471d78f67')]:
    use_legacy = number == 2
    catalogue = 4 if use_legacy else 3
    status = ('The historical CSV hash is recorded; saved-component replay of this legacy blend has not been verified locally.'
              if use_legacy else 'An earlier saved-component replay matched the historical CSV. This does not verify retraining from raw inputs.')
    cells=[md(f'''# Short notebook {number}: hydropower commitment prediction
**Group:** 167{'  '}
**Authors:** Hallvard Grundt — 115526; Raviksan Navaratnarajah — 109485; Bøye Tvenge Einang — 139410.

**Selected submission #{catalogue}; historical public ROC-AUC: {score}.**

## Reproduction instructions
Set `RAW_DATA` to the supplied project folder and run all cells in a fresh kernel. The notebook reads the ten supplied files and rebuilds all features and models offline. It creates temporary Python modules and checkpoints; no saved predictions, model weights or repository checkout are required.

Use Python 3.11.15, the pinned requirements, at most four CPU threads, 32 GB RAM and **one NVIDIA Tesla T4 (16 GB)**. The GPU model groups run sequentially. The original models used separate Python environments, so the combined environment still needs a complete rerun. CPU XGBoost training changes the recipe.

**Verification status:** {status} Neither complete notebook has been run from raw inputs on the required GPU. Total runtime and peak RAM remain unmeasured. The final cell writes a receipt, checks the schema and resource limits, and requires an exact historical CSV hash match.

## 1. Raw inputs and a fresh working directory
'''),code(setup),code(f'STARTUP_SEEDS = {seeds!r}\nHISTORICAL_SHA = {sha!r}\nOUTPUT_NAME = "submission_{score}.csv"\nCATALOGUE_NUMBER = {catalogue}\nUSE_LEGACY = {use_legacy!r}\n')]
    cells.append(md('## 2. Self-contained feature and hydraulic solver code\nThe following cells store readable implementation code in the fresh working directory. These are generated intermediate files, not dependencies distributed alongside the notebook.'))
    notebook_modules = dict(module_sources)
    if use_legacy:
        notebook_modules.update(legacy_sources)
    for name,source in notebook_modules.items():
        # Plain source strings remain readable/editable; no compressed or encoded code payload.
        cells.append(md(f'### {name.replace("_"," ").title()}'))
        cells.append(code(f'(PKG / "{name}.py").write_text('+repr(source)+', encoding="utf-8")\n'))
    cells += [code(imports),md('## 3. Compute hydraulic features from supplied inputs'),code(lp_code),md('## 4. Build complete-week features and the fixed training sample'),code(features),code('COMPONENT_PARAMS = '+repr(component_params)+'\nEXPECTED_COLUMNS = '+repr(columns)+'\n'),md('## 5. Fit seven component groups and apply each Vinje projection'),code(traincode),md('## 6. Fit startup16'+(' seeds 4173 and 1, then average' if number==1 else ' seed 4173')),code(poolcode)]
    if use_legacy:
        cells += [md('## 7. Add the two legacy models\nFit seeds 299 and 673 with the fixed tree counts. Preserve each CSV round-trip, average them, add 5% to the startup blend, and apply Vinje projection again.'),
                  code('LEGACY_CONFIGS = '+repr(legacy_configs)+'\nLEGACY_COLUMNS = '+repr(legacy_columns)+'\nLEGACY_HISTORICAL_HASHES = '+repr(legacy_historical_hashes)+'\n'),code(legacy_code)]
    cells += [md('## '+str(8 if use_legacy else 7)+'. Apply Tokke projection, export and check'),
              code('probabilities = project_tokke(probabilities)\n'),code(export)]
    # Store long embedded module sources as readable multiline literals, not escaped one-line blobs.
    for c in cells:
        if c['cell_type']=='code' and ''.join(c['source']).startswith('(PKG / '):
            line=''.join(c['source']); name=line.split('"')[1].removesuffix('.py');source=notebook_modules[name]
            assert "'''" not in source, f"Embedded delimiter collision: {name}"
            c['source']=(f'module_source = r\'\'\'\n{source}\'\'\'\n(PKG / "{name}.py").write_text(module_source.lstrip("\\n"), encoding="utf-8")\n').splitlines(True)
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11.15'}},'cells':cells}
    for i,c in enumerate(cells):c['id']=f'cell-{i:03d}'
    path=OUT/'canvas'/f'Short notebook {number}.ipynb';path.write_text(json.dumps(nb,indent=1))
    for i,c in enumerate(cells):
        if c['cell_type']=='code':compile(''.join(c['source']),f'{path.name} cell {i}','exec')
    notebook_records.append({'path':str(path.relative_to(OUT)), 'catalogue_number':catalogue,
        'historical_score':score, 'historical_csv_sha256':sha, 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'code_cells':sum(c['cell_type']=='code' for c in cells), 'syntax_checked':True,
        'full_raw_training_executed':False, 'module_count':len(notebook_modules)})
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
(OUT/'authoring/runtime_sections.json').write_text(json.dumps(dict(setup=setup,imports=imports,lp=lp_code,features=features,train=traincode,pool=poolcode,legacy=legacy_code,export=export,module_sources=module_sources,legacy_sources=legacy_sources,legacy_configs=legacy_configs,legacy_columns=legacy_columns)))
(OUT/'evidence/notebook_build.json').write_text(json.dumps({
    'source_revision':SOURCE_REVISION, 'source_sha256':source_hashes, 'component_manifest_sha256':manifest_hashes,
    'notebooks':notebook_records, 'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'legacy_source_adaptations':['Package imports renamed to sintef_project.legacy_*.',
        'Only the legacy raw-file paths and ROOT are adapted to the supplied project folder.',
        'generator_features and training_rows are extracted unchanged from round5_lightgbm.py.'],
    'legacy_csv_policy':'Default pandas to_csv then default pd.read_csv before float64 mean; final output %.9g.',
    'documentation_only_corrections':['Embedded Tokke docstring: pair averaging preserves totals up to roundoff; executable projection code unchanged.'],
    'resource_policy':'Sequential fits; max 4 CPU threads; one CUDA device; large tables released before legacy.',
    'remaining_validation':['Independent notebook schema/import and source-equivalence audit.',
        'Both complete raw-input notebooks on the pinned CUDA environment; measure runtime/RAM and compare historical hashes.',
        'No new label scoring or training performed during authoring.'],
}, indent=2))
print('Wrote and syntax-checked both standalone notebooks')
