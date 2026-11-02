"""62 input-only C-minus-A features; strict complete-cache 664-to-726 loader."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sintef_project.lp_features import Network, UNIT_ORDER

KINDS=['state','delta_qmin_ratio','delta_power_fraction','delta_economic_margin']
SHARED=['hyljelihyl_fill','remaining_vinje_fill','main_flow_40','hyljelihyl_bypass_40',
        'remaining_bypass_40','river_shadow_wedge']
COLUMNS=[f'river_c_{kind}_{unit}' for unit in UNIT_ORDER for kind in KINDS]+[f'river_c_{k}' for k in SHARED]


def specification() -> dict:
    """Freeze formulas and each arm's own topology before supervised use."""
    return {'version':'hyljelihyl_river_62_v1','columns':COLUMNS,'dtype':'float32','prefix_columns':664,
        'unit_fields':['clip(C.state,0,1)','(C.Q-A.Q)/initial_Qmin','(C.P-A.P)/initial_Pmax',
            '(A_local_dual-C_local_dual)/(initial_first_segment_energy*(1+mean(abs(price))))'],
        'local_dual':'lambda_up-lambda_down using each arm own unit_up/unit_down; sea dual zero',
        'shared_fields':['C.V_Hyljelihyl/0.26','C.V_remaining_Vinje/307.00',
            'C.main_volume/(0.0036*40)','C.Hyljelihyl_bypass_volume/(0.0036*40)',
            'C.remaining_bypass_volume/(0.0036*40)',
            '(C.lambda_remaining-C.lambda_Hyljelihyl)/(initial_first_segment_energy_Vinje_G1*(1+mean(abs(price))))'],
        'eligibility':'Complete physically optimal C results; exact independent-worker replay; full-cache independent audit required',
        'prefix':'All664 existing names,dtypes,values and bytes retained; append62 only'}


def compute_features(case: dict, split: dict, result: dict, reference: dict,
                     original_network: Network, split_network: Network) -> np.ndarray:
    """Compute168x62 only from proven physical inputs/results, never targets."""
    if original_network.units!=UNIT_ORDER or split_network.units!=UNIT_ORDER:
        raise ValueError('Generator order changed')
    if len(original_network.pools)!=9 or split_network.pools!=[*original_network.pools,'Hyljelihyl']:
        raise ValueError('Unexpected split topology')
    expected_down=original_network.unit_down.copy();expected_down[12]=9
    if (original_network.unit_down[12]!=4 or not np.array_equal(original_network.unit_up,split_network.unit_up)
        or not np.array_equal(expected_down,split_network.unit_down)):raise ValueError('Unit routing changed')
    if not np.isclose(split['capacity'][9],.26,rtol=0,atol=1e-12) or not np.isclose(split['capacity'][4],307.,rtol=0,atol=1e-12):
        raise ValueError('Split capacities changed')
    for field,shape in [('price',(168,)),('qmin',(14,)),('pmax',(14,))]:
        value=np.asarray(case[field])
        if value.shape!=shape or not np.isfinite(value).all():raise ValueError(field)
    if np.shape(case['energy'])!=(14,4) or not np.isfinite(case['energy']).all():raise ValueError('energy')
    for physical,network in [(result,split_network),(reference,original_network)]:
        for field,shape in [('state',(14,168)),('flow',(14,168)),('power',(14,168)),('water_shadow',(len(network.pools),168))]:
            value=np.asarray(physical[field])
            if value.shape!=shape or not np.isfinite(value).all():raise ValueError(field)
        if int(physical['status'])!=0:raise ValueError('Non-optimal physical output')
    for field,shape in [('storage',(10,168)),('bypass',(12,168)),('river_volume',(168,))]:
        value=np.asarray(result[field])
        if value.shape!=shape or not np.isfinite(value).all():raise ValueError(field)
    if result['river_volume'].tobytes()!=result['bypass'][11].tobytes():raise ValueError('River flow differs from physical edge')
    for field in ['minimum_slack','maximum_slack','bypass_slack','vest_slack']:
        if not np.isfinite(result[field]).all() or np.max(np.abs(result[field]))!=0:raise ValueError('Physical slack')
    state=result['state']
    if state.min() < -1e-5 or state.max() > 1+1e-5:raise ValueError('State tolerance')
    scale=1+np.abs(case['price']).mean();cols=[]
    for j in range(14):
        old=reference['water_shadow'][original_network.unit_up[j]].copy()
        new=result['water_shadow'][split_network.unit_up[j]].copy()
        old_down,new_down=original_network.unit_down[j],split_network.unit_down[j]
        if old_down>=0:old-=reference['water_shadow'][old_down]
        if new_down>=0:new-=result['water_shadow'][new_down]
        if min(case['qmin'][j],case['pmax'][j],case['energy'][j,0])<=0:raise ValueError('Conversion denominator')
        cols.extend([np.clip(state[j],0,1),(result['flow'][j]-reference['flow'][j])/case['qmin'][j],
            (result['power'][j]-reference['power'][j])/case['pmax'][j],(old-new)/(case['energy'][j,0]*scale)])
    cols.extend([result['storage'][9]/.26,result['storage'][4]/307.,result['river_volume']/(.0036*40),
        result['bypass'][10]/(.0036*40),result['bypass'][4]/(.0036*40),
        (result['water_shadow'][4]-result['water_shadow'][9])/(case['energy'][7,0]*scale)])
    values=np.column_stack(cols).astype(np.float32)
    if values.shape!=(168,62) or not np.isfinite(values).all():raise ValueError('Feature shape/finite')
    return values


def append_preserving_prefix(prefix: pd.DataFrame, values: np.ndarray, expected_columns: list[str]) -> pd.DataFrame:
    """Append62 with literal dtype/byte preservation of all664 original columns."""
    if len(expected_columns)!=664 or prefix.columns.tolist()!=expected_columns:raise ValueError('Exact664 prefix required')
    if values.shape!=(len(prefix),62) or values.dtype!=np.float32 or not np.isfinite(values).all():raise ValueError('Feature rows')
    if set(COLUMNS).intersection(prefix.columns):raise ValueError('Feature collision')
    result=pd.concat([prefix,pd.DataFrame(values,columns=COLUMNS,index=prefix.index)],axis=1)
    for name in expected_columns:
        if result[name].dtype!=prefix[name].dtype or result[name].to_numpy().tobytes()!=prefix[name].to_numpy().tobytes():
            raise ValueError('Prefix bytes changed')
    return result


def add_river_cost_features(prefix: pd.DataFrame, root: Path, cache_dir: Path,
                           manifest_sha256: str, protocol_sha256: str,
                           audit_path: Path, audit_sha256: str) -> tuple[pd.DataFrame,dict]:
    """Reject partial/unproven caches; compact loading depends on pinned full physical audit."""
    def sha(path):
        value=hashlib.sha256()
        with Path(path).open('rb')as f:
            for block in iter(lambda:f.read(1<<20),b''):value.update(block)
        return value.hexdigest()
    root=Path(root).resolve()
    if root.name=='sintef_project':root=root.parent
    cache=Path(cache_dir);cache=cache if cache.is_absolute()else root/cache
    audit_path=Path(audit_path);audit_path=audit_path if audit_path.is_absolute()else root/audit_path
    if sha(cache/'manifest.json')!=manifest_sha256 or sha(cache/'protocol.json')!=protocol_sha256 or sha(audit_path)!=audit_sha256:
        raise ValueError('Externally pinned cache/protocol/audit changed')
    m=json.loads((cache/'manifest.json').read_text());p=json.loads((cache/'protocol.json').read_text());a=json.loads(audit_path.read_text())
    if (m['status']!='complete' or m.get('failures') or not m.get('all_cases_physical_and_replay_verified')
        or m['protocol_sha256']!=protocol_sha256 or m['case_ids']!=p['case_ids'] or len(m['case_ids'])!=3647
        or set(m['completed_cases'])!={str(i)for i in m['case_ids']}):raise ValueError('Incomplete or unverified cache')
    if (a['status']!='passed' or a['cache_manifest_sha256']!=manifest_sha256 or a['cache_protocol_sha256']!=protocol_sha256
        or a['features_sha256']!=m['features_sha256'] or a['case_count']!=3647
        or not a.get('all_cases_physical_and_replay_verified') or not a.get('canonical_reuse_verified')):
        raise ValueError('Independent full-cache proof missing')
    if (m['feature_specification']!=specification() or p['feature_specification']!=specification()
        or any(m[k]!=p[k]for k in ['source','data','prefix_columns','replay_execution_type'])):raise ValueError('Formula/lineage mismatch')
    for group in ['source','data']:
        for name,h in p[group].items():
            if sha(root/name)!=h:raise ValueError('Source/data changed: '+name)
    for pathkey,hashkey in [('prefix_protocol','prefix_protocol_sha256'),('continuous_manifest','continuous_manifest_sha256')]:
        if sha(root/p[pathkey])!=p[hashkey]:raise ValueError('Prefix/control changed')
    feature_path=cache/'features.npz'
    if sha(feature_path)!=m['features_sha256'] or m['artifacts']['features.npz']!=m['features_sha256']:raise ValueError('Feature bytes changed')
    project=root/'sintef_project'
    train=pd.read_csv(project/'data/kernel/Unit_commitment_decisions.csv',usecols=['Run No','starttime'])
    cases=pd.concat([train,pd.read_csv(project/'prediction_mapping.csv')],ignore_index=True)
    if (cases['Run No'].tolist()!=m['case_ids'] or pd.to_datetime(cases.starttime,utc=True).astype(str).tolist()!=m['case_dates']
        or len(prefix)!=3647*168 or not np.array_equal(prefix['horizon_hour'].to_numpy(),np.tile(np.arange(168),3647))):
        raise ValueError('Case/date/hour alignment')
    with np.load(feature_path,allow_pickle=False)as z:
        if (z['features'].shape!=(3647,168,62) or z['features'].dtype!=np.float32 or z['case_ids'].tolist()!=m['case_ids']
            or z['columns'].tolist()!=COLUMNS or z['units'].tolist()!=UNIT_ORDER):raise ValueError('Feature schema')
        values=z['features'].reshape(len(prefix),62).copy()
    result=append_preserving_prefix(prefix,values,p['prefix_columns'])
    return result,{'manifest_sha256':manifest_sha256,'protocol_sha256':protocol_sha256,'audit_sha256':audit_sha256,
        'features_sha256':m['features_sha256'],'shape':list(values.shape),'columns':COLUMNS,'specification':specification(),
        'replay_execution_type':m['replay_execution_type']}
