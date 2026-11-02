"""Strict externally audited aggregate loader; original integer feature source stays frozen."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sintef_project.lp_features import UNIT_ORDER
from sintef_project.integer_water_budget_features import COLUMNS,feature_specification,append_preserving_prefix

COMPLETION_SHA='7da83d48119d48a0826a6d3edf01a265ee932edb8ec7f899be6dc2bb9dce8984'
EXCEPTION_AUDIT_SHA='2708cafb0b2ed2ae7f0b43ee1515d63dc80ebcd3908109277a2a7977fde66b02'


def digest(path: Path) -> str:
    """Hash bytes without exposing file contents."""
    result=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1<<20),b''):result.update(block)
    return result.hexdigest()


def validate_aggregate_manifest(manifest: dict,protocol: dict) -> None:
    """Require all cases and the sole explicitly accepted numerical exception."""
    m,p=manifest,protocol
    if (p['version']!='integer_water_budget_aggregate_v1' or m['status']!='complete'
        or not m.get('all_cases_accepted_and_replayed') or m.get('all_cases_native_strict_gate_passed')is not False
        or m.get('original_cache_still_ineligible')is not True or m.get('failures')):
        raise ValueError('Incomplete or incorrectly relabelled aggregate')
    if len(p['case_ids'])!=3647 or len(set(p['case_ids']))!=3647 or m['case_ids']!=p['case_ids'] or m['case_dates']!=p['case_dates']:
        raise ValueError('Full case metadata required')
    if set(m['completed_cases'])!={str(cid)for cid in p['case_ids']}:raise ValueError('Missing aggregate receipts')
    groups=[m['ordinary_case_ids'],m['imported_case_ids'],m['exception_case_ids']]
    expected=[3010,636,1]
    if any(len(g)!=n or len(set(g))!=n for g,n in zip(groups,expected)):
        raise ValueError('Incorrect disjoint provenance counts')
    if len(set().union(*map(set,groups)))!=3647 or set().union(*map(set,groups))!=set(p['case_ids']):
        raise ValueError('Overlapping or missing provenance groups')
    if (groups[1]!=p['imported_case_ids'] or groups[2]!=p['exception_case_ids'] or groups[2]!=[2012]
        or [m['ordinary_case_count'],m['imported_case_count'],m['exception_count']]!=expected):
        raise ValueError('Unapproved exception or import population')
    for field in ['source','data','environment','feature_specification','prefix_columns']:
        if m[field]!=p[field]:raise ValueError('Aggregate recipe changed: '+field)
    if m['feature_specification']!=feature_specification():raise ValueError('Frozen84 formulas changed')
    if (m['completion_manifest_sha256']!=p['completion_manifest_sha256'] or m['completion_manifest_sha256']!=COMPLETION_SHA
        or m['exception_actual_audit_sha256']!=p['exception_actual_audit_sha256'] or m['exception_actual_audit_sha256']!=EXCEPTION_AUDIT_SHA):
        raise ValueError('Accepted exception evidence changed')


def add_integer_water_features(prefix: pd.DataFrame,root: Path,cache_dir: Path,
                               manifest_sha256: str,protocol_sha256: str,*,
                               audit_path: Path|None=None,audit_sha256: str|None=None) -> tuple[pd.DataFrame,dict]:
    """Append84 only after externally pinned full aggregate physical acceptance."""
    if audit_path is None or audit_sha256 is None:raise ValueError('Explicit independent aggregate audit required')
    root=Path(root).resolve()
    if root.name=='sintef_project':root=root.parent
    cache=Path(cache_dir);cache=cache if cache.is_absolute()else root/cache
    audit_path=Path(audit_path);audit_path=audit_path if audit_path.is_absolute()else root/audit_path
    if digest(cache/'manifest.json')!=manifest_sha256 or digest(cache/'protocol.json')!=protocol_sha256 or digest(audit_path)!=audit_sha256:
        raise ValueError('Pinned aggregate/protocol/audit changed')
    m=json.loads((cache/'manifest.json').read_text());p=json.loads((cache/'protocol.json').read_text());a=json.loads(audit_path.read_text())
    validate_aggregate_manifest(m,p)
    if m['protocol_sha256']!=protocol_sha256:raise ValueError('Protocol identity differs')
    if (a['status']!='passed' or a['cache_manifest_sha256']!=manifest_sha256 or a['cache_protocol_sha256']!=protocol_sha256
        or a['features_sha256']!=m['features_sha256'] or a['case_count']!=3647
        or not a.get('all_cases_physical_and_replay_verified') or not a.get('compatible_receipts_verified')
        or a['compatible_case_count']!=636 or a['exception_case_ids']!=[2012] or a['exception_count']!=1
        or a['ordinary_case_count']!=3010 or a['imported_case_count']!=636
        or a['completion_manifest_sha256']!=COMPLETION_SHA or a['exception_actual_audit_sha256']!=EXCEPTION_AUDIT_SHA):
        raise ValueError('Full independent aggregate audit differs')
    for group in ['source','data']:
        for name,h in p[group].items():
            if digest(root/name)!=h:raise ValueError('Frozen source/data changed: '+name)
    for name,hash_name in [('prefix_protocol','prefix_protocol_sha256'),('original_protocol','original_protocol_sha256'),
                           ('original_base_protocol','original_base_protocol_sha256')]:
        if digest(root/p[name])!=p[hash_name]:raise ValueError('Preserved physical recipe changed')
    if digest(root/p['continuous_cache']/'manifest.json')!=p['continuous_cache_manifest_sha256']:
        raise ValueError('Continuous control changed')
    path=cache/'features.npz'
    if digest(path)!=m['features_sha256'] or m['artifacts']['features.npz']!=m['features_sha256']:raise ValueError('Feature bytes changed')
    project=root/'sintef_project'
    train=pd.read_csv(project/'data/kernel/Unit_commitment_decisions.csv',usecols=['Run No','starttime'])
    cases=pd.concat([train,pd.read_csv(project/'prediction_mapping.csv')],ignore_index=True)
    if (cases['Run No'].tolist()!=m['case_ids'] or pd.to_datetime(cases.starttime,utc=True).astype(str).tolist()!=m['case_dates']
        or len(prefix)!=3647*168 or not np.array_equal(prefix['horizon_hour'].to_numpy(),np.tile(np.arange(168),3647))):
        raise ValueError('Case/date/hour alignment differs')
    with np.load(path,allow_pickle=False)as z:
        if (z['case_ids'].tolist()!=m['case_ids'] or z['columns'].tolist()!=COLUMNS or z['units'].tolist()!=UNIT_ORDER
            or z['features'].shape!=(3647,168,84) or z['features'].dtype!=np.float32):raise ValueError('Feature schema differs')
        values=z['features'].reshape(len(prefix),84).copy()
    result=append_preserving_prefix(prefix,values,p['prefix_columns'])
    for name in prefix:
        if prefix[name].dtype!=result[name].dtype or prefix[name].to_numpy().tobytes()!=result[name].to_numpy().tobytes():
            raise ValueError('Frozen664 prefix bytes changed')
    return result,{'manifest_sha256':manifest_sha256,'protocol_sha256':protocol_sha256,'audit_sha256':audit_sha256,
        'features_sha256':m['features_sha256'],'columns':COLUMNS,'shape':list(values.shape),'feature_specification':feature_specification(),
        'ordinary_case_count':3010,'imported_case_count':636,'exception_count':1,'exception_case_ids':[2012],
        'completion_manifest_sha256':COMPLETION_SHA,'exception_actual_audit_sha256':EXCEPTION_AUDIT_SHA}
