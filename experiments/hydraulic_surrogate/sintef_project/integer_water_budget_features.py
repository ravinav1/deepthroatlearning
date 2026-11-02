"""Proposed 84-column input-only extension of the frozen 664 feature prefix."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

from sintef_project.lp_features import Network, UNIT_ORDER

TARGETS = ['Hogga_G1', 'Lio_G1', 'Haukeli_G1']
KINDS = ['state', 'delta_qmin_ratio', 'delta_power_fraction', 'start', 'stop', 'conditional_margin_change']
COLUMNS = [f'integer_water_{kind}_{unit}' for unit in UNIT_ORDER for kind in KINDS]


def feature_specification() -> dict:
    """Return every formula before any training or full-cache construction."""
    return {'version': 'integer_water_budget_84_v1', 'columns': COLUMNS, 'units': UNIT_ORDER,
            'dtype': 'float32', 'prefix_columns': 664, 'binary_units': TARGETS,
            'state': 'Raw unit state, clipped to [0,1] only within numerical tolerance; exactly three selected units are binary.',
            'delta_qmin_ratio': '(integer flow - continuous SHOP flow) / initial Qmin',
            'delta_power_fraction': '(integer power - continuous SHOP power) / initial case-feasible Pmax',
            'start': 'Saved start variable for the transition into t=1..167; t=0 is zero, with no charged boundary transition.',
            'stop': 'Saved stop variable for the transition into t=1..167; t=0 is zero.',
            'conditional_margin_change': '(continuous SHOP local dual - fixed-integer-path local dual) / (initial first-segment energy * (1 + mean(abs(price))))',
            'local_dual': 'lambda_up - lambda_down; the downstream sea dual is zero. Integer-path duals are conditional LP duals, not global MILP shadow values.',
            'eligibility': 'Only proven, physically feasible integer results with bitwise conditional LP replays; complete cache loader additionally requires fresh MILP replays.'}


def compute_features(case: dict, result: dict, reference: dict,
                     network: Network, proof: dict) -> np.ndarray:
    """Compute 168 × 84 columns using only pinned physical inputs and outputs."""
    if network.units != UNIT_ORDER:
        raise ValueError('Unexpected generator ordering')
    if np.shape(case['price'])!=(168,) or not np.isfinite(case['price']).all():
        raise ValueError('Expected one finite 168-hour price vector')
    for physical in [result,reference]:
        if np.shape(physical['water_shadow'])!=(len(network.pools),168) or not np.isfinite(physical['water_shadow']).all():
            raise ValueError('Expected finite reservoir-by-hour conditional duals')
    if not (proof.get('proven_optimal') and proof.get('mip_status') == 0
            and proof.get('fixed_lp_replay_bitwise')):
        raise ValueError('Unproven or unreplayed integer result is ineligible')
    for source, names in [(result, ['state', 'flow', 'power']), (reference, ['flow', 'power'])]:
        for name in names:
            if np.shape(source[name]) != (14, 168) or not np.isfinite(source[name]).all():
                raise ValueError(f'Invalid physical array: {name}')
    state = np.asarray(result['state'])
    if state.min() < -1e-5 or state.max() > 1+1e-5:
        raise ValueError('State exceeds physical tolerance')
    chosen = state[[UNIT_ORDER.index(u) for u in TARGETS]]
    if np.max(np.abs(chosen-np.rint(chosen))) > 1e-5:
        raise ValueError('Selected unit states are not binary')
    for name in ['raw_start', 'raw_stop']:
        values = np.asarray(result[name])
        if values.shape != (14, 167) or not np.isfinite(values).all() or values.min() < -1e-5 or values.max() > 1+1e-5:
            raise ValueError('Invalid transition variable')
    for name in ['minimum_slack','maximum_slack','bypass_slack','vest_slack']:
        if not np.isfinite(result[name]).all() or np.max(np.abs(result[name])) != 0:
            raise ValueError('Nonzero or invalid physical slack')
    scale = 1+np.abs(case['price']).mean()
    columns = []
    for j, _ in enumerate(UNIT_ORDER):
        up, down = network.unit_up[j], network.unit_down[j]
        new_dual = np.asarray(result['water_shadow'])[up].copy()
        old_dual = np.asarray(reference['water_shadow'])[up].copy()
        if down >= 0:
            new_dual -= result['water_shadow'][down]
            old_dual -= reference['water_shadow'][down]
        if min(case['qmin'][j],case['pmax'][j],case['energy'][j,0],scale) <= 0:
            raise ValueError('Nonpositive unit conversion')
        columns.extend([np.clip(state[j],0,1),
                        (result['flow'][j]-reference['flow'][j])/case['qmin'][j],
                        (result['power'][j]-reference['power'][j])/case['pmax'][j],
                        np.r_[0.,result['raw_start'][j]],
                        np.r_[0.,result['raw_stop'][j]],
                        (old_dual-new_dual)/(case['energy'][j,0]*scale)])
    values = np.column_stack(columns).astype(np.float32)
    if values.shape != (168,84) or not np.isfinite(values).all():
        raise ValueError('Invalid integer water-budget features')
    return values


def append_preserving_prefix(prefix: pd.DataFrame, values: np.ndarray,
                             expected_columns: list[str]) -> pd.DataFrame:
    """Append aligned numeric columns without changing any existing value or dtype."""
    if len(expected_columns) != 664 or prefix.columns.tolist() != expected_columns:
        raise ValueError('The exact frozen 664-column prefix is required')
    if values.shape != (len(prefix),84) or values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError('Expected aligned finite float32 integer feature rows')
    if set(COLUMNS).intersection(prefix.columns):
        raise ValueError('Integer columns already present')
    additions = pd.DataFrame(values,columns=COLUMNS,index=prefix.index)
    result = pd.concat([prefix,additions],axis=1)
    for name in expected_columns:
        if result[name].dtype != prefix[name].dtype or not np.array_equal(result[name].to_numpy(),prefix[name].to_numpy()):
            raise ValueError('Appending physical features changed the prefix')
    return result


def add_integer_water_features(prefix: pd.DataFrame, root: Path, cache_dir: Path,
                               manifest_sha256: str, protocol_sha256: str) -> tuple[pd.DataFrame, dict]:
    """Load only an externally pinned, complete cache into the unchanged prefix.

    The independent cache audit and model protocol pin the full manifest; GPU
    packaging may carry its compact features and proof index without raw solver
    arrays. The full physical payload remains at its audited original location.
    """
    def sha(path: Path) -> str:
        value=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1<<20),b''):
                value.update(block)
        return value.hexdigest()
    repository=Path(root).resolve()
    if repository.name=='sintef_project':
        repository=repository.parent
    cache=Path(cache_dir)
    if not cache.is_absolute():
        cache=repository/cache
    if sha(cache/'manifest.json')!=manifest_sha256 or sha(cache/'protocol.json')!=protocol_sha256:
        raise ValueError('Explicitly pinned integer cache identity changed')
    m=json.loads((cache/'manifest.json').read_text())
    p=json.loads((cache/'protocol.json').read_text())
    if (m['status']!='complete' or not m.get('all_cases_proven_and_replayed') or m.get('failures')
            or m['protocol_sha256']!=protocol_sha256 or m['case_ids']!=p['case_ids']
            or len(m['case_ids'])!=3647 or set(m['completed_cases'])!={str(i) for i in m['case_ids']}):
        raise ValueError('An incomplete or unproven integer cache is ineligible')
    if (m['feature_specification']!=feature_specification() or p['feature_specification']!=feature_specification()
            or m['source']!=p['source'] or m['data']!=p['data'] or m['prefix_columns']!=p['prefix_columns']):
        raise ValueError('Cache formulas or lineage changed')
    for group in ['source','data']:
        for name,expected in p[group].items():
            if sha(repository/name)!=expected:
                raise ValueError(f'Integer cache {group} changed: {name}')
    if sha(repository/p['prefix_protocol'])!=p['prefix_protocol_sha256']:
        raise ValueError('The existing 664 prefix protocol changed')
    if sha(repository/p['continuous_cache']/'manifest.json')!=p['continuous_cache_manifest_sha256']:
        raise ValueError('The continuous physical control changed')
    feature_path=cache/'features.npz'
    if sha(feature_path)!=m['features_sha256'] or m['artifacts']['features.npz']!=m['features_sha256']:
        raise ValueError('Integer feature bytes changed')
    project=repository/'sintef_project'
    train=pd.read_csv(project/'data/kernel/Unit_commitment_decisions.csv',usecols=['Run No','starttime'])
    cases=pd.concat([train,pd.read_csv(project/'prediction_mapping.csv')],ignore_index=True)
    if (cases['Run No'].tolist()!=m['case_ids'] or pd.to_datetime(cases.starttime,utc=True).astype(str).tolist()!=m['case_dates']
            or len(prefix)!=len(cases)*168 or not np.array_equal(prefix['horizon_hour'].to_numpy(),np.tile(np.arange(168),len(cases)))):
        raise ValueError('Integer cache case/date/hour alignment differs')
    with np.load(feature_path,allow_pickle=False) as z:
        if (z['case_ids'].tolist()!=m['case_ids'] or z['columns'].tolist()!=COLUMNS or z['units'].tolist()!=UNIT_ORDER
                or z['features'].shape!=(3647,168,84) or z['features'].dtype!=np.float32):
            raise ValueError('Integer cache row/column schema differs')
        values=z['features'].reshape(len(prefix),84).copy()
    result=append_preserving_prefix(prefix,values,p['prefix_columns'])
    return result,{'manifest_sha256':manifest_sha256,'protocol_sha256':protocol_sha256,
                  'features_sha256':m['features_sha256'],'shape':list(values.shape),'columns':COLUMNS,
                  'feature_specification':feature_specification(),'proof':'All 3647 cases proven with exact fresh MILP and conditional LP replays; manifest externally pinned.'}
