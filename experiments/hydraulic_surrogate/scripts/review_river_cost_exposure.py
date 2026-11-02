"""Read-only topology and saved pre-2021 dispatch exposure; never solve or read targets."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shlex
import sys

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    """Return the SHA256 of a local evidence file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Describe only the original twelve metadata cases and supplied physical inputs."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    assert Path(args.run_id).name == args.run_id and args.run_id not in ['', '.', '..']
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    selection = ROOT / 'docs/all14_integer_water_budget_diagnostic_protocol.json'
    assert digest(selection) == '14711cf1aa752bcd0932932d59cad48371e1c0c9482d296301e67b2fd0207abc'
    protocol = json.loads(selection.read_text())
    cache = ROOT / protocol['cache']
    cache_manifest = json.loads((cache / 'manifest.json').read_text())
    assert digest(cache / 'manifest.json') == 'e3d9f53ec7ac67a0afcc48c9630a215ad0a2be4abbfc07aa03cdc3a4345d8163'
    topology_path = ROOT / 'sintef_project/data/extended/Tokke_Vinje_topology.yaml'
    model = yaml.safe_load(topology_path.read_text())['model']
    inflow_path = ROOT / 'sintef_project/data/kernel/Historical_inflow_1958_2025.csv'
    volume_path = ROOT / 'sintef_project/data/kernel/Historical_volume_2015_2024.csv'
    columns = ['Hyljelihyl', 'Langeidvatn', 'Staavatn', 'Langesae']
    inflow = pd.read_csv(inflow_path, usecols=['date', *columns])
    inflow.index = pd.to_datetime(inflow.pop('date'), utc=True)
    volume = pd.read_csv(volume_path, usecols=['date', *columns])
    volume.index = pd.to_datetime(volume.pop('date'), utc=True)
    inputs = {str(p.relative_to(ROOT)): digest(p) for p in
              [selection, cache / 'manifest.json', topology_path, inflow_path, volume_path]}
    cases = []
    for case in protocol['cases']:
        cid = case['case_id']
        start = pd.Timestamp(case['date'])
        assert start + pd.Timedelta(days=7) <= pd.Timestamp('2021-01-01', tz='UTC')
        path = cache / f'case_{cid:04d}_result.npz'
        assert digest(path) == cache_manifest['artifacts'][path.name]
        inputs[str(path.relative_to(ROOT))] = digest(path)
        with np.load(path, allow_pickle=False) as saved:
            flow = saved['flow'] / 0.0036  # Mm3 per one-hour interval -> m3/s.
            bypass = saved['bypass'] / 0.0036
            times = pd.date_range(start, periods=168, freq='h')
            natural = inflow.loc[times]
            # Known explicitly routed arrivals: Kjela turbine, Kjelavatn bypass,
            # and Hyljelihyl's own natural inflow. Other pooled bypass routing
            # is ambiguous, so this is a saved-dispatch exposure, not a solution.
            known_arrivals = flow[12] + bypass[9] + natural['Hyljelihyl'].to_numpy()
            cases.append({
                'case_id': cid, 'date': str(start),
                'hyljelihyl_initial_mm3': float(volume.loc[start, 'Hyljelihyl']),
                'hyljelihyl_known_arrival_m3s_mean': float(known_arrivals.mean()),
                'hyljelihyl_known_arrival_m3s_max': float(known_arrivals.max()),
                'hyljelihyl_known_arrival_hours_above_40': int(np.sum(known_arrivals > 40)),
                'hyljelihyl_known_arrival_mm3': float(known_arrivals.sum() * 0.0036),
                'kjela_flow_m3s_max': float(flow[12].max()),
                'vesle_kjela_flow_m3s_max': float(flow[13].max()),
                'haukeli_flow_m3s_max': float(flow[10].max()),
                'upstream_natural_inflow_m3s_max': natural.max().to_dict(),
                'interpretation': 'Known arrivals from the unchanged saved dispatch; '
                                  'not reconstructed river flow, a constraint violation, '
                                  'or evidence that the new optimum must cross a cost breakpoint.',
            })
    curves = {}
    for name in ['r_Staavatn', 'r_Langesae', 'r_Langeidvatn', 'r_Hyljelihyl']:
        river = model['river'][name]
        curve = river['flow_cost_curve']
        x, y = np.asarray(curve['x'], dtype=float), np.asarray(curve['y'], dtype=float)
        assert x[0] == 0 and np.all(np.diff(x) > 0) and np.all(np.diff(y) >= 0)
        curves[name] = {
            'main_river': river['main_river'], 'breakpoints_m3s': x.tolist(),
            'marginal_cost_currency_per_hour_per_m3s': y.tolist(),
            'cost_examples_currency_per_hour': {
                str(q): float(np.sum(np.clip(q - x[:-1], 0, np.diff(x)) * y[:-1])
                              + max(q - x[-1], 0) * y[-1])
                for q in [20, 40, 60, 100, 200, 400, 1000, 1100]},
            'tail': 'Last marginal slope continues beyond 1000; the supplied river has no max_flow_const.',
        }
    report = {
        'status': 'complete_read_only_review', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'command': shlex.join([sys.executable, *sys.argv]), 'labels_decoded': False,
        'optimization_calls': 0, 'training': False, 'automatic_promotion': False,
        'source': {str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))},
        'inputs': inputs, 'cases': cases, 'curves': curves,
        'environment': {'python': sys.version, 'platform': platform.platform(),
                        'numpy': np.__version__, 'pandas': pd.__version__, 'yaml': yaml.__version__},
        'currency_note': 'Official attribute units say NOK; our existing LP assumes supplied '
                         'cost numbers share the EUR input numeraire. The competition command '
                         'sequence is unknown. Preserve that assumption for an isolated ablation; '
                         'do not fit an exchange-rate multiplier.',
        'official_references': ['https://docs.shop.sintef.energy/objects/river/river.html#flow-cost-curve',
                                'https://docs.shop.sintef.energy/objects/river/river.html#main-river'],
    }
    (out / 'source.py').write_bytes(Path(__file__).read_bytes())
    (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'manifest_sha256': digest(out / 'manifest.json'), 'cases': cases}, indent=2))


if __name__ == '__main__':
    main()
