"""Seal the prospective LGB726 test only after full physical-cache acceptance."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_lgb_river726 as runner
from scripts.train_lgb_shop664 import sha


def relative(path: Path) -> str:
    """Require a real input within the project and store its portable path."""
    return path.resolve().relative_to(ROOT).as_posix()


def main() -> None:
    """Freeze actual full-cache proof and original controls without decoding labels."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--cache-manifest-sha256', required=True)
    parser.add_argument('--cache-protocol-sha256', required=True)
    parser.add_argument('--cache-audit', type=Path, required=True)
    parser.add_argument('--cache-audit-sha256', required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/lgb_river726_protocol.json')
    args = parser.parse_args()
    record = {'directory': relative(args.cache), 'manifest_sha256': args.cache_manifest_sha256,
              'protocol_sha256': args.cache_protocol_sha256, 'audit': relative(args.cache_audit),
              'audit_sha256': args.cache_audit_sha256}
    manifest, physical = runner.check_cache_contract(record)
    assert sha(ROOT / runner.BASE_PROTOCOL) == runner.BASE_SHA
    original = json.loads((ROOT / runner.BASE_PROTOCOL).read_text())
    sources = ['scripts/train_lgb_river726.py', 'scripts/prepare_lgb_river726.py',
               'sintef_project/hyljelihyl_river_features.py', 'tests/test_lgb_river726.py',
               'docs/river726_learned_test_design.md']
    inputs = {runner.BASE_PROTOCOL: runner.BASE_SHA,
              record['directory'] + '/manifest.json': record['manifest_sha256'],
              record['directory'] + '/protocol.json': record['protocol_sha256'],
              record['audit']: record['audit_sha256']}
    controls = {}
    for year in ['2021', '2022']:
        directory = ROOT / f'artifacts/lgb-shop664-{year}-001'
        mp, prediction = directory / 'manifest.json', directory / f'validation_{year}.npz'
        m = json.loads(mp.read_text())
        assert m['status'] == 'complete' and m['protocol_sha256'] == runner.BASE_SHA
        assert m['artifacts'][prediction.name] == sha(prediction)
        controls[year] = {'manifest': relative(mp), 'manifest_sha256': sha(mp),
                          'prediction': relative(prediction), 'prediction_sha256': sha(prediction)}
        audit = ROOT / f'artifacts/lgb-shop664-{year}-independent-audit-001/manifest.json'
        controls[year].update(audit=relative(audit), audit_sha256=sha(audit))
        runner.check_control(controls[year], year)
        inputs[relative(mp)] = sha(mp)
        inputs[relative(prediction)] = sha(prediction)
        inputs[relative(audit)] = sha(audit)
    protocol = {'version': runner.VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'source': {name: sha(ROOT / name) for name in sources}, 'inputs': inputs,
        'base_protocol': runner.BASE_PROTOCOL, 'base_protocol_sha256': runner.BASE_SHA,
        'environment': runner.base.environment(), 'cache': record,
        'cache_feature_specification': manifest['feature_specification'],
        'feature_columns': original['feature_columns'] + physical['feature_specification']['columns'],
        'units': original['units'], 'anchor': original['anchor'], 'control': controls,
        'samples': original['samples'], 'folds': original['folds'],
        'model_parameters': original['model_parameters'], 'candidate_weights': runner.base.WEIGHTS,
        'early_stopping': False, 'final_training': False, 'reused_development_years': True,
        'formula': 'project_vinje((1-w)*saved_projected_public_bestSHOP664+w*rawLGB726),float64',
        'gate': 'Maximum2021 AUC with lower-weight ties; positive winner must strictly exceed identity and matched rawLGB664 at the same weight. Independently audit lock before2022; evaluate only identity and locked weight, requiring the same two strict gains.',
        'resources': {'model_cpu_threads': 2, 'minimum_free_percent_before_build': 40,
                      'minimum_free_percent_before_unit': 20, 'maximum_simultaneous_large_local_jobs': 1},
        'hypothesis': 'Supplied convex river cost and separate Hyljelihyl storage improve information beyond the exact664 prefix.'}
    runner.check_closure(protocol)
    with args.output.open('x') as stream:
        json.dump(protocol, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'protocol': str(args.output), 'sha256': sha(args.output), 'training': False,
                      'labels_decoded': False}), flush=True)


if __name__ == '__main__':
    main()
