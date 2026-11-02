"""Independently review frozen canonical115 screen without feature construction."""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROTOCOL = 'docs/canonical_pooled_screen_protocol_v2.json'
PROTOCOL_SHA = 'dc1ed39784e6382b815ec37445ba17220561ddae388c6caa041ba38b5fe51684'
PREFLIGHT = 'artifacts/canonical-pooled-screen-v2-preflight-001/manifest.json'
PREFLIGHT_SHA = '639aa54baa95575857cbda05e866728ef9ae149f9f4ee9b87fa5d9f0b4a46d48'


def sha(path: Path) -> str:
    """Hash one immutable file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Verify metadata, preserved mathematical contract and guarded execution."""
    out = ROOT / 'artifacts/canonical-pooled-screen-v2-independent-preflight-001'
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    report = {'status': 'initializing', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': [sys.executable, *sys.argv], 'protocol_sha256': PROTOCOL_SHA,
              'input_preflight_manifest_sha256': PREFLIGHT_SHA, 'training': False,
              'competition_features_built': False, 'target_values_decoded': False}
    try:
        assert sha(ROOT / PROTOCOL) == PROTOCOL_SHA and sha(ROOT / PREFLIGHT) == PREFLIGHT_SHA
        p = json.loads((ROOT / PROTOCOL).read_text())
        pre = json.loads((ROOT / PREFLIGHT).read_text())
        assert pre['status'] == 'passed_preflight_only' and pre['protocol_sha256'] == PROTOCOL_SHA
        assert pre['labels_decoded'] is False and pre['feature_build'] is False
        for name, expected in pre['artifacts'].items():
            assert sha((ROOT / PREFLIGHT).parent / name) == expected, name
        for group in ['source', 'inputs']:
            for name, expected in p[group].items():
                assert sha(ROOT / name) == expected, name
        old = json.loads((ROOT / p['previous_unrun_protocol']).read_text())
        assert sha(ROOT / p['previous_unrun_protocol']) == p['previous_unrun_protocol_sha256']
        for name, expected in old['source'].items():
            assert sha(ROOT / name) == expected, name
        for key in set(old) - {'source', 'inputs', 'created_utc', 'version', 'resources'}:
            assert p[key] == old[key], key
        assert {k: v for k, v in p['resources'].items() if k != 'checks'} == {
            k: v for k, v in old['resources'].items() if k != 'checks'}
        from scripts import screen_canonical_pooled_v2 as candidate
        from scripts import screen_canonical_pooled as original
        from sintef_project import canonical_unit_features as adapter
        for name in ['early_splits', 'read_screen_labels', 'scores', 'select_candidate', 'predict_replay']:
            assert ast.dump(ast.parse(inspect.getsource(getattr(candidate, name)))) == ast.dump(
                ast.parse(inspect.getsource(getattr(original, name))))
        real_read = candidate.pd.read_csv
        reads = []
        def guarded_read(path, *args, **kwargs):
            if Path(path).name == 'Unit_commitment_decisions.csv':
                assert kwargs.get('usecols') == ['Run No', 'starttime'], 'Target values requested during preflight'
                reads.append(str(path))
            return real_read(path, *args, **kwargs)
        with patch.object(candidate.pd, 'read_csv', side_effect=guarded_read):
            _, _, metadata, (train, screen, sample) = candidate.check_closure(p)
        dates = candidate.pd.to_datetime(metadata.starttime, utc=True)
        rng = candidate.np.random.default_rng(4173)
        for year in [2018, 2019, 2020]:
            end = candidate.pd.Timestamp(f'{year}-01-01', tz='UTC')
            allowed = candidate.np.flatnonzero((dates + candidate.pd.Timedelta(hours=167)) < end)
            rows = (allowed[:, None] * 168 + candidate.np.arange(168)).ravel()
            expected = candidate.np.sort(rng.choice(rows, int(len(rows) * .4), replace=False))
        assert allowed.tolist() == train.tolist() and expected.dtype == sample.dtype
        assert expected.shape == sample.shape and expected.tobytes() == sample.tobytes()
        assert len(train) == 1820 and len(sample) == 122304 and len(screen) == 360
        assert dates.iloc[train[-1]].strftime('%Y-%m-%d') == '2019-12-25'
        assert dates.iloc[screen[0]].strftime('%Y-%m-%d') == '2020-01-01'
        assert dates.iloc[screen[-1]].strftime('%Y-%m-%d') == '2020-12-25'
        assert adapter.feature_specification(p['prefix_columns'], p['units']) == p['canonical_specification']
        adapter_proof = json.loads((ROOT / 'artifacts/canonical-unit-adapter-independent-audit-001/manifest.json').read_text())
        assert adapter_proof['status'] == 'passed' and adapter_proof['all_115_fields_all_14_units_scalar_oracle_bitwise']
        assert adapter_proof['source']['sintef_project/canonical_unit_features.py'] == p['source']['sintef_project/canonical_unit_features.py']
        oof = json.loads((ROOT / p['oof_manifest']).read_text())
        oof_audit = json.loads((ROOT / 'artifacts/full14-early-oof-result-independent-audit-001/manifest.json').read_text())
        assert oof_audit['status'] == 'passed' and oof_audit['oof_manifest_sha256'] == sha(ROOT / p['oof_manifest'])
        assert oof_audit['all_42_checkpoint_replays_exact'] and not oof_audit['screen_2020_labels_read']
        assert oof['artifacts'][Path(p['raw_anchor']).name] == p['inputs'][p['raw_anchor']]
        identity = json.loads((ROOT / 'artifacts/pooled-pairwise-residual-screen-001/manifest.json').read_text())
        assert identity['status'] == 'complete' and identity['oof_manifest_sha256'] == oof_audit['oof_manifest_sha256']
        assert identity['artifacts'][Path(p['projected_anchor']).name] == p['inputs'][p['projected_anchor']]
        source = inspect.getsource(candidate.main)
        assert "build_verified_features(ROOT / original['stage'], 2019, loader_protocol)" in source
        assert source.index("resources_before_prediction_seal") < source.index("truth = read_screen_labels")
        assert source.index("resources_after_scoring") < source.index("report.update(status='complete'")
        command = [sys.executable, '-m', 'unittest', 'tests.test_canonical_pooled_screen_v2', 'tests.test_canonical_unit_features']
        tested = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        (out / 'tests.log').write_text(tested.stdout + tested.stderr)
        assert tested.returncode == 0
        report.update(status='passed', environment=candidate.base.environment(),
                      tests_command=command, tests_exit_code=0, target_csv_metadata_reads=reads,
                      training_cases=1820, sampled_hours=122304, pooled_training_rows=1712256,
                      screen_cases=360, first_half_cases=int((dates.iloc[screen] < candidate.pd.Timestamp('2020-07-01', tz='UTC')).sum()),
                      original_sample_bytes_verified=True, adapter_source_bound_to_scalar_oracle=True,
                      oof_anchor_bound_to_actual_42_checkpoint_audit=True, v1_mathematical_contract_unchanged=True,
                      before_labels_and_after_scoring_resource_checks_verified=True,
                      source_review='From-scratch unweighted canonical115 BCE;14matched350-tree controls and one1400-round pooled trajectory with700/1400prefix checkpoints; no early stopping; full prediction replays sealed before2020 targets; exact fixed grid and lower-round/lower-weight ties; full+half strict anchor/control gates and no rescue.',
                      limitations='Reused2020 development screen. Observed35-minute/8GiB limits are not hard OS caps. Real execution still requires root scheduling behind the current726 fit; no later validation/final/submission authorized.')
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        shutil.copy2(__file__, out / Path(__file__).name)
        if 'p' in locals():
            for name in p['source']:
                dest = out / 'source' / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, dest)
        report.update(runtime_seconds=time.monotonic() - start,
                      artifacts={str(q.relative_to(out)): sha(q) for q in out.rglob('*') if q.is_file()})
        (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
        print(report['status'], sha(out / 'manifest.json'), flush=True)


if __name__ == '__main__':
    main()
