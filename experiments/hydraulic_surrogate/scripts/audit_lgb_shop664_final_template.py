"""Audit unsealed conditional LGB664 final templates without features or targets."""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_pooled_pairwise_residual_result import project


def sha(path: Path) -> str:
    """Return an exact file digest."""
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main() -> None:
    """Check immutable prerequisite bytes and the reviewed conditional recipe."""
    out = ROOT/'artifacts/lgb-shop664-final-template-independent-audit-001'
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    report = {'status':'initializing', 'timestamp_utc':datetime.now(timezone.utc).isoformat(),
              'training':False, 'targets_decoded':False, 'full_feature_build':False,
              'environment':{'python':sys.version, 'platform':platform.platform(),
                             'numpy':np.__version__, 'pandas':pd.__version__}}
    try:
        protocol = ROOT/'docs/lgb_shop664_protocol.json'
        assert sha(protocol) == '882e8c3124ba834322c8f062a7544e9ca3e9512b385e8830db387c38f7c97d08'
        p = json.loads(protocol.read_text())
        for section in ['source','inputs','data']:
            for name, expected in p[section].items():
                assert sha(ROOT/name) == expected, name
        names = ['scripts/train_lgb_shop664_final.py', 'scripts/export_lgb_shop664_final.py',
                 'scripts/prepare_lgb_shop664_final.py', 'tests/test_lgb_shop664_final.py',
                 'docs/lgb_shop664_final.md']
        report['reviewed_source'] = {name:sha(ROOT/name) for name in names}
        for name in names:
            if name.endswith('.py'):
                compile((ROOT/name).read_text(), name, 'exec')
            dest = out/'source'/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name, dest)
        first = ROOT/'artifacts/lgb-shop664-2021-001/manifest.json'
        proof = ROOT/'artifacts/lgb-shop664-2021-independent-audit-001/manifest.json'
        assert sha(first) == '88faf9b155271b5ab46488031b9510d5aecbb6a9bbd1abe0c19554808b5bf413'
        assert sha(proof) == '31e11af7408082765541ba8cf45d7c4c7bcfc2884d80a664a7d5969961d0e005'
        assert json.loads(proof.read_text())['eligible_for_2022']
        original = ROOT/'artifacts/lgb-normalized-lp-context-joint-final-001/manifest.json'
        m = json.loads(original.read_text())
        assert m['status'] == 'complete' and m['arguments']['years'] == []
        params = dict(m['model_parameters']); params['n_jobs'] = 2
        assert params == p['model_parameters']
        sample = original.parent/'train_sample_final.npy'
        expected = np.sort(np.random.default_rng(4173).choice(2922*168, int(2922*168*.4), replace=False))
        buffer = io.BytesIO(); np.save(buffer, expected, allow_pickle=False)
        expected_sha = hashlib.sha256(buffer.getvalue()).hexdigest()
        assert len(expected) == 196358 and expected_sha == sha(sample) == m['artifacts'][sample.name]
        anchor_dir = ROOT/'artifacts/submission-cuda-xgb664-anchor-blend-001'
        anchor_manifest = anchor_dir/'manifest.json'
        assert sha(anchor_manifest) == 'dcd1617cb60aa29dce9c9a64bff16fbc25f00ef569912ea1b80c21eea5aa973c'
        anchor_info = json.loads(anchor_manifest.read_text())
        for name, expected_hash in anchor_info['artifacts'].items():
            assert sha(anchor_dir/name) == expected_hash, name
        anchor = np.load(anchor_dir/'test_predictions.npy', allow_pickle=False)
        assert anchor.dtype == np.float64 and anchor.shape == (725*168,14)
        assert np.array_equal(project(anchor,p['units']),anchor)
        audit = ROOT/'artifacts/cuda-dual-final-submissions-independent-audit-001/manifest.json'
        assert sha(audit) == 'ff88843e5eb8aa1d351ceb36b0640afb08dfc82d2490ecba294e3dacf272aa7d'
        assert json.loads(audit.read_text())['status'] == 'passed'
        tests = subprocess.run([sys.executable,'-m','unittest','tests.test_lgb_shop664_final'],
                               cwd=ROOT, text=True, capture_output=True)
        (out/'tests.txt').write_text(tests.stdout+tests.stderr)
        assert tests.returncode == 0
        assert not (ROOT/'docs/lgb_shop664_final_protocol.json').exists()
        report.update(status='passed_template_only', validation_protocol_sha256=sha(protocol),
                      original_full_sample_sha256=expected_sha, sample_rows=len(expected),
                      model_parameters=params, anchor_manifest_sha256=sha(anchor_manifest),
                      tests_passed=3, actual_final_protocol_sealed=False,
                      reviewed_gates=['Actual complete accepted 2021 and 2022 artifacts and independent proofs with the same immutable positive lock are required before protocol creation.',
                                      'The original 196358-row full-data sample and original 350-tree parameters are preserved; only approved two-thread execution and 664 input schema differ.',
                                      'All 14 checkpoints replay all 121800 test hours in float64 before float32 raw storage; inference mode reloads exact original checkpoints without fitting.',
                                      'The export requires a separate final-manifest-bound all14 replay proof and preserves the already-projected accepted SHOP anchor before one locked-weight blend.',
                                      'Exact test IDs/hour-unit layout, finite probabilities, CSV byte hash and readback error are checked; no remote submission or diagnostic is introduced.'],
                      limitations='This approves unsealed source templates only. Actual accepted 2022 evidence, final protocol byte audit, final fit/checkpoint replay and independent actual CSV audit remain required.')
    except BaseException:
        report.update(status='failed',error=traceback.format_exc())
        raise
    finally:
        shutil.copy2(__file__,out/Path(__file__).name)
        report.update(runtime_seconds=time.monotonic()-start,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      source_sha256=sha(Path(__file__)))
        report['artifacts']={f.relative_to(out).as_posix():sha(f) for f in out.rglob('*') if f.is_file()}
        (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'status':report['status'],'manifest_sha256':sha(out/'manifest.json')}),flush=True)


if __name__ == '__main__':
    main()
