"""Review conditional726 final templates without issuing a final training recipe."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_lgb_shop664 import sha


def main() -> None:
    """Record exact reviewed source, rejection fixtures and acceptance boundaries."""
    start = time.monotonic()
    out = ROOT / 'artifacts/lgb-river726-final-template-independent-audit-001'
    out.mkdir(exist_ok=False)
    sources = ['scripts/train_lgb_river726_final.py', 'scripts/prepare_lgb_river726_final.py',
               'scripts/export_lgb_river726_final.py', 'scripts/audit_lgb_river726_final.py',
               'tests/test_lgb_river726_final.py', 'docs/lgb_river726_final.md']
    prior_path = ROOT / 'docs/lgb_shop664_final_protocol.json'
    assert sha(prior_path) == '5513e67123a968ded3790ec6bcda898d50d3682685fa7f62020ea25108e322ad'
    previous = json.loads(prior_path.read_text())
    for name, expected in previous['source'].items():
        assert sha(ROOT / name) == expected, name
    assert sha(ROOT / 'docs/lgb_river726_protocol.json') == 'c71d6a026f7ca3e802464df4be7b8874ab8592f54d38d85cd97e2b9040762b7c'
    command = [sys.executable, '-m', 'unittest', 'tests.test_lgb_river726_final']
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    (out / 'tests.log').write_text(completed.stdout + completed.stderr)
    for name in sources:
        destination = out / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    shutil.copy2(__file__, out / Path(__file__).name)
    report = {'status': 'passed_template_only' if completed.returncode == 0 else 'failed',
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': [sys.executable, *sys.argv], 'tests_command': command,
              'tests_exit_code': completed.returncode, 'source': {name: sha(ROOT / name) for name in sources},
              'final_protocol_emitted': False, 'features_built': False, 'training': False,
              'validation_protocol_sha256': 'c71d6a026f7ca3e802464df4be7b8874ab8592f54d38d85cd97e2b9040762b7c',
              'review': [
                  'Final recipe requires independently accepted actual2021 and2022 results with the same positive lockedweight and strict anchor/matchedcontrol gains.',
                  'The original196358-row seed4173 final sample and350-tree parameters are preserved; existing accepted projectedSHOP anchor and fixed Vinje projection retain all nested stages.',
                  'Final features use the original664 staged loader then the actual independently audited62 riverfeatures; cache/protocol/audit hashes survive into final and export records.',
                  'Trainer requires exactfinalprotocol independent approval; every14 native350tree726feature checkpoint must replay all121800 test rows in float64 before saving float32 output.',
                  'Independent final auditor rebuilds prefix and extension, checks per-column prefix bytes, native model parameters and two full predictions; CSV path uses separate generic isotonic projection and exact byte reconstruction.',
                  'Tests reject failed2022 acceptance, changed cache and weight; synthetic two-case export checks nested composition, generator/hour mapping and template ID order.'],
              'limitations': 'Template review only. No final protocol or model exists. Actual accepted folds, exact final-protocol review, checkpoint replay and CSV audit are still required.',
              'runtime_seconds': time.monotonic() - start}
    report['artifacts'] = {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}
    (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'sha256': sha(out / 'manifest.json')}), flush=True)
    assert completed.returncode == 0


if __name__ == '__main__':
    main()
