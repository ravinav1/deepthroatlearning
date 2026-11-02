"""Import externally hashed2022 GPU outputs and evaluate only an audited2021 lock."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
RECIPES = {'shop664fullsample': {'protocol': 'cuda_xgb664_fullsample_protocol.json',
                       'protocol_sha256': '2a5a114ae75d9a34d4a1f580ff7c3a781332fa92120605501b5b76bc72f31c5d',
                       'verifier': 'verify_cuda_xgb664_fullsample_return.py',
                       'selector': 'select_cuda_xgb664_fullsample_blend.py',
                       'trainer': 'train_cuda_xgb664_fullsample.py',
                       'gpu_run': 'cuda-xgb664-fullsample-2022-001',
                       'smoke_run': 'cuda-xgb664-fullsample-smoke-002'}}


def digest(path: Path) -> str:
    """Hash a file without decoding its content."""
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition: bool, message: str) -> None:
    """Reject altered provenance before selection."""
    if not condition:
        raise ValueError(message)


def module(path: Path, name: str):
    """Import a previously hash-verified source file."""
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main() -> None:
    """Verify both-year provenance, then run the unchanged fixed-weight2022 selector."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', choices=RECIPES, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--archive-sha256', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--smoke-manifest-sha256', required=True)
    parser.add_argument('--selection-lock', type=Path, required=True)
    parser.add_argument('--selection-lock-sha256', required=True)
    parser.add_argument('--lock-audit', type=Path, required=True)
    parser.add_argument('--lock-audit-sha256', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--selection-run-id', required=True)
    args = parser.parse_args()
    for name in [args.run_id, args.selection_run_id]:
        require(Path(name).name == name and name not in ['', '.', '..'], 'Invalid output ID')
    require(not (ROOT / 'artifacts' / args.selection_run_id).exists(), 'Selection output already exists')
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'run_id': args.run_id,
              'experiment': args.experiment, 'year': 2022, 'status': 'initializing', 'training': False,
              'kaggle': None, 'command': shlex.join([sys.executable, *sys.argv]),
              'source_sha256': digest(Path(__file__)), 'environment': {'python': sys.version, 'platform': platform.platform()}}
    try:
        recipe = RECIPES[args.experiment]
        protocol_path = ROOT / 'docs' / recipe['protocol']
        require(digest(protocol_path) == recipe['protocol_sha256'], 'Frozen protocol changed')
        protocol = json.loads(protocol_path.read_text())
        for name, key in [(recipe['verifier'], 'return_verifier_sha256'), (recipe['selector'], 'local_selector_sha256'),
                          (recipe['trainer'], 'trainer_sha256')]:
            source = ROOT / 'scripts' / name
            require(digest(source) == protocol[key], f'Changed frozen source: {name}')
            target = out / 'frozen_source' / name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(source, target)
        require(digest(ROOT / 'sintef_project/postprocess.py') == protocol['postprocess_sha256'], 'Projection source changed')
        require(digest(args.selection_lock) == args.selection_lock_sha256
                and digest(args.lock_audit) == args.lock_audit_sha256, '2021 lock or independent audit changed')
        audit = json.loads(args.lock_audit.read_text())
        require(audit['status'] == 'passed' and audit['selection_lock_sha256'] == args.selection_lock_sha256
                and audit['eligible_for_2022'], 'No independent eligible2021 audit')
        trainer = module(ROOT / 'scripts' / recipe['trainer'], 'reviewed2022_trainer')
        lock = trainer.validate_lock(args.selection_lock, args.selection_lock_sha256, protocol, recipe['protocol_sha256'])
        helper = module(ROOT / 'scripts' / recipe['verifier'], 'reviewed2022_verifier')
        imported = out / 'download'
        helper.import_archive(args.archive, args.archive_sha256, imported, recipe['gpu_run'], recipe['smoke_run'])
        directory = imported / recipe['gpu_run']
        smoke = imported / recipe['smoke_run'] / 'manifest.json'
        report['verification'] = helper.verify_return(directory, smoke, args.manifest_sha256,
            args.smoke_manifest_sha256, protocol, 2022, ROOT, recipe['protocol_sha256'])
        command = [sys.executable, str(ROOT / 'scripts' / recipe['selector']), '--year', '2022',
                   '--gpu-run', str(directory), '--gpu-manifest-sha256', args.manifest_sha256,
                   '--smoke-manifest', str(smoke), '--smoke-manifest-sha256', args.smoke_manifest_sha256,
                   '--protocol', str(protocol_path), '--run-id', args.selection_run_id,
                   '--selection-lock', str(args.selection_lock.resolve()), '--selection-lock-sha256', args.selection_lock_sha256]
        with (out / 'selection.log').open('x') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        selection_path = ROOT / 'artifacts' / args.selection_run_id / 'manifest.json'
        selection = json.loads(selection_path.read_text())
        require(selection['status'] == 'complete' and selection['selected_weight'] == lock['selected_weight'], 'Selection changed or incomplete')
        require(set(selection['scores']) == {str(0.), str(lock['selected_weight'])}, 'Unexpected2022 weight search')
        report.update(status='complete', protocol_sha256=recipe['protocol_sha256'], archive_sha256=args.archive_sha256,
                      hosted_manifest_sha256=args.manifest_sha256, hosted_smoke_manifest_sha256=args.smoke_manifest_sha256,
                      selection_lock_sha256=args.selection_lock_sha256, lock_audit_sha256=args.lock_audit_sha256,
                      selection_command=command, selection_manifest_sha256=digest(selection_path),
                      selected_weight=selection['selected_weight'], selected_micro_auc=selection['selected_micro_auc'],
                      anchor_micro_auc=selection['anchor_micro_auc'], control_micro_auc=selection['control_micro_auc'],
                      accepted=selection['accepted'], meets_transfer_gate=selection['meets_transfer_gate'])
        print(json.dumps({key: report[key] for key in ['status', 'selected_weight', 'selected_micro_auc',
              'anchor_micro_auc', 'control_micro_auc', 'accepted', 'meets_transfer_gate']}, indent=2), flush=True)
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        shutil.copy2(__file__, out / Path(__file__).name)
        report['runtime_seconds'] = time.monotonic() - started
        report['artifacts'] = {p.relative_to(out).as_posix(): digest(p) for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        record = (f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus {report["status"]}. '
                  f'Command `{report["command"]}`. Strict2022 GPU import and unchanged2021 weight evaluation; '
                  f'weight {report.get("selected_weight")}, AUC {report.get("selected_micro_auc")}, '
                  f'anchor {report.get("anchor_micro_auc")}, control {report.get("control_micro_auc")}, '
                  f'accepted {report.get("accepted")}. Runtime {report["runtime_seconds"]:.3f}s; '
                  'all hashes, environment and source snapshots in the manifest. No fit or Kaggle action.\n')
        (out / 'experiment_record.md').write_text(record)
        with (ROOT / 'docs/experiments.md').open('a') as stream:
            stream.write(record)


if __name__ == '__main__':
    main()
