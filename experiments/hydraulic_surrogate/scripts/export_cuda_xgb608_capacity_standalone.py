"""Export a predeclared standalone capacity608 GPU diagnostic after strict primary export."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shlex
import shutil
import sys
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.export_cuda_xgb608_capacity_final import digest, require, submission_frame
from sintef_project.postprocess import project_vinje


def main() -> None:
    """Create a distinct immutable diagnostic; never tune against public scores."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--verified-blend-run', type=Path, required=True)
    parser.add_argument('--verified-blend-manifest-sha256', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    require(Path(args.run_id).name == args.run_id and args.run_id not in ['', '.', '..'], 'Invalid run ID')
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {'run_id': args.run_id, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'initializing', 'command': shlex.join([sys.executable, *sys.argv]),
              'training': False, 'kaggle': None, 'purpose': 'Predeclared standalone capacity diagnostic',
              'environment': {'python': sys.version, 'platform': platform.platform(),
                              'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scikit-learn']}}}
    try:
        require(digest(args.protocol) == args.protocol_sha256, 'Diagnostic protocol changed')
        protocol = json.loads(args.protocol.read_text())
        require(protocol['version'] == 'cuda_xgb608_capacity_standalone_diagnostic_v1', 'Wrong diagnostic')
        for name, expected in protocol['source'].items():
            require(digest(ROOT / name) == expected, f'Source changed: {name}')
        for name, expected in protocol['inputs'].items():
            require(digest(ROOT / name) == expected, f'Input changed: {name}')
        scores = {}
        for year in [2021, 2022]:
            record = protocol['folds'][str(year)]
            with np.load(ROOT / record['prediction'], allow_pickle=False) as saved:
                require(saved['units'].tolist() == protocol['units'], 'Fold unit order changed')
                scores[str(year)] = float(roc_auc_score(saved['labels'].ravel(), project_vinje(saved['predictions'].astype(np.float64), protocol['units']).ravel()))
            require(scores[str(year)] == record['projected_standalone_auc'], 'Standalone validation score differs')
        primary = args.verified_blend_run.resolve()
        require(digest(primary / 'manifest.json') == args.verified_blend_manifest_sha256, 'Primary verification manifest changed')
        verified = json.loads((primary / 'manifest.json').read_text())
        require(verified['status'] == 'complete' and verified['protocol_sha256'] == protocol['final_protocol_sha256']
                and verified['final_full_test_replay_bitwise'] is True and verified['anchor_source_reconstruction_bitwise'] is True,
                'Primary strict final export must complete first')
        for name, expected in verified['artifacts'].items():
            require(digest(primary / name) == expected, f'Primary verified artifact changed: {name}')
        matches = [p for p in (primary / 'download').rglob('manifest.json')
                   if digest(p) == verified['hosted_manifest_sha256']]
        require(len(matches) == 1, 'Unique verified final model manifest required')
        directory = matches[0].parent
        final = json.loads(matches[0].read_text())
        require(final['status'] == 'complete' and final['mode'] == 'final' and final['completed_units'] == 14,
                'Final model scope differs')
        with np.load(directory / 'test_predictions.npz', allow_pickle=False) as saved:
            raw, ids, units = saved['predictions'], saved['case_ids'], saved['units'].tolist()
        replay = np.load(directory / 'replay_test_predictions.npy', allow_pickle=False)
        require(raw.dtype == replay.dtype == np.float32 and raw.tobytes() == replay.tobytes()
                and units == protocol['units'], 'Raw final replay/schema differs')
        require(raw.shape == replay.shape == (725*168,14) and ids.tolist() == protocol['test_case_ids'], 'Final dimensions/IDs differ')
        prediction = project_vinje(raw.astype(np.float64), units)
        template = pd.read_csv(ROOT / 'sintef_project/sample_submission.csv')
        frame = submission_frame(prediction, ids, units, template)
        require(frame.shape == (725, 2353), 'Submission dimensions changed')
        np.save(out / 'standalone_test_predictions.npy', prediction)
        frame.to_csv(out / 'submission_standalone_projected.csv', index=False, float_format='%.9g')
        back = pd.read_csv(out / 'submission_standalone_projected.csv')
        require(back.columns.tolist() == template.columns.tolist() and back['Run No'].equals(template['Run No']), 'CSV IDs/schema changed')
        error = float(np.max(np.abs(back.iloc[:, 1:].to_numpy() - frame.iloc[:, 1:].to_numpy())))
        require(error <= 5.1e-10, 'CSV precision changed')
        report.update(status='complete', protocol_sha256=args.protocol_sha256,
                      primary_verified_manifest_sha256=args.verified_blend_manifest_sha256,
                      hosted_manifest_sha256=verified['hosted_manifest_sha256'],
                      formula='project_vinje(rawCapacity608GPU),float64; no blend or calibrated weights',
                      validation_scores=scores, raw_replay_bitwise=True, shape=list(frame.shape),
                      probability_range=[float(prediction.min()), float(prediction.max())],
                      csv_readback_max_error=error, submission_sha256=digest(out / 'submission_standalone_projected.csv'),
                      diagnostic_export={'name':'standalone_projected_capacity608', 'formula':protocol['formula'],
                         'validation_scores':scores, 'primary_validation_scores':protocol['primary_selection']['selected_scores'],
                         'automatic_promotion':False, 'submission_sha256':digest(out / 'submission_standalone_projected.csv'),
                         'predictions_sha256':digest(out / 'standalone_test_predictions.npy'), 'csv_readback_max_error':error},
                      promotion='Diagnostic only; weaker development scores disclosed; no automatic best-result promotion')
        print(json.dumps({key: report[key] for key in ['status', 'submission_sha256', 'validation_scores']}), flush=True)
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        shutil.copy2(__file__, out / 'export_cuda_xgb608_capacity_standalone.py')
        if args.protocol.is_file():
            shutil.copy2(args.protocol, out / 'protocol.json')
        report['runtime_seconds'] = time.monotonic() - started
        report['artifacts'] = {p.relative_to(out).as_posix(): digest(p) for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        (out / 'experiment_record.md').write_text(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus {report["status"]}. Command `{report["command"]}`. Predeclared rawCapacity608GPU plus fixedVinjeprojection; no weight search/training. Validation {report.get("validation_scores")}. CSV SHA {report.get("submission_sha256")}. Full provenance in manifest; runtime {report["runtime_seconds"]:.3f}s. No Kaggle score at export.\n')


if __name__ == '__main__':
    main()
