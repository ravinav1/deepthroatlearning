"""Measure a small installed-library multilabel MLP on synthetic inputs only."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import sys
import time
import traceback
import warnings

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[key] = "2"
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Run three epochs on 4096 synthetic rows, with no competition inputs."""
    out = ROOT / "artifacts/cpu-mlp-smoke-001"
    out.mkdir(exist_ok=False)
    start = time.perf_counter()
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "command": shlex.join([sys.executable, *sys.argv]),
              "real_data_loaded": False, "gpu_used": False, "seed": 4173,
              "environment": {"python": sys.version, "platform": platform.platform(),
                              "packages": {k: importlib.metadata.version(k) for k in ["numpy", "scikit-learn", "threadpoolctl"]}},
              "kaggle": None}
    try:
        rng = np.random.default_rng(4173)
        x = rng.standard_normal((4096, 395), dtype=np.float32)
        y = rng.integers(0, 2, size=(4096, 14), dtype=np.uint8)
        model = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
                              alpha=.0001, batch_size=512, learning_rate_init=.001,
                              max_iter=3, shuffle=True, random_state=4173, tol=0.,
                              early_stopping=False, n_iter_no_change=1000)
        report["model_parameters"] = model.get_params()
        with threadpool_limits(2), warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", ConvergenceWarning)
            fit_start = time.perf_counter()
            model.fit(x, y)
            fit_seconds = time.perf_counter() - fit_start
            predictions = model.predict_proba(x[:64])
            report["threadpools"] = threadpool_info()
        if model.n_iter_ != 3 or predictions.shape != (64, 14) or not np.isfinite(predictions).all():
            raise ValueError("Unexpected fixed-epoch multilabel behavior")
        report.update(status="complete", fit_seconds=fit_seconds,
                      epoch_seconds=fit_seconds / 3, loss_curve=model.loss_curve_,
                      output_activation=model.out_activation_,
                      parameters=sum(p.size for p in model.coefs_ + model.intercepts_),
                      parameter_dtypes=sorted({str(p.dtype) for p in model.coefs_ + model.intercepts_}),
                      warnings=[str(w.message) for w in captured],
                      estimate_note="Synthetic timing only; no claim about real training convergence or resource use.")
        print(json.dumps({k: report[k] for k in ["status", "fit_seconds", "epoch_seconds", "parameters", "output_activation"]}), flush=True)
    except BaseException:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        shutil.copy2(__file__, out / "source.py")
        report["source_sha256"] = hashlib.sha256((out / "source.py").read_bytes()).hexdigest()
        report["runtime_seconds"] = time.perf_counter() - start
        (out / "manifest.json").write_text(json.dumps(report, indent=2))
        with (ROOT / "docs/experiments.md").open("a") as handle:
            handle.write(f'\n## cpu-mlp-smoke-001 — {report["timestamp_utc"]}\n\n'
                         f'Status {report["status"]}. Command `{report["command"]}`. '
                         'Synthetic4096×395 inputs and14 random binary outputs only; no competition data or GPU, no packages installed. '
                         'MLP128/64 ReLU, Adam.001, batch512, three fixedepochs, seed4173, two BLASthreads. '
                         f'Fit seconds {report.get("fit_seconds")}. Expanded settings, versions, warnings and source hash: `artifacts/cpu-mlp-smoke-001/manifest.json`. No Kaggle submission.\n')


if __name__ == "__main__":
    main()
