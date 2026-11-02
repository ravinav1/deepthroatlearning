"""Inspect the preinstalled Kaggle runtime without training or installing packages."""
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import torch

uv_path = shutil.which("uv")
report = {
    "python": sys.version,
    "platform": platform.platform(),
    "packages": {},
    "uv_path": uv_path,
    "uv_version": subprocess.run([uv_path, "--version"], capture_output=True, text=True, check=True).stdout.strip() if uv_path else None,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "devices": [],
    "training": False,
    "package_installation": False,
}
for package in ["torch", "numpy", "pandas", "scipy", "scikit-learn", "xgboost", "pyyaml"]:
    try:
        report["packages"][package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        report["packages"][package] = None
for index in range(torch.cuda.device_count()):
    prop = torch.cuda.get_device_properties(index)
    report["devices"].append({"index": index, "name": prop.name, "memory_bytes": prop.total_memory})
Path("/kaggle/working/runtime_probe.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
assert report["cuda_available"], "CUDA is unavailable; no CPU fallback."
