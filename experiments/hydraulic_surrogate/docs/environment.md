# Local ML environment

Established 2026-11-02. Use `.venv-ml/bin/python` from the repository root. Python 3.11.15, macOS/Darwin 27.0.0, arm64. Native libraries use installed Homebrew libomp. Main system Python is 3.14.6 and the original `.venv` is empty; do not use it for experiments.

The user requested uv. The final environment was created/reconciled with uv and all ML distributions were installed via uv. Initial root `python3 -m venv .venv` / pip setup failed on DNS. Existing cached Python 3.11 and package distributions were discovered read-only. An initial copy of cached distributions was superseded by uv reinstall of the same distributions. Default uv cache writes were denied; all subsequent uv calls used a writable cache. Direct uv network install failed DNS. Standalone approved curl calls successfully downloaded official PyPI wheels; compound shell commands containing curl could still fail DNS.

## Reproduction

On an unrestricted network, use:

```sh
uv venv --python 3.11 .venv-ml
uv pip install --python .venv-ml/bin/python numpy==2.4.6 pandas==3.0.6 scikit-learn==1.9.1 scipy==1.17.1 lightgbm==4.7.0 catboost==1.2.10 xgboost==3.2.0 pyyaml==6.0.3 pypdf==6.19.0
```

Executed locally:

```sh
uv --cache-dir /private/tmp/tdt4173-uv venv --python /Users/heap/.local/share/uv/python/cpython-3.11-macos-aarch64-none/bin/python3.11 --allow-existing .venv-ml
uv --cache-dir /private/tmp/tdt4173-uv pip install --python .venv-ml/bin/python --no-index --find-links /private/tmp/tdt4173-wheels --reinstall numpy pandas scikit-learn scipy xgboost lightgbm pypdf pyyaml cloudpickle
uv --cache-dir /private/tmp/tdt4173-uv pip install --python .venv-ml/bin/python --no-index --find-links /private/tmp/tdt4173-wheels catboost
```

Packages in `/private/tmp/tdt4173-wheels` are local wheel archives reconstructed from installed uv cache distributions (preserving contents, WHEEL, METADATA and RECORD) except LightGBM, CatBoost, Plotly and Graphviz, downloaded directly from PyPI URLs. The offline wheel archive is transient and excluded from source control. Full provenance below distinguishes reconstructed archive hashes from upstream release hashes.

## Runtime versions

```text
catboost==1.2.10
cloudpickle==3.1.2
contourpy==1.3.3
cycler==0.12.1
fonttools==4.67.0
graphviz==0.21
joblib==1.6.0
kiwisolver==1.5.1
lightgbm==4.7.0
matplotlib==3.11.1
narwhals==2.26.0
numpy==2.4.6
packaging==26.3
pandas==3.0.6
pillow==13.0.0
plotly==7.1.0
pyparsing==3.3.2
pypdf==6.19.0
python-dateutil==2.9.0.post0
pyyaml==6.0.3
scikit-learn==1.9.1
scipy==1.17.1
six==1.17.0
threadpoolctl==3.7.0
tzdata==2026.4
xgboost==3.2.0
```

Core imports passed; sklearn binary AUC smoke `[0,1]` against `[0.1,0.9]` returned 1.0. This is an environment check, not competition validation. Two-tree LightGBM, CatBoost, and XGBoost classifier fit/predict smoke checks also passed on synthetic 100-row data (2 threads, no written model artifacts). These smoke results are unrelated to competition validation.

## Wheel archive SHA-256

```json
{
  "catboost-1.2.10-cp311-cp311-macosx_11_0_universal2.whl": "ab2e84237308d62bae236b1ecba2e3867697f96bdbaf0ca68dafc2c886946406",
  "cloudpickle-3.1.2-py3-none-any.whl": "cb0d960b541bb707b95072c862f40e188d6084062f30196489c8d4b094eac406",
  "contourpy-1.3.3-cp311-cp311-macosx_11_0_arm64.whl": "38983726b46f774c773456206a63569123b170746a68132fad08bc08829908b0",
  "cycler-0.12.1-py3-none-any.whl": "40b57a85caf67fd3d72615837fb9050c63e1e351c86c2beefc5d435c05f478a3",
  "fonttools-4.67.0-cp311-cp311-macosx_10_9_universal2.whl": "04f9e7269108f672b7c3a0b2bbe768265b05db0e9467838fe63728677911361d",
  "graphviz-0.21-py3-none-any.whl": "54f33de9f4f911d7e84e4191749cac8cc5653f815b06738c54db9a15ab8b1e42",
  "joblib-1.6.0-py3-none-any.whl": "48469f98b2f79507b56cc4585b494ffb064c07ae2e6aba655ac6196bc09ce76d",
  "kiwisolver-1.5.1-cp311-cp311-macosx_11_0_arm64.whl": "148fd360eed10dcbc0e87dd3853832db60aef8f18659752afb8365fab91b971a",
  "lightgbm-4.7.0-py3-none-macosx_12_0_arm64.whl": "129535462686f274df179133643118c5c5c5667167fe6c3a28d955f0b3c8e868",
  "matplotlib-3.11.1-cp311-cp311-macosx_11_0_arm64.whl": "4316b8ccf412a51e46269a8a89508bdc46160c87c7080e3740b2feb923a2f0ff",
  "narwhals-2.26.0-py3-none-any.whl": "107f757f3fdf27c81c34abc76adf77cf40e1c477b0aba32734563958a13ff005",
  "numpy-2.4.6-cp311-cp311-macosx_14_0_arm64.whl": "b9f33293eacc4d11483b8e779386316c2b62db8397b92a35466c52b7a64b9507",
  "packaging-26.3-py3-none-any.whl": "2c8a89c3d57f5c81d328f8b8efa93766ef4d57adae8072b16faf3679492dd86e",
  "pandas-3.0.6-cp311-cp311-macosx_11_0_arm64.whl": "67415152d9f18d8a767a4c9ab6dfe6c788077304269f2bc542e24270e42e3b6d",
  "pillow-13.0.0-cp311-cp311-macosx_11_0_arm64.whl": "1aa845140b847e78387f71542e24739324a6ebc9d798cc88b89e667c74ddb669",
  "plotly-7.1.0-py3-none-any.whl": "dbb7fa18afce40d0a8e80d1bf162eceb3faa0ce5a77fe741ad09a74cf78f53f3",
  "pyparsing-3.3.2-py3-none-any.whl": "37e52c86e7f9369246c31f881270edf72361d190c07d2cf3679e6f0e2d0619f8",
  "pypdf-6.19.0-py3-none-any.whl": "801597b2ba8cf0725fb9ba7e6ef597fe336e25aaa400afedc27c37b388016009",
  "python_dateutil-2.9.0.post0-py2.py3-none-any.whl": "2b67f5b3b36061bd315dcc7e48b7720eb5268a8f54ee4f99ff1b359cdda721fe",
  "pyyaml-6.0.3-cp311-cp311-macosx_11_0_arm64.whl": "c553d7301fbb07951391b65dbb161c0779d726d1491235e89af4645fcef72605",
  "scikit_learn-1.9.1-cp311-cp311-macosx_12_0_arm64.whl": "be6d852b25b03c11cfbd5a20ee79dab1da5fb1da57ad20b10513842b217cee72",
  "scipy-1.17.1-cp311-cp311-macosx_14_0_arm64.whl": "c43a8cb742bd3c9f4a27af753f88dae0c3450372efb996b40131b298a1b01b00",
  "six-1.17.0-py2.py3-none-any.whl": "749f69e5a308a26ec84f9bcd28dd31709d006d469a854ec496caa5aa526bbbf9",
  "threadpoolctl-3.7.0-py3-none-any.whl": "a9e3c335246287102eec6b09b6947720b3ea6e7240d06020b875b276056357a3",
  "tzdata-2026.4-py2.py3-none-any.whl": "dce6f4f6d56ccfdd80c1ad96596e9292e484a8f9f57d97dd9002594dfa346f1d",
  "xgboost-3.2.0-py3-none-macosx_12_0_arm64.whl": "3289c5aa01dbf2c78cad93fa68995ec41648004f1c477d183effbf6e099f31a3"
}
```

## uv project workflow

`pyproject.toml` pins every runtime dependency shown above and restricts Python to 3.11. The environment is not an installable package. On a machine with normal PyPI access, synchronize with:

```sh
UV_PROJECT_ENVIRONMENT=.venv-ml uv sync --python 3.11
UV_PROJECT_ENVIRONMENT=.venv-ml uv run python -m unittest discover -s tests
```

For the current offline environment, reconcile the exact project dependencies with:

```sh
uv --cache-dir /private/tmp/tdt4173-uv pip install --python .venv-ml/bin/python --no-index --find-links /private/tmp/tdt4173-wheels -r pyproject.toml
.venv-ml/bin/python -m unittest discover -s tests
```

A portable registry-based `uv.lock` was not produced: uv cannot resolve PyPI DNS here, and locking against local wheel archives would embed transient absolute `/private/tmp` paths. Exact project pins and wheel SHA-256 values preserve this session's package recipe. The first successful normal-network `uv sync` should produce and retain `uv.lock`.
