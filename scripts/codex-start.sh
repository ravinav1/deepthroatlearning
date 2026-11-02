#!/usr/bin/env bash
# Portable, explicit entry point. No install hook or background training daemon.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash scripts/codex-start.sh [check|start|improve] [--offline] [--hashes]
  check    Inspect config, model catalog, data and artifacts; do not run a model.
  start    Open Codex for a startup audit only (default); allow missing artifacts.
  improve  Resume authorized improvement in bounded experiment cycles.
  --offline  With check only: skip the model service; availability stays unverified.
  --hashes   Hash data and best artifacts instead of just checking sizes.
Environment: DTL_CODEX_MODEL=gpt-6-astra DTL_CODEX_EFFORT=ultra
             DTL_CODEX_PYTHON=python3 (Python 3.11+), DTL_CODEX_TIMEOUT=45
There is no silent model fallback. Read docs/CODEX_AUTOSTART.md first.
HELP
}

mode="${1:-start}"
case "$mode" in
    -h|--help) usage; exit 0 ;;
    check|start|improve) if [ "$#" -gt 0 ]; then shift; fi ;;
    *) usage >&2; exit 64 ;;
esac
offline=0
hashes=0
for option in "$@"; do
    case "$option" in
        --offline) offline=1 ;;
        --hashes) hashes=1 ;;
        *) usage >&2; exit 64 ;;
    esac
done
if [ "$offline" = 1 ] && [ "$mode" != check ]; then
    printf '%s\n' '--offline is only valid with check.' >&2
    exit 64
fi
if [ "$mode" = improve ]; then hashes=1; fi
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$repo_root"
model="${DTL_CODEX_MODEL:-gpt-6-astra}"
effort="${DTL_CODEX_EFFORT:-ultra}"
python_bin="${DTL_CODEX_PYTHON:-python3}"
timeout_seconds="${DTL_CODEX_TIMEOUT:-45}"
case "$model:$effort" in
    *[!a-zA-Z0-9._:-]*) printf '%s\n' 'Invalid model or effort identifier.' >&2; exit 64 ;;
esac
mkdir -p models/codex-session
report="models/codex-session/preflight-$(date -u +%Y%m%dT%H%M%SZ)-$$.json"
lock=models/codex-session/session.lock
if [ "$mode" != check ]; then
    if ! mkdir "$lock" 2>/dev/null; then
        printf 'A launcher lock exists: %s/%s\nInspect its owner and live jobs before removing a stale lock.\n' "$repo_root" "$lock" >&2
        exit 73
    fi
    trap 'rm -f "$lock/owner.txt"; rmdir "$lock" 2>/dev/null || true' EXIT
    printf 'pid=%s\nhost=%s\nmode=%s\nstarted_utc=%s\n' "$$" "$(hostname)" "$mode" "$(date -u +%FT%TZ)" > "$lock/owner.txt"
fi

"$python_bin" - "$repo_root" "$mode" "$offline" "$hashes" "$model" "$effort" "$report" "$timeout_seconds" <<'PY'
"""Preflight uses only the standard library; no training or inference imports."""
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone

if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11+ is required; set DTL_CODEX_PYTHON to that interpreter.")
import tomllib

root, mode, offline, hashes, model, effort, report_path, timeout = sys.argv[1:]
root = Path(root)
timeout = int(timeout)
if not 1 <= timeout <= 300:
    raise SystemExit("DTL_CODEX_TIMEOUT must be between 1 and 300 seconds.")
report = {
    "utc": datetime.now(timezone.utc).isoformat(),
    "mode": mode,
    "requested_model": model,
    "requested_effort": effort,
    "model_execution_verified": False,
    "hashes_checked": hashes == "1",
    "errors": [],
}


def inspect_files(entries):
    result = {"expected": len(entries), "present": 0, "missing": [], "invalid": []}
    for relative, expected in entries.items():
        path = root / relative
        if not path.is_file():
            result["missing"].append(relative)
            continue
        result["present"] += 1
        if path.stat().st_size != expected["size_bytes"]:
            result["invalid"].append({"path": relative, "reason": "size mismatch"})
        elif hashes == "1":
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != expected["sha256"]:
                result["invalid"].append({"path": relative, "reason": "SHA256 mismatch"})
    return result


class CatalogClient:
    """Read-only JSON-RPC metadata client with a bounded lifetime."""

    def __init__(self):
        self.state_dir = tempfile.TemporaryDirectory(prefix="dtl-codex-preflight-")
        self.proc = subprocess.Popen(
            ["codex", "--strict-config", "-C", str(root), "app-server", "--stdio",
             "-c", f'model="{model}"', "-c", f'model_reasoning_effort="{effort}"',
             "-c", "agents.enabled=true",
             "-c", "agents.max_concurrent_threads_per_session=3",
             "-c", f'agents.default_subagent_model="{model}"',
             "-c", f'agents.default_subagent_reasoning_effort="{effort}"',
             "-c", "sqlite_home=" + json.dumps(self.state_dir.name)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            # Do not persist user/global configuration, authentication data or raw logs.
            stderr=subprocess.DEVNULL, text=True, cwd=root,
        )
        self.messages = queue.Queue()
        self.counter = 0
        self.deadline = time.monotonic() + timeout
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        try:
            for line in self.proc.stdout:
                try:
                    self.messages.put(json.loads(line))
                except json.JSONDecodeError:
                    pass
        finally:
            self.messages.put(None)

    def send(self, message):
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def call(self, method, params):
        self.counter += 1
        request_id = self.counter
        self.send({"id": request_id, "method": method, "params": params})
        while True:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Codex metadata check timed out")
            try:
                message = self.messages.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError("Codex metadata check timed out") from exc
            if message is None:
                raise RuntimeError("Codex metadata service exited; run codex doctor --summary")
            if message.get("id") == request_id and "method" not in message:
                if "error" in message:
                    code = message["error"].get("code")
                    raise RuntimeError(f"Codex {method} failed (RPC code {code}); inspect with codex doctor")
                return message["result"]
            if "id" in message and "method" in message:
                # Metadata checks never approve a server request or invoke tools.
                self.send({"id": message["id"], "error": {"code": -32601, "message": "Metadata-only client"}})

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.state_dir.cleanup()


try:
    with (root / ".codex/config.toml").open("rb") as handle:
        config = tomllib.load(handle)
    roles = []
    for path in sorted((root / ".codex/agents").glob("*.toml")):
        with path.open("rb") as handle:
            role = tomllib.load(handle)
        for field in ("name", "description", "developer_instructions"):
            if not isinstance(role.get(field), str) or not role[field].strip():
                raise ValueError(f"{path.name}: missing {field}")
        roles.append(role["name"])
    if set(roles) != {"coordinator", "research", "experiment", "independent_audit"} or len(roles) != 4:
        raise ValueError("Expected four distinct project agent roles")
    report["project_roles"] = roles
    report["project_preference"] = {key: config.get(key) for key in ("model", "model_reasoning_effort")}
    recipe = json.loads((root / "configs/hydraulic_surrogate_best.json").read_text())
    inventory = json.loads((root / "reports/hydraulic_surrogate/artifacts.json").read_text())
    report["data"] = inspect_files({value["repository_path"]: value for value in recipe["data"].values()})
    best = {"models/hydraulic_surrogate/" + name: value for name, value in inventory["files"].items() if value["group"] == "best"}
    if not best:
        raise ValueError("Best artifact inventory is empty")
    report["best_artifacts"] = inspect_files(best)
    report["free_disk_gib"] = round(shutil.disk_usage(root).free / 1024**3, 2)
    report["cpu_count"] = os.cpu_count()
    report["hosted_jobs_checked"] = False
    report["live_jobs_checked"] = False
    # Presence/hash checks do not establish absence of jobs or sufficient free RAM.
    if offline == "1":
        report["model_catalog_status"] = "not checked (offline)"
    elif shutil.which("codex") is None:
        report["errors"].append("Codex CLI not found; install/sign in, then rerun check")
    else:
        version = subprocess.run(["codex", "--version"], capture_output=True, text=True, timeout=10, check=True)
        report["codex_version"] = version.stdout.strip()
        client = CatalogClient()
        try:
            client.call("initialize", {"clientInfo": {"name": "deepthroatlearning_preflight", "version": "1"}})
            client.send({"method": "initialized", "params": {}})
            resolved = client.call("config/read", {"cwd": str(root), "includeLayers": True})
            effective = resolved["config"]
            report["effective_configuration"] = {key: effective.get(key) for key in ("model", "model_reasoning_effort")}
            agent_settings = effective.get("agents") or {}
            report["effective_configuration"]["agents"] = {
                key: agent_settings.get(key) for key in (
                    "enabled", "max_concurrent_threads_per_session",
                    "default_subagent_model", "default_subagent_reasoning_effort",
                )
            }
            # Only retain paths and disabled reasons from layer metadata, not configs.
            layers = resolved.get("layers") or []
            report["project_config_loaded"] = any(
                layer.get("name", {}).get("type") == "project"
                and layer.get("name", {}).get("dotCodexFolder") == str(root / ".codex")
                and not layer.get("disabledReason") for layer in layers
            )
            if not report["project_config_loaded"]:
                report["errors"].append("Project config is not loaded; review/trust this checkout in Codex, then rerun")
            catalog = []
            cursor = None
            while True:
                params = {"limit": 100, "includeHidden": False}
                if cursor:
                    params["cursor"] = cursor
                page = client.call("model/list", params)
                for item in page["data"]:
                    catalog.append({"model": item["model"], "efforts": [value["reasoningEffort"] for value in item.get("supportedReasoningEfforts", [])]})
                cursor = page.get("nextCursor")
                if not cursor:
                    break
            report["advertised_models"] = catalog
            supported = any(item["model"] == model and effort in item["efforts"] for item in catalog)
            report["model_catalog_status"] = "advertised" if supported else "not advertised"
            if not supported:
                report["errors"].append("Requested model/effort not advertised; select a listed fallback explicitly")
            if effective.get("model") != model or effective.get("model_reasoning_effort") != effort:
                report["errors"].append("Effective model/effort differs from the requested launch settings")
            expected_agents = {
                "enabled": True,
                "max_concurrent_threads_per_session": 3,
                "default_subagent_model": model,
                "default_subagent_reasoning_effort": effort,
            }
            for key, expected in expected_agents.items():
                actual = agent_settings.get(key)
                if type(actual) is not type(expected) or actual != expected:
                    report["errors"].append(
                        f"Effective agents.{key} differs from the requested launch setting"
                    )
        finally:
            client.close()
except (OSError, ValueError, KeyError, RuntimeError, TimeoutError, subprocess.SubprocessError) as exc:
    report["errors"].append(str(exc))

complete = all(
    name in report and not report[name]["missing"] and not report[name]["invalid"]
    for name in ("data", "best_artifacts")
)
report["data_and_best_complete"] = complete
Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
print(f"Preflight report: {report_path}")
print(f"Requested model/effort: {model}/{effort}; catalog: {report.get('model_catalog_status', 'unverified')}")
print("This metadata check does not execute a model or prove successful inference.")
for name in ("data", "best_artifacts"):
    if name in report:
        item = report[name]
        print(f"{name}: {item['present']}/{item['expected']} present, {len(item['invalid'])} invalid")
if not complete:
    print("Restore inputs/artifacts using docs/DATA_AND_ARTIFACTS.md; no training is ready.")
for error in report["errors"]:
    print("ERROR: " + error, file=sys.stderr)
if report["errors"]:
    raise SystemExit(1)
if not complete and mode != "start":
    raise SystemExit(2)
PY

if [ "$mode" = check ]; then exit 0; fi
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
prompt="Read AGENTS.md, GOALS.md, docs/CODEX_AUTOSTART.md, docs/HANDOFF.md and docs/best_result.md. Act as the coordinator role defined in .codex/agents/coordinator.toml. The preflight report is $report. This session requests $model with $effort reasoning; distinguish effective configuration/catalog evidence from actual successful execution and record model evidence. Verify the project roles are available. The user explicitly authorizes deploying research, experiment and independent_audit agents for improvement: use at most three children, no grandchildren, exclusive paths and resource leases. Inspect real local and hosted jobs before any expensive work, preserve all existing results, and write a durable handoff including failures. Never submit to Kaggle from this startup request."
if [ "$mode" = start ]; then
    prompt="$prompt Perform the startup audit only: reconcile state, list missing data/artifacts with the download commands in docs/DATA_AND_ARTIFACTS.md, and identify the next eligible experiment. Do not train, replay all checkpoints, download large assets or launch hosted jobs in this audit."
else
    prompt="$prompt Resume authorized improvement from the current handoff in bounded, preregistered experiment cycles. You may implement and run local experiments after resource and duplicate-job gates pass. Default to one heavy local process, four total CPU threads and a one-hour wall-time budget per experiment including setup; a stricter frozen protocol wins, and existing user authorization for larger resource/time scope remains valid. Check actual free RAM against the declared peak before launch. If an eligible experiment cannot fit the authorized limits, complete its protocol and feasibility audit, checkpoint that resource requirement, and continue other useful authorized work. Do not launch new hosted jobs from this startup request. Obtain independent review of each measured outcome, integrate it, preserve the best, and checkpoint the exact next action between cycles. Continue the user's requested ongoing goal work without another permission round; a cycle boundary or small improvement is neither competition success nor cancellation of prior authorization. Stop only at the user's requested stopping condition, an actual authorization boundary or a genuine blocker, while preserving a durable handoff."
fi
codex --strict-config -C "$repo_root" -m "$model" \
    -c "model_reasoning_effort=\"$effort\"" \
    -c 'agents.enabled=true' -c 'agents.max_concurrent_threads_per_session=3' \
    -c "agents.default_subagent_model=\"$model\"" \
    -c "agents.default_subagent_reasoning_effort=\"$effort\"" \
    --sandbox workspace-write --ask-for-approval on-request "$prompt"
