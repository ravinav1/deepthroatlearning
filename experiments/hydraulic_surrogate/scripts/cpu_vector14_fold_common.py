"""Prospective CPU shared-vector fold contract and immutable selection helpers."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scripts.cpu_vector14_portability import sha
ROOT = Path(__file__).resolve().parents[1]
ROUNDS = [600, 1800]
WEIGHTS = [0., .1, .25, .5]


def read_protocol(path: Path, expected: str) -> dict:
    """Check prospective recipe and byte closure before accessing model inputs."""
    assert sha(path) == expected
    p = json.loads(path.read_text())
    assert p['version'] == 'cpu_vector14_664_v1' and p['candidate_rounds'] == ROUNDS and p['candidate_weights'] == WEIGHTS
    assert p['model_parameters']['device'] == 'cpu' and p['model_parameters']['multi_strategy'] == 'multi_output_tree'
    assert p['model_parameters']['max_bin'] == 128 and p['model_parameters']['n_jobs'] == 2
    assert len(p['feature_columns']) == 664 and len(p['units']) == 14
    for group in ['source', 'inputs', 'data']:
        for name, value in p[group].items():
            assert sha(ROOT / name) == value, name
    return p


def review(path: Path, expected: str, protocol_sha: str) -> dict:
    """Require an actual independent protocol/source approval before execution."""
    assert sha(path) == expected
    a = json.loads(path.read_text())
    assert a['status'] == 'passed' and a['protocol_sha256'] == protocol_sha
    return a


def choose(scores: dict[str, float]) -> tuple[int, float]:
    """Select higher2021 AUC, then lower rounds, then lower weight without tolerance."""
    choices = [(r, w) for r in ROUNDS for w in WEIGHTS]
    assert set(scores) == {f'{r}:{w}' for r, w in choices}
    assert all(np.isfinite(v) and 0 <= v <= 1 for v in scores.values())
    return max(choices, key=lambda rw: (scores[f'{rw[0]}:{rw[1]}'], -rw[0], -rw[1]))


def validate_lock(path: Path, expected: str, p: dict, protocol_sha: str,
                  audit_path: Path, audit_sha: str) -> dict:
    """Lock both capacities and blend before building2022 features or decoding labels."""
    assert sha(path) == expected and sha(audit_path) == audit_sha
    lock = json.loads(path.read_text()); a = json.loads(audit_path.read_text())
    assert a['status'] == 'passed' and a['selection_lock_sha256'] == expected and a['eligible_for_2022']
    assert a['protocol_sha256'] == protocol_sha
    assert lock['version'] == 'cpu_vector14_2021_lock_v1' and lock['protocol_sha256'] == protocol_sha
    assert lock['candidate_rounds'] == ROUNDS and lock['candidate_weights'] == WEIGHTS
    selected = choose(lock['scores_2021'])
    assert selected == (lock['selected_rounds'], lock['selected_weight']) and selected[1] > 0
    assert lock['eligible_for_2022'] and lock['selected_2021_micro_auc'] == lock['scores_2021'][f'{selected[0]}:{selected[1]}']
    assert all(lock['anchor_2021_micro_auc'] == lock['scores_2021'][f'{r}:0.0'] for r in ROUNDS)
    assert lock['selected_2021_micro_auc'] > max(lock['anchor_2021_micro_auc'], lock['control_2021_micro_auc'])
    for kind in ['anchor', 'control']:
        assert lock[kind + '_prediction_sha256'] == p[kind]['2021']['prediction_sha256']
    return lock


def artifacts(folder: Path) -> dict:
    """Hash only top-level immutable evidence files, excluding the final manifest."""
    return {p.name: sha(p) for p in folder.iterdir() if p.is_file() and p.name != 'manifest.json'}


def verify_files(folder: Path, hashes: dict) -> None:
    """Reject unsafe payload names or changed input/output bytes."""
    for name, expected in hashes.items():
        assert Path(name).name == name and not (folder / name).is_symlink()
        assert sha(folder / name) == expected, name
