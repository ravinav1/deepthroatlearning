"""Small, hand-computed semantic checks for case-local temporal features."""

import hashlib
import json
from pathlib import Path

import numpy as np

from temporal_features import HOURS, favourable_periods, temporal_ranks

ROOT = Path(__file__).resolve().parents[1]


def equal(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def check():
    path = ROOT / "reports/round3_temporal_feature_checks.json"
    assert not path.exists()
    price = np.zeros((3, HOURS))
    price[0, :4] = [1, 2, 2, 4]
    price[1] = np.arange(HOURS)
    price[2] = 5
    ranks = temporal_ranks(price)
    equal(ranks["temporal_rank_week"][0, :4], np.array([164.5, 166, 166, 167.5]) / HOURS)
    equal(ranks["temporal_rank_week"][0, 4:], 82 / HOURS)
    equal(ranks["temporal_rank_remaining"][0, :4], [164.5 / 168, 165 / 167, 164.5 / 166, 164.5 / 165])
    equal(ranks["temporal_rank_remaining"][0, 4:], .5)
    equal(ranks["temporal_rank_week"][1], (np.arange(HOURS) + .5) / HOURS)
    equal(ranks["temporal_rank_remaining"][1], .5 / (HOURS - np.arange(HOURS)))
    for values in ranks.values():
        equal(values[2], .5)
    equal(ranks["temporal_rank_remaining"][:, -1], .5)
    reverse_ranks = temporal_ranks(price[:, ::-1])
    equal(reverse_ranks["temporal_rank_week"], ranks["temporal_rank_week"][:, ::-1])
    equal(reverse_ranks["temporal_rank_remaining"][1], 1 - .5 / (HOURS - np.arange(HOURS)))
    # Suffix values must ignore all earlier hours, even when those are changed dramatically.
    changed = price.copy()
    changed[:, :8] = 999
    equal(temporal_ranks(changed)["temporal_rank_remaining"][:, 8:], ranks["temporal_rank_remaining"][:, 8:])

    active = np.zeros((4, HOURS), dtype=bool)
    active[0, :3] = True
    active[0, 5:7] = True
    active[0, -1] = True
    active[1] = True
    active[3, 10:13] = True
    values = np.where(active, 2., 0.)
    water = np.ones(len(values))
    periods = favourable_periods(values, water)
    equal(periods["temporal_favourable_duration"][0, :8], [3, 3, 3, 0, 0, 2, 2, 0])
    equal(periods["temporal_favourable_elapsed"][0, :8], [1, 2, 3, 0, 0, 1, 2, 0])
    equal(periods["temporal_favourable_remaining"][0, :8], [3, 2, 1, 0, 0, 2, 1, 0])
    equal(periods["temporal_favourable_previous_distance"][0, :8], [0, 0, 0, 1, 2, 0, 0, 1])
    equal(periods["temporal_favourable_next_distance"][0, :8], [0, 0, 0, 2, 1, 0, 0, 160])
    equal(periods["temporal_favourable_previous_distance"][3, :10], HOURS)
    equal(periods["temporal_favourable_next_distance"][3, 13:], HOURS)
    equal(periods["temporal_favourable_left_censored"][0], np.r_[np.ones(3), np.zeros(165)])
    equal(periods["temporal_favourable_right_censored"][0], np.r_[np.zeros(167), 1])
    equal(periods["temporal_favourable_duration"][1], HOURS)
    equal(periods["temporal_favourable_elapsed"][1], np.arange(1, HOURS + 1))
    equal(periods["temporal_favourable_remaining"][1], np.arange(HOURS, 0, -1))
    for side in ("left", "right"):
        equal(periods[f"temporal_favourable_{side}_censored"][1], 1)
    for name, result in periods.items():
        equal(result[2], HOURS if name.endswith("distance") else 0)
    equality = favourable_periods(np.ones((1, HOURS)), np.ones(1))
    equal(equality["temporal_favourable_duration"], 0)
    # Time reversal exchanges elapsed/remaining, previous/next and left/right flags.
    reversed_periods = favourable_periods(values[:, ::-1], water)
    for left, right in (("duration", "duration"), ("elapsed", "remaining"),
                        ("previous_distance", "next_distance"), ("left_censored", "right_censored")):
        equal(periods[f"temporal_favourable_{left}"], reversed_periods[f"temporal_favourable_{right}"][:, ::-1])

    for function, arguments in ((temporal_ranks, (price,)), (favourable_periods, (values, water))):
        expected = function(*arguments)
        scaled = function(*(argument * 13 for argument in arguments))
        for name in expected:
            equal(expected[name], scaled[name])
            assert "price" not in name and "water_value" not in name
            assert expected[name].shape == arguments[0].shape
            assert expected[name].dtype == np.float32 and np.isfinite(expected[name]).all()
        # Batched computation must exactly equal separate cases; reversal of case order is immaterial.
        separate = [function(*(argument[i:i + 1] for argument in arguments)) for i in range(len(arguments[0]))]
        reversed_cases = function(*(argument[::-1] for argument in arguments))
        for name in expected:
            equal(expected[name], np.concatenate([row[name] for row in separate]))
            equal(expected[name], reversed_cases[name][::-1])
    report = {
        "passed": True,
        "checks": ["hand-computed ranks and tied midranks", "flat/single-suffix midpoint equals0.5",
                   "monotone chronological ranks and reversal", "suffix ignores earlier hours",
                   "hand-computed duration/elapsed/remaining and distances", "strict positive margin",
                   "absent favourable hour sentinel168", "active boundary censor flags",
                   "all-active and all-inactive cases", "period reversal symmetry",
                   "positive joint monetary scaling invariance", "batch/case-order isolation",
                   "finite float32 outputs with (cases,168) shape", "dimensionless names evade monetary rescaling"],
        "feature_names": [*ranks, *periods],
        "semantics": "Input-only proxies. Entire supplied 168h price schedule may be used; no labels, adjacent-case data, fitted parameters or actual reservoir trajectories. Duration is censored at boundaries and equality is not favourable.",
        "module_sha256": hashlib.sha256((ROOT / "src/temporal_features.py").read_bytes()).hexdigest(),
        "check_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    with path.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    check()
