"""One-reservoir directional split and supplied convex main-river cost.

Build every existing generator/state coefficient through the frozen SHOP module.
No existing module is modified. Matrix capture performs no optimization.
"""

from dataclasses import replace
from unittest.mock import patch
import warnings

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, hstack, vstack

from sintef_project.lp_features import Network
from sintef_project.lp_shop_relaxation import ShopRelaxationLP, effective_energy, state_interval

VINJE, HY = 4, 9
HY_BYPASS, MAIN = 10, 11
BREAKPOINTS = np.array([0., 40., 100., 200., 400., 700., 1000.])
MARGINAL = np.array([1., 65., 239., 564., 1107., 1758., 1758.])


def split_network(base: Network) -> Network:
    """Preserve old routes except explicitly redirected arrivals and new edges."""
    assert len(base.pools) == 9 and base.pools[4] == 'Vinje'
    assert len(base.units) == 14 and base.units[12] == 'Kjela_G1'
    assert len(base.bypass_up) == 10
    result = replace(base, pools=[*base.pools, 'Hyljelihyl'], units=list(base.units),
                     unit_up=base.unit_up.copy(), unit_down=base.unit_down.copy(),
                     bypass_up=np.r_[base.bypass_up, HY, HY],
                     bypass_down=np.r_[base.bypass_down, 1, VINJE],
                     spill_down=np.r_[base.spill_down, 1],
                     vest_units=base.vest_units.copy(),
                     vest_bypasses=np.r_[base.vest_bypasses, False, False],
                     vest_spills=np.r_[base.vest_spills, False])
    result.unit_down[12] = HY
    result.bypass_down[9] = HY
    result.spill_down[8] = HY
    return result


def split_case(case: dict, initial: float, natural: np.ndarray,
               minimum: np.ndarray, capacity: float = .26) -> dict:
    """Partition the old pool exactly; other physical inputs remain unchanged."""
    result = {k: np.asarray(v).copy() for k, v in case.items()}
    t = len(case['price'])
    natural, minimum = np.asarray(natural, dtype=np.float64), np.asarray(minimum, dtype=np.float64)
    assert natural.shape == minimum.shape == (t,)
    assert np.isfinite(natural).all() and np.isfinite(minimum).all()
    assert 0 <= initial <= capacity and (minimum >= 0).all() and (minimum <= capacity).all()
    for key, value in [('initial', initial), ('capacity', capacity)]:
        result[key] = np.r_[np.asarray(case[key], dtype=np.float64), value]
        result[key][VINJE] -= value
    for key, value in [('natural', natural), ('minimum', minimum)]:
        result[key] = np.vstack([np.asarray(case[key], dtype=np.float64), value])
        result[key][VINJE] -= value
    assert np.min(result['natural']) >= -1e-12 and np.min(result['minimum']) >= -1e-12
    result['terminal_value'] = np.r_[case['terminal_value'], case['terminal_value'][VINJE]]
    result['bypass_minimum'] = np.vstack([case['bypass_minimum'], case['bypass_minimum'][4], np.zeros(t)])
    result['bypass_minimum'][4] = 0
    return result


class Captured(Exception):
    """Internal control flow ending matrix construction before a solver call."""


def capture_problem(network: Network, case: dict, seconds: float = 20.) -> tuple:
    """Obtain exact frozen matrices/coefficients with zero optimization calls."""
    model = ShopRelaxationLP(network, len(case['price']), case['energy'].shape[1])
    captured = {}

    def capture(c, **kwargs):
        captured.update(c=np.asarray(c).copy(), **kwargs)
        raise Captured()

    with patch('sintef_project.lp_shop_relaxation.linprog', capture):
        try:
            model.solve(case, 1., seconds)
        except Captured:
            pass
    assert captured
    return model, captured


def make_problem(network: Network, case: dict, arm: str, seconds: float = 20.) -> tuple:
    """Build A unchanged, B directed/free, or C directed/supplied cost."""
    if arm not in ['A', 'B', 'C']:
        raise ValueError('Unknown fixed arm')
    model, problem = capture_problem(network, case, seconds)
    segment_indices = np.empty((0, len(case['price'])), dtype=int)
    if arm == 'A':
        return model, problem, segment_indices
    assert network.pools[-1] == 'Hyljelihyl' and len(network.pools) == 10
    base, t = model.base, model.horizon
    # Extra bypass retains the supplied1000m3/s cap. Main river has no cap.
    problem['c'][base.bypass[MAIN]] = 0.
    problem['bounds'][base.bypass[MAIN], 1] = np.inf
    problem['bounds'][base.spill[HY], 1] = 0.
    if arm == 'C':
        old_n = len(problem['c'])
        segment_indices = np.arange(old_n, old_n + 7 * t).reshape(7, t)
        scale = 1 + np.abs(case['price']).mean()
        problem['c'] = np.r_[problem['c'], np.repeat(MARGINAL / .0036 / scale, t)]
        widths = np.r_[np.diff(BREAKPOINTS) * .0036, np.inf]
        problem['bounds'] = np.vstack([problem['bounds'], np.c_[np.zeros(7*t), np.repeat(widths, t)]])
        problem['A_eq'] = hstack([problem['A_eq'], coo_matrix((problem['A_eq'].shape[0], 7*t))], format='csc')
        problem['A_ub'] = hstack([problem['A_ub'], coo_matrix((problem['A_ub'].shape[0], 7*t))], format='csc')
        rows, cols, values = [], [], []
        for hour in range(t):
            rows.append(hour); cols.append(base.bypass[MAIN, hour]); values.append(-1.)
            for segment in range(7):
                rows.append(hour); cols.append(segment_indices[segment, hour]); values.append(1.)
        river_eq = coo_matrix((values, (rows, cols)), shape=(t, len(problem['c']))).tocsc()
        problem['A_eq'] = vstack([problem['A_eq'], river_eq], format='csc')
        problem['b_eq'] = np.r_[problem['b_eq'], np.zeros(t)]
    return model, problem, segment_indices


def feasibility(problem: dict, x: np.ndarray) -> dict:
    """Return original matrix residuals, without another optimization."""
    bounds = problem['bounds']
    return {'equality': float(np.max(np.abs(problem['A_eq'] @ x - problem['b_eq']))),
            'inequality': float(max(0., np.max(problem['A_ub'] @ x - problem['b_ub']))),
            'bounds': float(max(0., np.max(bounds[:, 0] - x), np.max(x - bounds[:, 1])))}


def kkt(problem: dict, saved: dict) -> dict:
    """Check full primal/dual feasibility and complementarity of saved output."""
    x = saved['x']
    stationarity = (problem['c'] - problem['A_eq'].T @ saved['eqlin_marginals']
                    - problem['A_ub'].T @ saved['ineqlin_marginals']
                    - saved['lower_marginals'] - saved['upper_marginals'])
    upper = problem['bounds'][:, 1]
    finite = np.isfinite(upper)
    dual = (problem['b_eq'] @ saved['eqlin_marginals']
            + problem['b_ub'] @ saved['ineqlin_marginals']
            + problem['bounds'][:, 0] @ saved['lower_marginals']
            + upper[finite] @ saved['upper_marginals'][finite])
    return {**feasibility(problem, x),
            'stationarity': float(np.max(np.abs(stationarity))),
            'dual_sign': float(max(0., saved['ineqlin_marginals'].max(),
                                  -saved['lower_marginals'].min(), saved['upper_marginals'].max())),
            'complementarity': float(max(np.max(np.abs((problem['b_ub']-problem['A_ub']@x)*saved['ineqlin_marginals'])),
                                         np.max(np.abs((x-problem['bounds'][:,0])*saved['lower_marginals'])),
                                         np.max(np.abs((upper[finite]-x[finite])*saved['upper_marginals'][finite])))),
            'scaled_primal_dual_gap': float(abs(problem['c'] @ x - dual))}


def project_to_original(original_model: ShopRelaxationLP, split_model: ShopRelaxationLP,
                        x: np.ndarray) -> np.ndarray:
    """Map every split variable needed by the original constraints to old indices."""
    old, new = original_model.base, split_model.base
    target = np.zeros(original_model.n_variables)
    target[old.q] = x[new.q]
    for name in ['v', 'low_slack', 'high_slack']:
        oi, ni = getattr(old, name), getattr(new, name)
        target[oi] = x[ni[:9]]
        target[oi[VINJE]] += x[ni[HY]]
    target[old.spill] = x[new.spill[:9]]
    for name in ['bypass', 'bypass_slack']:
        oi, ni = getattr(old, name), getattr(new, name)
        target[oi] = x[ni[:10]]
        target[oi[4]] += x[ni[HY_BYPASS]]
    target[old.vest_slack] = x[new.vest_slack]
    for name in ['state', 'start', 'stop']:
        target[getattr(original_model, name)] = x[getattr(split_model, name)]
    return target


def solve(network: Network, case: dict, arm: str, seconds: float = 20.) -> tuple:
    """Solve one declared arm and retain all full primal/dual arrays."""
    model, problem, segments = make_problem(network, case, arm, seconds)
    saved_calls = []
    if arm == 'A':
        # Use the original solver and original result decoding exactly.
        def capture_and_solve(c, **kwargs):
            result = linprog(c, **kwargs)
            saved_calls.append(result)
            return result
        with patch('sintef_project.lp_shop_relaxation.linprog', capture_and_solve):
            result = model.solve(case, 1., seconds)
        native = saved_calls[0]
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Unrecognized options detected.*')
            native = linprog(**problem)
        if not native.success:
            raise RuntimeError(f'LP failed {native.status}: {native.message}')
        b, x, t = model.base, native.x, model.horizon
        scale = 1 + np.abs(case['price']).mean()
        energy = effective_energy(case, t)
        flow, power = x[b.q].sum(axis=1), (x[b.q] * energy).sum(axis=1)
        lower, upper = state_interval(case, flow, power)
        result = {'flow': flow, 'power': power, 'state': x[model.state],
                  'raw_start': x[model.start], 'raw_stop': x[model.stop],
                  'state_lower': lower, 'state_upper': upper,
                  'segment_flow': x[b.q], 'storage': x[b.v], 'bypass': x[b.bypass],
                  'spill': x[b.spill], 'water_shadow': -native.eqlin.marginals[:len(network.pools)*t].reshape(-1,t)*scale,
                  'minimum_slack': x[b.low_slack], 'maximum_slack': x[b.high_slack],
                  'bypass_slack': x[b.bypass_slack], 'vest_slack': x[b.vest_slack],
                  'objective': float(-native.fun*scale), 'status': int(native.status),
                  'river_volume': x[b.bypass[MAIN]], 'river_segment_volume': x[segments],
                  'river_cost': float(np.sum(x[segments]*MARGINAL[:,None]/.0036)) if arm == 'C' else 0.}
        transition = float(np.sum(x[model.start]*case['start_cost'][:,None])
                           + np.sum(x[model.stop]*case['stop_cost'][:,None]))
        penalty = case.get('slack_penalty', 1e7+100*np.max(np.abs(case['terminal_value'])))
        reconstructed = (np.sum(power*case['price'][None,:]) + case['terminal_value']@x[b.v][:,-1]
                         - float(case.get('bypass_cost',10/.0036))*x[b.bypass[:MAIN]].sum()
                         - float(case.get('spill_cost',1000/.0036))*x[b.spill].sum()
                         - penalty*sum(x[j].sum() for j in [b.low_slack,b.high_slack,b.bypass_slack,b.vest_slack])
                         - transition-result['river_cost'])
        np.testing.assert_allclose(result['objective'],reconstructed,rtol=2e-11,atol=1e-5)
        result['reconstructed_objective'] = float(reconstructed)
        result['transition_cost'] = transition
    saved = {'x':native.x, 'objective_scaled':np.array(native.fun)}
    for name in ['eqlin','ineqlin','lower','upper']:
        for field in ['marginals','residual']:
            saved[name+'_'+field] = np.asarray(getattr(getattr(native,name),field))
    proof = kkt(problem, saved)
    assert max(proof.values()) <= 1e-5, proof
    return result, saved, proof
