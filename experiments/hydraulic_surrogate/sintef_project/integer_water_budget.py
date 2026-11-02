"""Bounded integer-state ablation of the unchanged SHOP-inspired network LP."""
from __future__ import annotations

import time
import warnings
from unittest.mock import patch

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from sintef_project.lp_shop_relaxation import ShopRelaxationLP


class IntegerWaterBudgetLP(ShopRelaxationLP):
    """Make selected unit states binary, retaining all original network economics."""

    def solve_integer(self, case: dict, unit_indices: list[int], mip_seconds: float = 8.,
                      fixed_lp_seconds: float = 1.) -> tuple[dict, dict]:
        """Solve one bounded MILP then replay its incumbent with fixed-state LP duals.

        The two fixed-state LP calls each have a one-second cap by default.
        Conditional duals are not claimed to be global MILP shadow values.
        A time-limited integer incumbent is retained but explicitly unproven.
        """
        if not unit_indices or len(set(unit_indices)) != len(unit_indices):
            raise ValueError('Expected unique selected unit indices')
        if min(unit_indices) < 0 or max(unit_indices) >= len(self.network.units):
            raise ValueError('Selected unit outside network')
        metadata = {'status': 'initializing', 'selected_units': [self.network.units[j] for j in unit_indices],
                    'mip_time_limit_seconds': mip_seconds, 'fixed_lp_time_limit_seconds_each': fixed_lp_seconds,
                    'mip_rel_gap_requested': 0., 'seed': None, 'proven_optimal': False}
        self.last_integer_metadata = metadata
        selected = self.state[unit_indices].ravel()
        stored = {}
        self.last_integer_arrays = stored

        def adapter(c, *, A_ub, b_ub, A_eq, b_eq, bounds, method, options):
            integral = np.zeros(len(c), dtype=np.uint8)
            integral[selected] = 1
            tick = time.monotonic()
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Unrecognized options detected.*')
                integer = milp(c=np.asarray(c), integrality=integral,
                    bounds=Bounds(bounds[:, 0], bounds[:, 1]),
                    constraints=[LinearConstraint(A_eq, b_eq, b_eq), LinearConstraint(A_ub, -np.inf, b_ub)],
                    options={'time_limit': mip_seconds, 'mip_rel_gap': 0., 'threads': 1})
            scale = 1. + np.abs(case['price']).mean()
            metadata.update(status='integer_solved', mip_status=int(integer.status), mip_message=str(integer.message),
                mip_seconds=time.monotonic()-tick, has_incumbent=integer.x is not None,
                mip_gap=float(integer.mip_gap) if getattr(integer, 'mip_gap', None) is not None else None,
                mip_node_count=int(integer.mip_node_count) if getattr(integer, 'mip_node_count', None) is not None else None,
                mip_objective_scaled=float(integer.fun) if integer.fun is not None else None,
                mip_dual_bound_scaled=float(integer.mip_dual_bound) if getattr(integer, 'mip_dual_bound', None) is not None else None)
            metadata['proven_optimal'] = bool(integer.status == 0)
            if integer.x is None or not np.isfinite(integer.x).all():
                raise RuntimeError('No finite integer incumbent within the fixed time budget')
            states = np.rint(integer.x[selected])
            if np.max(np.abs(integer.x[selected]-states)) > 1e-5 or not np.isin(states, [0, 1]).all():
                raise RuntimeError('Returned incumbent is not integral')
            stored['integer_solution'] = integer.x.copy()
            stored['integer_states'] = states.reshape(len(unit_indices), self.horizon)
            fixed = np.asarray(bounds).copy()
            fixed[selected, 0] = fixed[selected, 1] = states
            fixed_options = {**options, 'time_limit': fixed_lp_seconds}
            tick = time.monotonic()
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Unrecognized options detected.*')
                first = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                                bounds=fixed, method=method, options=fixed_options)
                second = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                                 bounds=fixed, method=method, options=fixed_options)
            metadata['fixed_lp_replays_seconds'] = time.monotonic()-tick
            if not first.success or not second.success:
                raise RuntimeError(f'Fixed-state LP/replay failed: {first.status}/{second.status}')
            np.testing.assert_array_equal(first.x, second.x)
            np.testing.assert_array_equal(first.eqlin.marginals, second.eqlin.marginals)
            np.testing.assert_allclose(first.x[selected], states, rtol=0, atol=1e-10)
            # Reoptimizing continuous variables cannot worsen the incumbent.
            if first.fun > integer.fun + 1e-6:
                raise RuntimeError('Fixed-state LP worsened the MILP incumbent objective')
            metadata.update(status='complete', fixed_lp_replay_bitwise=True,
                            conditional_dual_interpretation='LP shadow values conditional on fixed selected integer states',
                            incumbent_improvement_eur=float((integer.fun-first.fun)*scale),
                            objective_gap_eur=float((first.fun-integer.mip_dual_bound)*scale)
                            if getattr(integer, 'mip_dual_bound', None) is not None else None)
            return first

        with patch('sintef_project.lp_shop_relaxation.linprog', adapter):
            result = super().solve(case, 1., time_limit=fixed_lp_seconds)
        result.update(stored)
        return result, metadata
