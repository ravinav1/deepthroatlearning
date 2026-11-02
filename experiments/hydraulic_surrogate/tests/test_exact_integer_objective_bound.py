"""Exact objective-gap algebra tests with no optimizer or competition data."""
from fractions import Fraction
import unittest
import numpy as np
from scipy.sparse import csc_matrix
from scripts.certify_integer_objective_gap import exact_bound


def rational(value):
    return Fraction.from_float(float(value))


def decoded(value):
    return Fraction(int(value['numerator']), int(value['denominator']))


def fixture(c, native, solution, *, eq=None, be=None, ub=None, bu=None,
            ye=None, yu=None, yl=None, yh=None, bounds=None):
    """Construct an arbitrary finite stored LP tuple without solving it."""
    c = np.asarray(c, dtype=float)
    n = len(c)
    eq = np.zeros((0, n)) if eq is None else np.asarray(eq, dtype=float)
    ub = np.zeros((0, n)) if ub is None else np.asarray(ub, dtype=float)
    problem = {'c': c, 'A_eq': csc_matrix(eq), 'b_eq': np.zeros(len(eq)) if be is None else np.asarray(be, dtype=float),
               'A_ub': csc_matrix(ub), 'b_ub': np.zeros(len(ub)) if bu is None else np.asarray(bu, dtype=float),
               'bounds': np.tile([0., np.inf], (n, 1)) if bounds is None else np.asarray(bounds, dtype=float)}
    lp = {'solution': np.asarray(solution, dtype=float), 'objective': float(c @ solution),
          'eqlin_marginals': np.zeros(len(eq)) if ye is None else np.asarray(ye, dtype=float),
          'ineqlin_marginals': np.zeros(len(ub)) if yu is None else np.asarray(yu, dtype=float),
          'lower_marginals': np.zeros(n) if yl is None else np.asarray(yl, dtype=float),
          'upper_marginals': np.zeros(n) if yh is None else np.asarray(yh, dtype=float)}
    return problem, np.asarray(native, dtype=float), lp, float(c @ native)


def independent_identity_and_bound(problem, native, lp, reported_native):
    """Use a dense exact identity and signed products rather than source branches."""
    c = list(map(rational, problem['c']))
    xn, xl = list(map(rational, native)), list(map(rational, lp['solution']))
    delta = [l-n for l, n in zip(xl, xn)]
    stationarity = c.copy()
    algebra, bound = Fraction(0), Fraction(0)
    for matrix, rhs, dual, is_eq in [('A_eq', 'b_eq', 'eqlin_marginals', True), ('A_ub', 'b_ub', 'ineqlin_marginals', False)]:
        for row, right, multiplier in zip(problem[matrix].toarray(), problem[rhs], lp[dual], strict=True):
            row, right, multiplier = list(map(rational, row)), rational(right), rational(multiplier)
            rn = sum((a*x for a, x in zip(row, xn)), Fraction(0))-right
            rl = sum((a*x for a, x in zip(row, xl)), Fraction(0))-right
            algebra += multiplier*(rl-rn)
            bound += abs(multiplier)*(abs(rn)+abs(rl)) if is_eq else abs(multiplier*rl)+max(-multiplier*rn, 0)
            stationarity = [s-a*multiplier for s, a in zip(stationarity, row)]
    for j, (lo, hi) in enumerate(problem['bounds']):
        a, b = rational(lp['lower_marginals'][j]), rational(lp['upper_marginals'][j])
        stationarity[j] -= a+b
        algebra += (a+b)*delta[j]
        lo = rational(lo)
        bound += abs(a*(xl[j]-lo))+max(-a*(xn[j]-lo), 0)
        if np.isposinf(hi):
            bound += abs(b*delta[j])
        else:
            hi = rational(hi)
            bound += abs(b*(xl[j]-hi))+max(-b*(xn[j]-hi), 0)
    algebra += sum((s*d for s, d in zip(stationarity, delta)), Fraction(0))
    bound += sum((abs(s*d) for s, d in zip(stationarity, delta)), Fraction(0))
    lp_dot = sum((a*b for a, b in zip(c, xl)), Fraction(0))
    native_dot = sum((a*b for a, b in zip(c, xn)), Fraction(0))
    bound += abs(rational(lp['objective'])-lp_dot)+abs(rational(reported_native)-native_dot)
    return algebra, lp_dot-native_dot, bound


class ExactObjectiveBoundTests(unittest.TestCase):
    def test_equality_toy_and_positive_currency_scale(self):
        p, xn, lp, reported = fixture([1.], [.5], [1.], eq=[[1.]], be=[1.], ye=[1.])
        result = exact_bound(p, xn, lp, reported, 2.)
        self.assertEqual(decoded(result['exact_dot_difference_scaled']), Fraction(1, 2))
        self.assertEqual(decoded(result['bound_scaled']), Fraction(1, 2))
        self.assertEqual(decoded(result['bound_eur']), Fraction(1))
        self.assertFalse(result['within_existing_absolute_objective_budget'])

    def test_native_lower_and_upper_bound_violations(self):
        lower = fixture([2.], [-1.], [0.], yl=[2.])
        upper = fixture([-2.], [3.], [2.], yh=[-2.], bounds=[[0., 2.]])
        for data, part in [(lower, 'native_lower_violations'), (upper, 'native_upper_violations')]:
            with self.subTest(part=part):
                result = exact_bound(*data, 1.)
                self.assertEqual(decoded(result['parts_scaled'][part]), Fraction(2))
                self.assertEqual(decoded(result['bound_scaled']), Fraction(2))

    def test_wrong_dual_signs_are_charged_not_assumed_away(self):
        cases = [
            (fixture([2.], [0.], [1.], ub=[[1.]], bu=[1.], yu=[2.]), 'wrong_sign_inequality_terms'),
            (fixture([-2.], [1.], [0.], yl=[-2.]), 'wrong_sign_lower_terms'),
            (fixture([2.], [0.], [1.], yh=[2.], bounds=[[0., 1.]]), 'wrong_sign_upper_terms')]
        for data, part in cases:
            with self.subTest(part=part):
                result = exact_bound(*data, 1.)
                self.assertEqual(decoded(result['parts_scaled'][part]), Fraction(2))
                self.assertEqual(decoded(result['bound_scaled']), Fraction(2))

    def test_nonzero_multiplier_on_infinite_upper_bound(self):
        data = fixture([5.], [0.], [2.], yh=[5.])
        result = exact_bound(*data, 1.)
        self.assertEqual(decoded(result['parts_scaled']['infinite_upper_terms']), Fraction(10))
        self.assertEqual(decoded(result['bound_scaled']), Fraction(10))

    def test_imperfect_fixed_complementarity_is_included(self):
        data = fixture([1.], [1.], [1.], yl=[2.], yh=[-1.], bounds=[[0., 3.]])
        result = exact_bound(*data, 1.)
        self.assertEqual(decoded(result['parts_scaled']['fixed_lower_complementarity']), Fraction(2))
        self.assertEqual(decoded(result['parts_scaled']['fixed_upper_complementarity']), Fraction(2))
        self.assertEqual(decoded(result['reported_difference_scaled']), Fraction(0))
        self.assertEqual(decoded(result['bound_scaled']), Fraction(4))

    def test_both_reported_objective_errors_are_included(self):
        p, xn, lp, reported = fixture([0.], [0.], [0.])
        lp['objective'], reported = .25, -.5
        result = exact_bound(p, xn, lp, reported, 1.)
        self.assertEqual(decoded(result['parts_scaled']['reported_lp_dot_discrepancy']), Fraction(1, 4))
        self.assertEqual(decoded(result['parts_scaled']['reported_native_dot_discrepancy']), Fraction(1, 2))
        self.assertEqual(decoded(result['bound_scaled']), Fraction(3, 4))

    def test_binary64_cancellation_is_recovered_exactly(self):
        p, xn, lp, reported = fixture([1e16, 1., -1e16], [0., 0., 0.], [1., 1., 1.])
        lp['objective'] = 0.
        result = exact_bound(p, xn, lp, reported, 1.)
        self.assertEqual(decoded(result['exact_dot_difference_scaled']), Fraction(1))
        self.assertEqual(decoded(result['parts_scaled']['reported_lp_dot_discrepancy']), Fraction(1))

    def test_bound_is_one_sided_not_absolute_objective_distance(self):
        result = exact_bound(*fixture([1.], [100.], [0.], yl=[1.]), 1.)
        self.assertEqual(decoded(result['reported_difference_scaled']), Fraction(-100))
        self.assertEqual(decoded(result['bound_scaled']), Fraction(0))
        self.assertTrue(result['within_existing_absolute_objective_budget'])

    def test_random_exact_identities_and_independent_signed_product_bound(self):
        rng = np.random.default_rng(4173)
        for trial in range(128):
            n = int(rng.integers(1, 6))
            d = lambda shape: rng.integers(-8, 9, size=shape).astype(float)/4
            bounds = np.column_stack([np.full(n, -2.), np.where(rng.random(n) < .3, np.inf, 2.)])
            p, xn, lp, reported = fixture(d(n), d(n), d(n), eq=d((2, n)), be=d(2), ub=d((3, n)), bu=d(3),
                                          ye=d(2), yu=d(3), yl=d(n), yh=d(n), bounds=bounds)
            lp['objective'] += float(d(1)[0])/8
            reported += float(d(1)[0])/8
            result = exact_bound(p, xn, lp, reported, 2.5)
            algebra, actual, bound = independent_identity_and_bound(p, xn, lp, reported)
            with self.subTest(trial=trial):
                self.assertEqual(algebra, actual)
                self.assertEqual(decoded(result['exact_dot_difference_scaled']), actual)
                self.assertEqual(decoded(result['bound_scaled']), bound)
                self.assertEqual(decoded(result['bound_eur']), bound*Fraction(5, 2))
                self.assertLessEqual(rational(lp['objective'])-rational(reported), bound)

    def test_nonfinite_operands_and_invalid_infinite_bounds_rejected(self):
        for field in ['c', 'native', 'solution', 'eq', 'be', 'ub', 'bu', 'ye', 'yu', 'yl', 'yh', 'objective', 'reported', 'scale', 'lower', 'upper']:
            for invalid in [np.nan, np.inf, -np.inf]:
                if field == 'upper' and np.isposinf(invalid):
                    continue
                with self.subTest(field=field, invalid=invalid):
                    p, xn, lp, reported = fixture([1.], [0.], [1.], eq=[[1.]], be=[1.], ub=[[1.]], bu=[1.])
                    scale = 1.
                    if field == 'c': p['c'][0] = invalid
                    elif field == 'native': xn[0] = invalid
                    elif field == 'solution': lp['solution'][0] = invalid
                    elif field in ['eq', 'ub']: p['A_'+field] = csc_matrix([[invalid]])
                    elif field in ['be', 'bu']: p['b_eq' if field == 'be' else 'b_ub'][0] = invalid
                    elif field in ['ye', 'yu', 'yl', 'yh']: lp[{'ye':'eqlin_marginals', 'yu':'ineqlin_marginals', 'yl':'lower_marginals', 'yh':'upper_marginals'}[field]][0] = invalid
                    elif field == 'objective': lp['objective'] = invalid
                    elif field == 'reported': reported = invalid
                    elif field == 'scale': scale = invalid
                    elif field == 'lower': p['bounds'][0, 0] = invalid
                    else: p['bounds'][0, 1] = invalid
                    with self.assertRaises(ValueError): exact_bound(p, xn, lp, reported, scale)


if __name__ == '__main__':
    unittest.main()
