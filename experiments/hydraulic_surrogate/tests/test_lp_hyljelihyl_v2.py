"""Synthetic route, projection, coefficient and marginal-cost checks only."""

import unittest
import numpy as np
from scipy.optimize import linprog

from sintef_project.lp_features import Network, UNIT_ORDER, POOL_ORDER
from sintef_project.lp_hyljelihyl_v2 import (
    split_network, split_case, capture_problem, make_problem, solve,
    project_to_original, feasibility, BREAKPOINTS, MARGINAL, MAIN, HY)


def fixture(t=3):
    """Use the fixed topology with artificial volumes, curves and prices."""
    network = Network(list(POOL_ORDER), list(UNIT_ORDER),
        np.array([0,1,1,1,1,2,3,4,4,4,5,6,7,8]),
        np.array([-1,0,0,0,0,0,2,1,1,1,1,4,4,7]),
        np.array([0,1,2,3,4,4,5,6,7,8]),
        np.array([-1,0,0,0,1,0,1,4,4,4]),
        np.array([-1,0,0,0,0,1,4,4,4]),
        np.array([i==5 for i in range(14)]),
        np.array([False,True,True,True,False,True,False,False,False,False]),
        np.array([False,True,True,True,True,False,False,False,False]))
    case = {'price':np.array([10.,-2.,20.])[:t], 'initial':np.full(9,.5),
        'capacity':np.full(9,2.),'minimum':np.zeros((9,t)), 'natural':np.zeros((9,t)),
        'terminal_value':np.full(9,1.),'widths':np.full((14,1),.2), 'energy':np.full((14,1),2.),
        'qmin':np.full(14,.05),'qmax':np.full(14,.2),'pmin':np.full(14,.1),'pmax':np.full(14,.4),
        'start_cost':np.full(14,.1),'stop_cost':np.full(14,.2),
        'bypass_minimum':np.zeros((10,t)), 'vest_natural':np.zeros(t),'vest_minimum':np.zeros(t)}
    return network, case


class HyljelihylTests(unittest.TestCase):
    def test_step_widths_tail_and_convex_cost(self):
        widths = np.r_[np.diff(BREAKPOINTS), np.inf]
        for q, expected in [(20,20),(40,40),(60,1340),(100,3940),(1100,1175940)]:
            z=linprog(MARGINAL,A_eq=np.ones((1,7)),b_eq=[q],bounds=list(zip(np.zeros(7),widths)),method='highs')
            self.assertTrue(z.success); self.assertAlmostEqual(z.fun,expected)

    def test_source_case_unchanged_and_exact_coefficient_mutations(self):
        net,case=fixture(); copies={k:v.copy() for k,v in case.items()}
        split=split_case(case,.13,np.zeros(3),np.zeros(3)); n=split_network(net)
        for k in case:np.testing.assert_array_equal(case[k],copies[k])
        for k in case:
            if k not in ['initial','capacity','natural','minimum','terminal_value','bypass_minimum']:
                self.assertEqual(case[k].tobytes(),split[k].tobytes())
        m,raw=capture_problem(n,split); _,b,_=make_problem(n,split,'B'); _,c,z=make_problem(n,split,'C')
        expected=raw['c'].copy();expected[m.base.bypass[MAIN]]=0
        np.testing.assert_array_equal(b['c'],expected)
        bounds=raw['bounds'].copy();bounds[m.base.bypass[MAIN],1]=np.inf;bounds[m.base.spill[HY],1]=0
        np.testing.assert_array_equal(b['bounds'],bounds)
        for field in ['A_eq','A_ub']:
            self.assertEqual((raw[field]!=b[field][:raw[field].shape[0]]).nnz,0)
        np.testing.assert_array_equal(c['c'][:len(b['c'])],b['c'])
        self.assertEqual((c['A_ub'][:,:len(b['c'])]!=b['A_ub']).nnz,0)
        self.assertEqual((c['A_eq'][:b['A_eq'].shape[0],:len(b['c'])]!=b['A_eq']).nnz,0)
        self.assertTrue(np.isinf(c['bounds'][z[-1],1]).all())
        self.assertTrue((b['bounds'][m.base.bypass[10],1]==3.6).all())

    def test_all_old_constraints_and_minima_survive_projection(self):
        net,case=fixture();case['bypass_minimum'][4]=.015
        n=split_network(net);split=split_case(case,.13,np.zeros(3),np.zeros(3))
        a,pa=capture_problem(net,case); result_a,_,_=solve(net,case,'A')
        previous=result_a['objective']
        for arm in ['B','C']:
            result,saved,proof=solve(n,split,arm);m,_,_=make_problem(n,split,arm)
            projected=project_to_original(a,m,saved['x'])
            self.assertLessEqual(max(feasibility(pa,projected).values()),1e-7)
            self.assertLessEqual(result['objective'],previous+1e-7);previous=result['objective']
            self.assertTrue((result['bypass'][10]>=.015-1e-9).all())
            self.assertEqual(np.max(abs(result['spill'][HY])),0)

    def test_lower_pool_water_cannot_supply_upstream_bypass(self):
        net,case=fixture(1);case['initial'][:]=0;case['initial'][4]=1
        case['bypass_minimum'][4]=.1
        a,_,_=solve(net,case,'A');b,_,_=solve(split_network(net),split_case(case,0,np.zeros(1),np.zeros(1)),'B')
        self.assertEqual(float(a['bypass_slack'].max()),0)
        self.assertAlmostEqual(float(b['bypass_slack'][10,0]),.1)
        self.assertEqual(float(b['river_volume'][0]),0)

    def test_upstream_water_moves_forward_and_replay_is_literal(self):
        net,case=fixture(1);case['initial'][:]=0;case['initial'][4]=.13
        n=split_network(net);split=split_case(case,.13,np.zeros(1),np.zeros(1));split['terminal_value'][:]=100
        split['terminal_value'][9]=0  # Synthetic incentive to test one-way route and marginal dual sign.
        first,s1,_=solve(n,split,'B');second,s2,_=solve(n,split,'B')
        self.assertAlmostEqual(first['river_volume'][0],.13)
        for k in s1:self.assertEqual(s1[k].dtype,s2[k].dtype);self.assertEqual(s1[k].tobytes(),s2[k].tobytes())
        self.assertAlmostEqual(first['water_shadow'][4,0],first['water_shadow'][9,0])

    def test_interior_segment_shadow_wedge_has_correct_units_and_sign(self):
        net,case=fixture(1);case['initial'][:]=0;case['initial'][4]=.13
        split=split_case(case,.13,np.zeros(1),np.zeros(1))
        split['terminal_value'][:]=1000.;split['terminal_value'][9]=0.
        result,_,_=solve(split_network(net),split,'C')
        self.assertAlmostEqual(result['river_volume'][0],.13)
        self.assertAlmostEqual(result['river_cost'],.13/.0036)
        self.assertAlmostEqual(result['water_shadow'][4,0]-result['water_shadow'][9,0],1/.0036)

    def test_joint_bypass_cap_prevents_old_projection_overflow(self):
        net,case=fixture(1);case['initial'][:]=0;case['initial'][4]=3.73
        case['natural'][4]=3.47;case['price'][:]=0;case['terminal_value'][:]=0
        split=split_case(case,.13,np.array([3.47]),np.zeros(1))
        model,problem,_=make_problem(split_network(net),split,'B')
        # A deliberately adversarial synthetic objective tries to fill both
        # physical bypasses. The real balance/route/cap matrices remain intact.
        problem['c'][:]=0
        problem['c'][model.base.bypass[[4,10]]]=-1
        result=linprog(**problem)
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.x[model.base.bypass[[4,10]]].sum(),3.6)
        original,original_problem=capture_problem(net,case)
        projected=project_to_original(original,model,result.x)
        self.assertLessEqual(max(feasibility(original_problem,projected).values()),1e-7)
        # Removing just the new aggregate row recreates v1's forbidden 7.2.
        problem['A_ub']=problem['A_ub'][:-1];problem['b_ub']=problem['b_ub'][:-1]
        unbounded_projection=linprog(**problem)
        self.assertTrue(unbounded_projection.success)
        self.assertAlmostEqual(unbounded_projection.x[model.base.bypass[[4,10]]].sum(),7.2)


if __name__=='__main__':unittest.main()
