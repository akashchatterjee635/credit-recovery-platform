import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.engine.planner import RecoveryTrajectoryPlanner, MAX_HORIZON
from backend.engine.constraint_registry import DEFAULT_REGISTRY


def make_planner():
    return RecoveryTrajectoryPlanner(registry=DEFAULT_REGISTRY)


BASE = {'AMT_CREDIT': 300000.0, 'AMT_INCOME_TOTAL': 100000.0,
        'AMT_ANNUITY': 15000.0}


def test_no_change_returns_one_month():
    p = make_planner()
    result = p.generate_timeline(BASE, BASE)
    assert result['status'] == 'feasible'
    assert result['total_months'] == 1


def test_small_annuity_change_within_horizon():
    p = make_planner()
    target = BASE.copy()
    target['AMT_ANNUITY'] = 13000.0   # delta = 2000, cap = 2000/mo -> 1 month
    result = p.generate_timeline(BASE, target)
    assert result['status'] == 'feasible'
    assert result['total_months'] == 1


def test_large_income_change_infeasible():
    p = make_planner()
    target = BASE.copy()
    # income jump of 300k, cap=5000/mo -> 60 months needed -> infeasible
    target['AMT_INCOME_TOTAL'] = 400000.0
    result = p.generate_timeline(BASE, target)
    assert result['status'] == 'infeasible_within_horizon'
    assert result['months_required'] > MAX_HORIZON


def test_timeline_length_matches_total_months():
    p = make_planner()
    target = BASE.copy()
    target['AMT_ANNUITY'] = 9000.0   # delta=6000, cap=2000/mo -> 3 months
    result = p.generate_timeline(BASE, target)
    assert result['status'] == 'feasible'
    assert len(result['timeline']) == result['total_months']


def test_final_step_is_marked():
    p = make_planner()
    target = BASE.copy()
    target['AMT_ANNUITY'] = 9000.0
    result = p.generate_timeline(BASE, target)
    assert result['timeline'][-1]['is_final'] is True
