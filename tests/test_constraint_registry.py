import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.engine.constraint_registry import DEFAULT_REGISTRY


def test_registry_has_six_constraints():
    assert len(DEFAULT_REGISTRY.all_constraints()) == 6


def test_hard_constraints_are_all_hard():
    for c in DEFAULT_REGISTRY.hard_constraints():
        assert c.hard_or_soft == 'hard'


def test_monthly_cap_returns_correct_values():
    assert DEFAULT_REGISTRY.monthly_cap('AMT_INCOME_TOTAL') == 5000.0
    assert DEFAULT_REGISTRY.monthly_cap('AMT_CREDIT') == 50000.0
    assert DEFAULT_REGISTRY.monthly_cap('AMT_ANNUITY') == 2000.0


def test_monthly_cap_returns_none_for_unknown():
    assert DEFAULT_REGISTRY.monthly_cap('DAYS_BIRTH') is None


def test_get_by_id():
    c = DEFAULT_REGISTRY.get('DTI_MAX_001')
    assert c is not None
    assert c.params['max_dti'] == 0.40


def test_summary_is_string():
    s = DEFAULT_REGISTRY.summary()
    assert isinstance(s, str)
    assert 'DTI_MAX_001' in s
