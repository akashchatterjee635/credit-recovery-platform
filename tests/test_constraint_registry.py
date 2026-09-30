import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.engine.constraint_registry import DEFAULT_REGISTRY


EXPECTED_CONSTRAINTS = {
    "DTI_MAX_001",
    "ANNUITY_CREDIT_MIN_001",
    "ANNUITY_CREDIT_MAX_001",
    "MONTHLY_INCOME_CAP_001",
    "MONTHLY_CREDIT_CAP_001",
    "MONTHLY_ANNUITY_CAP_001",
    "MONTHLY_DEBT_PAYDOWN_CAP_001",
    "MONTHLY_OVERDUE_RESOLUTION_CAP_001",
    "MONTHLY_ACTIVE_CREDIT_CAP_001",
    "RECOURSE_THRESHOLD_001",
}

def test_registry_has_expected_constraints():
    actual = {c.constraint_id for c in DEFAULT_REGISTRY.all_constraints()}
    assert actual == EXPECTED_CONSTRAINTS


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
