'''Unit tests for FeasibilityGuard (4 gates).'''
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock
from backend.engine.validator import FeasibilityGuard
from backend.engine.constraint_registry import DEFAULT_REGISTRY
from backend.engine.feature_contract import FEATURE_CONTRACT_V2


def make_mock_model(risk_value):
    m = MagicMock()
    m.predict_risk = MagicMock(return_value=np.array([risk_value]))
    return m


BASE_APPLICANT = pd.DataFrame([{
    'AMT_CREDIT': 300000.0,
    'AMT_INCOME_TOTAL': 100000.0,
    'AMT_ANNUITY': 15000.0,
    'DAYS_BIRTH': -12000,
    'DAYS_EMPLOYED': -2000,
    'NAME_EDUCATION_TYPE': 'Higher education',
}])


def make_guard(risk_value):
    model = make_mock_model(risk_value)
    return FeasibilityGuard(model, 0.3, DEFAULT_REGISTRY, FEATURE_CONTRACT_V2)


def test_v_risk_passes_when_below_threshold():
    guard = make_guard(0.20)
    result = guard.validate(BASE_APPLICANT, BASE_APPLICANT)
    assert result.gate_results['V_risk'] is True


def test_v_risk_fails_when_above_threshold():
    guard = make_guard(0.45)
    result = guard.validate(BASE_APPLICANT, BASE_APPLICANT)
    assert result.gate_results['V_risk'] is False
    assert any('V_risk' in v for v in result.violations)


def test_v_structural_dti_violation():
    guard = make_guard(0.20)
    # Annuity = 50% of income -> violates DTI_MAX_001 (40%)
    high_ann = BASE_APPLICANT.copy()
    high_ann['AMT_ANNUITY'] = 50000.0
    result = guard.validate(high_ann, BASE_APPLICANT)
    assert result.gate_results['V_structural'] is False
    assert any('DTI' in v for v in result.violations)


def test_v_actionability_immutable_changed():
    guard = make_guard(0.20)
    changed = BASE_APPLICANT.copy()
    changed['DAYS_BIRTH'] = -9000   # IMMUTABLE changed
    result = guard.validate(changed, BASE_APPLICANT)
    assert result.gate_results['V_actionability'] is False


def test_v_plausibility_income_cap_exceeded():
    guard = make_guard(0.20)
    # Income jump of 200k over 12 months -> max allowed = 5000 * 12 = 60000
    big_jump = BASE_APPLICANT.copy()
    big_jump['AMT_INCOME_TOTAL'] = 300000.0
    result = guard.validate(big_jump, BASE_APPLICANT)
    assert result.gate_results['V_plausibility'] is False


def test_all_gates_pass_for_valid_candidate():
    guard = make_guard(0.20)
    # Small valid annuity reduction within caps
    valid = BASE_APPLICANT.copy()
    valid['AMT_ANNUITY'] = 10000.0   # within 3-10% of 300k credit
    result = guard.validate(valid, BASE_APPLICANT)
    assert result.gate_results['V_risk'] is True
    assert result.gate_results['V_structural'] is True
    assert result.gate_results['V_actionability'] is True


def test_passed_is_conjunction_of_gates():
    guard = make_guard(0.45)  # risk gate will fail
    result = guard.validate(BASE_APPLICANT, BASE_APPLICANT)
    assert result.passed is False
