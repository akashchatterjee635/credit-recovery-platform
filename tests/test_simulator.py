import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest

from backend.engine.simulator import (
    DisturbanceConfig,
    ActionExecutionModel,
    FinancialShockModel,
    PolicyEnvironment,
    EnvironmentSimulator,
    run_simulation,
)


def test_disturbance_config_defaults():
    cfg = DisturbanceConfig()
    assert cfg.p_miss == 0.10
    assert cfg.beta_alpha == 8.5
    assert cfg.beta_beta == 1.5
    assert cfg.p_income_shock == 0.05
    assert cfg.p_debt_shock == 0.05
    assert cfg.policy_shift is False
    assert cfg.base_threshold == 0.30
    assert cfg.shifted_threshold == 0.25
    assert cfg.policy_shift_step == 6


def test_action_execution_miss_and_completion():
    action_model = ActionExecutionModel()
    action_req = {'BUREAU_TOTAL_DEBT': -2000.0, 'AMT_ANNUITY': -200.0}

    # 1. Guaranteed miss
    cfg_miss = DisturbanceConfig(p_miss=1.0)
    rng1 = np.random.RandomState(42)
    realized_miss = action_model.apply_action(action_req, cfg_miss, rng=rng1)
    assert realized_miss['BUREAU_TOTAL_DEBT'] == 0.0
    assert realized_miss['AMT_ANNUITY'] == 0.0

    # 2. Guaranteed no miss
    cfg_no_miss = DisturbanceConfig(p_miss=0.0, beta_alpha=10.0, beta_beta=2.0)
    rng2 = np.random.RandomState(42)
    realized_no_miss = action_model.apply_action(action_req, cfg_no_miss, rng=rng2)
    assert realized_no_miss['BUREAU_TOTAL_DEBT'] < 0.0
    assert abs(realized_no_miss['BUREAU_TOTAL_DEBT']) <= 2000.0
    assert realized_no_miss['AMT_ANNUITY'] < 0.0
    assert abs(realized_no_miss['AMT_ANNUITY']) <= 200.0


def test_financial_shock_model():
    shock_model = FinancialShockModel()

    # Rare shock forced
    cfg_rare = DisturbanceConfig(
        p_income_shock=1.0,
        p_debt_shock=1.0,
        income_shock_min=0.50,
        income_shock_max=0.80,
        debt_shock_min=5000.0,
        debt_shock_max=25000.0,
    )
    rng = np.random.RandomState(123)
    shocks = shock_model.generate_shocks(cfg_rare, rng=rng)

    assert 'income_mult' in shocks
    assert 'debt_add' in shocks
    assert shocks['income_shock'] is True
    assert shocks['debt_shock'] is True
    assert 0.50 <= shocks['income_mult'] <= 0.80
    assert 5000.0 <= shocks['debt_add'] <= 25000.0

    # Baseline noise only (no rare shock)
    cfg_base = DisturbanceConfig(p_income_shock=0.0, p_debt_shock=0.0)
    rng_base = np.random.RandomState(123)
    shocks_base = shock_model.generate_shocks(cfg_base, rng=rng_base)
    assert shocks_base['income_shock'] is False
    assert shocks_base['debt_shock'] is False
    assert shocks_base['income_mult'] > 0.0
    assert shocks_base['debt_add'] >= 0.0


def test_policy_environment_step():
    policy_env = PolicyEnvironment(base_threshold=0.30)

    # Static policy
    cfg_static = DisturbanceConfig(policy_shift=False)
    assert policy_env.step(1, cfg_static) == 0.30
    assert policy_env.step(6, cfg_static) == 0.30
    assert policy_env.step(12, cfg_static) == 0.30

    # Shifting policy (shifts at step 6 to 0.25)
    cfg_shifting = DisturbanceConfig(policy_shift=True, shifted_threshold=0.25, policy_shift_step=6)
    assert policy_env.step(5, cfg_shifting) == 0.30
    assert policy_env.step(6, cfg_shifting) == 0.25
    assert policy_env.step(10, cfg_shifting) == 0.25


def test_environment_simulator_step_and_derived_ratios():
    initial_df = pd.DataFrame([{
        'AMT_INCOME_TOTAL': 100000.0,
        'AMT_CREDIT': 200000.0,
        'AMT_ANNUITY': 10000.0,
        'BUREAU_TOTAL_DEBT': 40000.0,
        'BUREAU_MAX_OVERDUE': 0.0,
        'BUREAU_ACTIVE_COUNT': 2,
        'DERIVED_DTI': 0.10,
        'DERIVED_ANNUITY_CREDIT_RATIO': 0.05,
        'DERIVED_CREDIT_INCOME_RATIO': 2.0,
    }])

    sim = EnvironmentSimulator()
    cfg = DisturbanceConfig.zero_disturbance()  # deterministic test
    rng = np.random.RandomState(42)

    new_df, log_dict = sim.step(
        state_df=initial_df,
        requested_action={'BUREAU_TOTAL_DEBT': -10000.0, 'AMT_ANNUITY': -2000.0},
        t=1,
        config=cfg,
        rng=rng,
    )

    assert isinstance(new_df, pd.DataFrame)
    assert isinstance(log_dict, dict)
    assert log_dict['t'] == 1
    assert 'realized_action' in log_dict
    assert 'shocks' in log_dict
    assert 'policy_threshold' in log_dict

    # Check debt paydown applied
    assert new_df['BUREAU_TOTAL_DEBT'].iloc[0] < 40000.0
    # Check derived ratios recomputed
    expected_dti = new_df['AMT_ANNUITY'].iloc[0] / new_df['AMT_INCOME_TOTAL'].iloc[0]
    assert np.isclose(new_df['DERIVED_DTI'].iloc[0], expected_dti)


def test_common_random_numbers_guarantee():
    initial_df = pd.DataFrame([{
        'AMT_INCOME_TOTAL': 80000.0,
        'AMT_CREDIT': 250000.0,
        'AMT_ANNUITY': 12000.0,
        'BUREAU_TOTAL_DEBT': 30000.0,
        'BUREAU_MAX_OVERDUE': 1000.0,
        'BUREAU_ACTIVE_COUNT': 3,
    }])

    cfg = DisturbanceConfig()
    sim = EnvironmentSimulator()

    # Two separate runs initialized with identical seeds
    rng_a = np.random.RandomState(2026)
    res_a, log_a = sim.step(initial_df, {'BUREAU_TOTAL_DEBT': -3000.0}, 1, cfg, rng=rng_a)

    rng_b = np.random.RandomState(2026)
    res_b, log_b = sim.step(initial_df, {'BUREAU_TOTAL_DEBT': -3000.0}, 1, cfg, rng=rng_b)

    pd.testing.assert_frame_equal(res_a, res_b)
    assert log_a == log_b


def test_run_simulation_trajectory():
    initial_df = pd.DataFrame([{
        'AMT_INCOME_TOTAL': 90000.0,
        'AMT_CREDIT': 200000.0,
        'AMT_ANNUITY': 10000.0,
        'BUREAU_TOTAL_DEBT': 35000.0,
    }])

    def dummy_controller(state, t, threshold):
        return {'action_t': {'BUREAU_TOTAL_DEBT': -1000.0}, 'risk_t': 0.35 - 0.01 * t}

    result = run_simulation(initial_df, controller=dummy_controller, max_steps=6, seed=42)

    assert 'trajectory' in result
    assert len(result['trajectory']) == 6
    assert 'final_state' in result
    assert result['trajectory'][0]['t'] == 1
    assert result['trajectory'][-1]['t'] == 6
