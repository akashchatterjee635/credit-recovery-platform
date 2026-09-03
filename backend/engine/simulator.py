'''
backend/engine/simulator.py

Simulates borrower environment dynamics under disturbances and macro shocks.
Core components:
  1. DisturbanceConfig: Probabilities and distribution parameters for execution noise,
     financial shocks, and policy environment shifts.
  2. ActionExecutionModel: Models execution noise (missed actions and partial completion)
     using Binomial and Beta distributions.
  3. FinancialShockModel: Generates rare heavy-tailed shocks (Uniform) and baseline
     monthly noise (Normal) on income and outstanding debt.
  4. PolicyEnvironment: Simulates lending policy shifts (e.g., threshold tightening)
     over the simulation horizon.
  5. EnvironmentSimulator: Main runner stepping borrower state forward in time given
     requested actions and disturbances, updating derived accounting ratios.

All stochastic components support Common Random Numbers (CRN) via explicit
np.random.RandomState instances to ensure rigorous paired counterfactual analysis.
'''
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class DisturbanceConfig:
    '''
    Configuration for environment disturbances and macro shifts.

    Moderate defaults:
      - p_miss: 0.10 (probability that an action is completely skipped)
      - beta_alpha: 8.5, beta_beta: 1.5 (completion ratio ~ Beta(8.5, 1.5), mean = 0.85)
      - p_income_shock: 0.05 (5% monthly chance of rare income drop)
      - p_debt_shock: 0.05 (5% monthly chance of rare unexpected debt expense)
      - policy_shift: False (whether recourse threshold shifts during horizon)
    '''
    # Action execution disturbances
    p_miss: float = 0.10
    beta_alpha: float = 8.5
    beta_beta: float = 1.5

    # Financial shock probabilities
    p_income_shock: float = 0.05
    p_debt_shock: float = 0.05

    # Financial shock distribution parameters
    income_shock_min: float = 0.50   # Rare shock: income drops to 50% - 80% of current
    income_shock_max: float = 0.80
    income_base_mean: float = 1.00   # Baseline: income multiplier ~ Normal(1.0, 0.02)
    income_base_std: float = 0.02

    debt_shock_min: float = 5000.0   # Rare shock: adds $5,000 - $25,000 unexpected debt
    debt_shock_max: float = 25000.0
    debt_base_mean: float = 0.0      # Baseline: debt fluctuation ~ Normal(0, 500), clipped >= 0
    debt_base_std: float = 500.0

    # Policy environment parameters
    policy_shift: bool = False
    base_threshold: float = 0.30     # Baseline approval/recourse threshold
    shifted_threshold: float = 0.25  # Shifted approval/recourse threshold
    policy_shift_step: int = 6       # Step at or after which policy shift takes effect
    macro_threshold_noise_std: float = 0.0  # Optional macro threshold volatility

    @classmethod
    def zero_disturbance(cls) -> DisturbanceConfig:
        '''Deterministic / disturbance-free baseline.'''
        return cls(
            p_miss=0.0,
            beta_alpha=1000.0,
            beta_beta=0.001,
            p_income_shock=0.0,
            p_debt_shock=0.0,
            income_base_std=0.0,
            debt_base_std=0.0,
            policy_shift=False,
        )

    @classmethod
    def low_disturbance(cls) -> DisturbanceConfig:
        '''Mild disturbance scenario.'''
        return cls(
            p_miss=0.03,
            beta_alpha=19.0,
            beta_beta=1.0,
            p_income_shock=0.02,
            p_debt_shock=0.02,
            policy_shift=False,
        )

    @classmethod
    def moderate_disturbance(cls) -> DisturbanceConfig:
        '''Moderate disturbance scenario (default benchmark settings).'''
        return cls()

    @classmethod
    def high_disturbance(cls) -> DisturbanceConfig:
        '''Severe stress scenario.'''
        return cls(
            p_miss=0.25,
            beta_alpha=5.0,
            beta_beta=2.0,
            p_income_shock=0.10,
            p_debt_shock=0.10,
            policy_shift=True,
        )


class ActionExecutionModel:
    '''
    Simulates borrower execution friction.
    For each requested feature action:
      1. Binomial trial determines if the action is missed entirely (delta = 0).
      2. If not missed, Beta(alpha, beta) determines completion percentage.
    '''

    def __init__(self, rng: Optional[np.random.RandomState] = None):
        self.rng = rng

    def _resolve_rng(self, rng: Optional[np.random.RandomState] = None) -> np.random.RandomState:
        if rng is not None:
            return rng
        if self.rng is not None:
            return self.rng
        return np.random.RandomState()

    def apply_action(
        self,
        action_dict: Dict[str, Any],
        config: DisturbanceConfig,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, float]:
        '''
        Applies execution noise to a requested action dict.
        Returns realized_action dict mapping feature_name -> realized_delta.
        '''
        # Fallback if called as ActionExecutionModel.apply_action(action_dict, config, rng)
        if not isinstance(self, ActionExecutionModel):
            effective_rng = rng if isinstance(rng, np.random.RandomState) else (
                config if isinstance(config, np.random.RandomState) else None
            )
            cfg = action_dict if isinstance(action_dict, DisturbanceConfig) else (
                config if isinstance(config, DisturbanceConfig) else DisturbanceConfig()
            )
            act = self if isinstance(self, dict) else (action_dict if isinstance(action_dict, dict) else {})
            inst = ActionExecutionModel(rng=effective_rng)
            return inst.apply_action(act, cfg, rng=effective_rng)

        effective_rng = self._resolve_rng(rng)

        if not action_dict:
            return {}

        # Normalize action_dict into {feature: delta}
        parsed_actions: Dict[str, float] = {}
        if 'actions' in action_dict and isinstance(action_dict['actions'], list):
            for item in action_dict['actions']:
                if isinstance(item, dict) and 'feature' in item:
                    f = item['feature']
                    val = float(item.get('monthly_change', 0.0))
                    direction = str(item.get('direction', '')).lower()
                    delta = -val if direction == 'decrease' else val
                    parsed_actions[f] = delta
        else:
            for k, v in action_dict.items():
                if isinstance(v, (int, float, np.number)):
                    parsed_actions[k] = float(v)

        realized: Dict[str, float] = {}
        # To guarantee CRN synchronization across different control regimes,
        # we MUST draw random variates for ALL potentially actionable features in fixed order,
        # regardless of whether the current regime requested an action for them.
        
        from backend.engine.feature_contract import FEATURE_CONTRACT_V3
        actionable_feats = sorted([k for k, v in FEATURE_CONTRACT_V3.items() if v.actionable])
        
        for feat in actionable_feats:
            # Draw variates unconditionally
            is_miss = bool(effective_rng.binomial(1, config.p_miss) == 1)
            completion_ratio = float(effective_rng.beta(config.beta_alpha, config.beta_beta))
            
            requested_delta = parsed_actions.get(feat, 0.0)
            if abs(requested_delta) < 1e-9:
                realized[feat] = 0.0
            else:
                if is_miss:
                    realized[feat] = 0.0
                else:
                    realized[feat] = float(requested_delta * completion_ratio)

        return realized

        return realized


class FinancialShockModel:
    '''
    Simulates monthly macro/micro financial shocks.
    Generates:
      - income_mult: multiplier applied to AMT_INCOME_TOTAL
        (Uniform for rare shock, Normal for baseline noise).
      - debt_add: additive lump-sum to BUREAU_TOTAL_DEBT
        (Uniform for rare shock, Normal clipped >= 0 for baseline noise).
    '''

    def __init__(self, rng: Optional[np.random.RandomState] = None):
        self.rng = rng

    def _resolve_rng(self, rng: Optional[np.random.RandomState] = None) -> np.random.RandomState:
        if rng is not None:
            return rng
        if self.rng is not None:
            return self.rng
        return np.random.RandomState()

    def generate_shocks(
        self,
        config: DisturbanceConfig,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, Any]:
        '''
        Generates financial shocks for the current step.
        Returns dict containing 'income_mult', 'debt_add', and occurrence flags.
        '''
        # Fallback if called as classmethod FinancialShockModel.generate_shocks(config, rng)
        if not isinstance(self, FinancialShockModel):
            effective_rng = rng if isinstance(rng, np.random.RandomState) else (
                config if isinstance(config, np.random.RandomState) else None
            )
            cfg = self if isinstance(self, DisturbanceConfig) else (
                config if isinstance(config, DisturbanceConfig) else DisturbanceConfig()
            )
            inst = FinancialShockModel(rng=effective_rng)
            return inst.generate_shocks(cfg, rng=effective_rng)

        effective_rng = self._resolve_rng(rng)

        # 1. Income shock
        is_income_shock = bool(effective_rng.binomial(1, config.p_income_shock) == 1)
        if is_income_shock:
            income_mult = float(effective_rng.uniform(config.income_shock_min, config.income_shock_max))
        else:
            raw_base = effective_rng.normal(config.income_base_mean, config.income_base_std)
            income_mult = float(max(0.0, raw_base))

        # 2. Debt shock
        is_debt_shock = bool(effective_rng.binomial(1, config.p_debt_shock) == 1)
        if is_debt_shock:
            debt_add = float(effective_rng.uniform(config.debt_shock_min, config.debt_shock_max))
        else:
            raw_debt = effective_rng.normal(config.debt_base_mean, config.debt_base_std)
            debt_add = float(max(0.0, raw_debt))

        return {
            'income_mult': income_mult,
            'debt_add': debt_add,
            'income_shock': is_income_shock,
            'debt_shock': is_debt_shock,
        }


class PolicyEnvironment:
    '''
    Simulates lender macro policy environment.
    Computes active risk threshold at step t.
    '''

    def __init__(self, base_threshold: Optional[float] = None):
        self.base_threshold = base_threshold

    def step(
        self,
        t: int,
        config: Optional[DisturbanceConfig] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> float:
        '''
        Returns active recourse threshold at time t.
        '''
        # Fallback if called as PolicyEnvironment.step(t, config, rng)
        if not isinstance(self, PolicyEnvironment):
            cfg = config if isinstance(config, DisturbanceConfig) else DisturbanceConfig()
            inst = PolicyEnvironment()
            return inst.step(t=t if isinstance(t, int) else 0, config=cfg, rng=rng)

        if config is None:
            config = DisturbanceConfig()

        base = self.base_threshold if self.base_threshold is not None else config.base_threshold

        if config.policy_shift and t >= config.policy_shift_step:
            threshold = config.shifted_threshold
        else:
            threshold = base

        if config.macro_threshold_noise_std > 0 and rng is not None:
            noise = float(rng.normal(0.0, config.macro_threshold_noise_std))
            threshold = float(np.clip(threshold + noise, 0.05, 0.95))

        return float(threshold)


class EnvironmentSimulator:
    '''
    Integrated environment simulator.
    Applies realized actions and financial shocks directly to the state DataFrame,
    recalculating derived accounting ratios and reporting policy threshold.
    '''

    def __init__(
        self,
        rng: Optional[np.random.RandomState] = None,
        action_model: Optional[ActionExecutionModel] = None,
        shock_model: Optional[FinancialShockModel] = None,
        policy_env: Optional[PolicyEnvironment] = None,
    ):
        self.rng = rng
        self.action_model = action_model or ActionExecutionModel(rng=rng)
        self.shock_model = shock_model or FinancialShockModel(rng=rng)
        self.policy_env = policy_env or PolicyEnvironment()

    def _resolve_rng(self, rng: Optional[np.random.RandomState] = None) -> np.random.RandomState:
        if rng is not None:
            return rng
        if self.rng is not None:
            return self.rng
        return np.random.RandomState()

    def step(
        self,
        state_df: Any,
        requested_action: Any = None,
        t: int = 1,
        config: Optional[DisturbanceConfig] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        '''
        Executes one environment simulation step.
        Returns:
          (new_state_df, log_dict)
        '''
        # Fallback if called as EnvironmentSimulator.step(state_df, requested_action, t, config, rng)
        if not isinstance(self, EnvironmentSimulator):
            actual_state_df = self
            actual_action = state_df if isinstance(state_df, dict) else (requested_action or {})
            actual_t = requested_action if isinstance(requested_action, int) else (t or 1)
            actual_cfg = t if isinstance(t, DisturbanceConfig) else (
                config if isinstance(config, DisturbanceConfig) else DisturbanceConfig()
            )
            actual_rng = config if isinstance(config, np.random.RandomState) else rng
            sim = EnvironmentSimulator(rng=actual_rng)
            return sim.step(actual_state_df, actual_action, actual_t, actual_cfg, rng=actual_rng)

        effective_rng = self._resolve_rng(rng)
        effective_config = config if config is not None else DisturbanceConfig()

        if not isinstance(state_df, pd.DataFrame):
            if isinstance(state_df, dict):
                state_df = pd.DataFrame([state_df])
            elif isinstance(state_df, pd.Series):
                state_df = pd.DataFrame([state_df])
            else:
                raise TypeError(f'state_df must be a pd.DataFrame, got {type(state_df)}')

        new_state_df = state_df.copy()
        row_idx = new_state_df.index[0]

        # 1. Exogenous shocks evaluated first to guarantee CRN alignment across policies
        shocks = self.shock_model.generate_shocks(effective_config, rng=effective_rng)

        # 2. Action execution with noise
        act_dict = requested_action if isinstance(requested_action, dict) else {}
        realized_action = self.action_model.apply_action(act_dict, effective_config, rng=effective_rng)

        # 3. Apply realized actions to state
        for feat, delta in realized_action.items():
            if feat in new_state_df.columns:
                curr_val = float(new_state_df.at[row_idx, feat])
                new_val = curr_val + delta

                # Non-negativity & domain bounds
                if feat in (
                    'AMT_INCOME_TOTAL', 'AMT_ANNUITY', 'AMT_CREDIT',
                    'BUREAU_TOTAL_DEBT', 'BUREAU_MAX_OVERDUE', 'BUREAU_ACTIVE_COUNT',
                ):
                    new_val = max(0.0, new_val)
                    if feat == 'BUREAU_ACTIVE_COUNT':
                        new_val = round(new_val)

                new_state_df.at[row_idx, feat] = new_val

        # 4. Apply financial shocks
        if 'AMT_INCOME_TOTAL' in new_state_df.columns:
            curr_inc = float(new_state_df.at[row_idx, 'AMT_INCOME_TOTAL'])
            new_state_df.at[row_idx, 'AMT_INCOME_TOTAL'] = max(0.0, curr_inc * shocks['income_mult'])

        if 'BUREAU_TOTAL_DEBT' in new_state_df.columns:
            curr_debt = float(new_state_df.at[row_idx, 'BUREAU_TOTAL_DEBT'])
            new_state_df.at[row_idx, 'BUREAU_TOTAL_DEBT'] = max(0.0, curr_debt + shocks['debt_add'])

        # 5. Recompute derived features
        curr_inc = float(new_state_df.at[row_idx, 'AMT_INCOME_TOTAL']) if 'AMT_INCOME_TOTAL' in new_state_df.columns else None
        curr_ann = float(new_state_df.at[row_idx, 'AMT_ANNUITY']) if 'AMT_ANNUITY' in new_state_df.columns else None
        curr_crd = float(new_state_df.at[row_idx, 'AMT_CREDIT']) if 'AMT_CREDIT' in new_state_df.columns else None

        if 'DERIVED_DTI' in new_state_df.columns:
            if curr_inc is not None and curr_ann is not None:
                new_state_df.at[row_idx, 'DERIVED_DTI'] = (curr_ann / curr_inc) if curr_inc > 0 else np.nan

        if 'DERIVED_ANNUITY_CREDIT_RATIO' in new_state_df.columns:
            if curr_crd is not None and curr_ann is not None:
                new_state_df.at[row_idx, 'DERIVED_ANNUITY_CREDIT_RATIO'] = (curr_ann / curr_crd) if curr_crd > 0 else np.nan

        if 'DERIVED_CREDIT_INCOME_RATIO' in new_state_df.columns:
            if curr_inc is not None and curr_crd is not None:
                new_state_df.at[row_idx, 'DERIVED_CREDIT_INCOME_RATIO'] = (curr_crd / curr_inc) if curr_inc > 0 else np.nan

        # 6. Policy environment step
        new_threshold = self.policy_env.step(t, effective_config, rng=effective_rng)

        # 7. Construct comprehensive log dictionary
        log_dict = {
            't': t,
            'requested_action': copy.deepcopy(act_dict),
            'realized_action': copy.deepcopy(realized_action),
            'shocks': copy.deepcopy(shocks),
            'income_mult': shocks.get('income_mult'),
            'debt_add': shocks.get('debt_add'),
            'income_shock': shocks.get('income_shock', False),
            'debt_shock': shocks.get('debt_shock', False),
            'policy_threshold': new_threshold,
            'state_before': state_df.iloc[0].to_dict(),
            'state_after': new_state_df.iloc[0].to_dict(),
        }

        return new_state_df, log_dict

    def run_trajectory(
        self,
        initial_state: pd.DataFrame,
        controller: Any,
        config: Optional[DisturbanceConfig] = None,
        max_steps: int = 12,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, Any]:
        '''Convenience wrapper to run a full multi-step trajectory.'''
        return run_simulation(
            initial_state=initial_state,
            controller=controller,
            config=config,
            max_steps=max_steps,
            rng=rng or self.rng,
            simulator=self,
        )


def run_simulation(
    initial_state: pd.DataFrame,
    controller: Any,
    config: Optional[DisturbanceConfig] = None,
    max_steps: int = 12,
    rng: Optional[np.random.RandomState] = None,
    seed: Optional[int] = None,
    simulator: Optional[EnvironmentSimulator] = None,
) -> Dict[str, Any]:
    '''
    Runs a full multi-step simulation trajectory.

    Args:
        initial_state: 1-row DataFrame of applicant's starting features.
        controller: An MPCController instance, callable, or list of action dicts.
        config: DisturbanceConfig specifying environment disturbances.
        max_steps: Horizon length (e.g. 12 months).
        rng: Explicit RandomState for CRN guarantee.
        seed: Seed to construct RandomState if rng not provided.
        simulator: EnvironmentSimulator instance (created if None).

    Returns:
        Dict with 'trajectory', 'final_state', and 'final_threshold'.
    '''
    effective_config = config if config is not None else DisturbanceConfig()
    effective_rng = rng if rng is not None else np.random.RandomState(seed if seed is not None else 42)
    sim = simulator if simulator is not None else EnvironmentSimulator(rng=effective_rng)

    current_state = initial_state.copy()
    trajectory_logs: List[Dict[str, Any]] = []
    current_threshold = effective_config.base_threshold

    for t in range(1, max_steps + 1):
        risk_t = None
        replan_triggered = False
        replan_reason = ''
        requested_action: Dict[str, float] = {}

        # 1. Controller query
        if hasattr(controller, 'step'):
            ctrl_out = controller.step(current_state, t, current_threshold)
            if isinstance(ctrl_out, dict):
                requested_action = ctrl_out.get('action_t', {})
                risk_t = ctrl_out.get('risk_t')
                replan_triggered = ctrl_out.get('replan_triggered', False)
                replan_reason = ctrl_out.get('reason', '')
            elif isinstance(ctrl_out, tuple) and len(ctrl_out) >= 2:
                requested_action = ctrl_out[0]
        elif callable(controller):
            ctrl_out = controller(current_state, t, current_threshold)
            if isinstance(ctrl_out, dict):
                requested_action = ctrl_out.get('action_t', ctrl_out)
                risk_t = ctrl_out.get('risk_t')
                replan_triggered = ctrl_out.get('replan_triggered', False)
                replan_reason = ctrl_out.get('reason', '')
            else:
                requested_action = ctrl_out if isinstance(ctrl_out, dict) else {}
        elif isinstance(controller, list):
            # Open-loop list of monthly action dicts
            if t - 1 < len(controller):
                item = controller[t - 1]
                if isinstance(item, dict):
                    requested_action = item.get('actions', item)
            requested_action = requested_action if isinstance(requested_action, dict) else {}

        # 2. Simulator step
        new_state, log_dict = sim.step(
            state_df=current_state,
            requested_action=requested_action,
            t=t,
            config=effective_config,
            rng=effective_rng,
        )

        current_threshold = log_dict['policy_threshold']

        entry = {
            't': t,
            'risk_t': risk_t,
            'policy_threshold': current_threshold,
            'replan_triggered': replan_triggered,
            'replan_reason': replan_reason,
            'requested_action': copy.deepcopy(requested_action),
            'realized_action': log_dict['realized_action'],
            'shocks': log_dict['shocks'],
            'income_mult': log_dict['income_mult'],
            'debt_add': log_dict['debt_add'],
            'income_shock': log_dict['income_shock'],
            'debt_shock': log_dict['debt_shock'],
            'state_before': log_dict['state_before'],
            'state_after': log_dict['state_after'],
        }
        trajectory_logs.append(entry)
        current_state = new_state

    return {
        'trajectory': trajectory_logs,
        'final_state': current_state,
        'final_threshold': current_threshold,
    }
