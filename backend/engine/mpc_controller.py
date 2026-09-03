"""
backend/engine/mpc_controller.py

Closed-loop receding-horizon MPC controller for credit recovery.

Key design decisions:
  1. Event-triggered replanning: R_t = 1 only if observed state/risk
     materially deviates from the EXPECTED trajectory of the current plan,
     or if lender policy changes. Simply being above tau_target mid-plan
     is NOT a trigger — the plan is allowed time to work.

  2. Normalized instability: I_t = sum_j w_j * |delta_x_j / s_j| so that
     a Rs10,000 debt revision does not numerically swamp a credit-line change.

  3. Four replan counters:
       N_trigger  = event condition fired
       N_attempt  = solver was called
       N_success  = solver returned a feasible plan
       N_failure  = solver was called but failed

  4. Explicit per-call target_threshold; never mutates solver global state.

Execution principle:
    a_t_exec = pi_t*[0]     (receding-horizon: only execute the first step)
    x_t+1    = T(x_t, c_t * a_t, xi_t)
    Replan if R_t = 1
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class MPCController:

    def __init__(
        self,
        risk_model,
        base_threshold: float,
        feature_contract: dict,
        solver_router=None,
        delta_safety: float = 0.05,
        gamma_stability: float = 0.5,
        delta_r: float = 0.03,    # risk deviation threshold for triggering replan
        delta_x: float = 0.10,    # normalised state-deviation threshold for replan
    ):
        self.risk_model       = risk_model
        self.base_threshold   = base_threshold
        self.delta_safety     = delta_safety
        self.feature_contract = feature_contract
        self.solver_router    = solver_router
        self.gamma_stability  = gamma_stability
        self.delta_r          = delta_r
        self.delta_x          = delta_x

        # Persistent plan state
        self.previous_plan_target: Optional[Dict[str, float]] = None

        # Expected trajectory: state we expect at the NEXT timestep under current plan
        self.expected_state_next: Optional[Dict[str, float]] = None
        self.expected_risk_next:  Optional[float]            = None

        # Feature scales (std of each actionable feature from training; set externally or lazily)
        self._feat_scales: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _actionable_features(self, state: pd.DataFrame) -> list:
        actionable_classes = (
            "CONDITIONALLY_ACTIONABLE", "ACTIONABLE_STATE", "ACTIONABLE_BEHAVIOUR"
        )
        return [
            f for f, d in self.feature_contract.items()
            if (d.actionable or d.feature_class in actionable_classes)
            and f in state.columns
        ]

    def _feat_scale(self, feat: str) -> float:
        """Return normalisation scale for feature (std proxy or 1.0)."""
        if self._feat_scales and feat in self._feat_scales:
            return max(self._feat_scales[feat], 1.0)
        # Fallback heuristic based on domain knowledge
        defaults = {
            "BUREAU_TOTAL_DEBT":  50_000.0,
            "BUREAU_MAX_OVERDUE": 10_000.0,
            "AMT_ANNUITY":        5_000.0,
            "BUREAU_ACTIVE_COUNT": 3.0,
        }
        return defaults.get(feat, 1.0)

    def _cost_weight(self, feat: str) -> float:
        d = self.feature_contract.get(feat)
        return d.cost_weight if d else 1.0

    def _normalised_distance(self, state_a: Dict[str, float],
                              state_b: Dict[str, float],
                              feats: list) -> float:
        """Weighted normalised L1 distance between two state dicts."""
        total = 0.0
        for f in feats:
            va = state_a.get(f, 0.0)
            vb = state_b.get(f, 0.0)
            s  = self._feat_scale(f)
            w  = self._cost_weight(f)
            total += w * abs(va - vb) / s
        return total

    def _state_to_dict(self, state: pd.DataFrame, feats: list) -> Dict[str, float]:
        return {f: float(state.iloc[0][f]) for f in feats if f in state.columns}

    # ------------------------------------------------------------------
    # Trigger condition   R_t = 0 or 1
    # ------------------------------------------------------------------

    def _needs_replan(
        self,
        current_state: pd.DataFrame,
        current_risk: float,
        policy_threshold: float,
        actionable: list,
    ) -> tuple[bool, str]:
        """
        Point-1 fix: compare observed (x_t, r_t) against the EXPECTED trajectory
        of the active plan (x_hat_t, r_hat_t), not raw threshold crossing.

        Replan triggers:
          A. No active plan yet.
          B. Policy / model changed since last step.
          C. |r_t - r_hat_t| > delta_r  (risk tracking error).
          D. d_norm(x_t, x_hat_t) > delta_x  (state tracking error).
        """
        # A. No plan yet
        if self.previous_plan_target is None:
            return True, "INITIAL_PLAN"

        # B. Policy change
        if policy_threshold != self.base_threshold:
            self.base_threshold = policy_threshold
            return True, "POLICY_CHANGE"

        # C. Risk tracking error
        if self.expected_risk_next is not None:
            risk_err = abs(current_risk - self.expected_risk_next)
            if risk_err > self.delta_r:
                return True, f"RISK_DEVIATION(|{current_risk:.3f}-{self.expected_risk_next:.3f}|={risk_err:.3f}>{self.delta_r})"

        # D. State tracking error
        if self.expected_state_next is not None:
            cur_dict   = self._state_to_dict(current_state, actionable)
            state_dist = self._normalised_distance(cur_dict, self.expected_state_next, actionable)
            if state_dist > self.delta_x:
                return True, f"STATE_DEVIATION(norm_dist={state_dist:.3f}>{self.delta_x})"

        return False, "ON_TRACK"

    # ------------------------------------------------------------------
    # Extract target from solver result
    # ------------------------------------------------------------------

    def _derive_target_from_result(
        self, res: dict, current_state: pd.DataFrame, actionable: list
    ) -> Optional[Dict[str, float]]:
        """
        Bug-1 fix: read new_state from solver result, not a non-existent 'roadmap' key.
        """
        new_state_dict = res.get("new_state")
        if not new_state_dict:
            return None
        target = {}
        for f in actionable:
            if f in new_state_dict:
                target[f] = float(new_state_dict[f])
            elif f in current_state.columns:
                target[f] = float(current_state.iloc[0][f])  # no change recommended
        return target if target else None

    # ------------------------------------------------------------------
    # Update expected trajectory after executing a_t
    # ------------------------------------------------------------------

    def _update_expected_trajectory(
        self, current_state: pd.DataFrame, action_t: Dict[str, float], actionable: list
    ):
        """
        Advance the expected state by the planned monthly action (perfect compliance).
        This defines x_hat_{t+1} and r_hat_{t+1}.
        """
        if self.previous_plan_target is None:
            self.expected_state_next = None
            self.expected_risk_next  = None
            return

        # x_hat_{t+1} = x_t + a_t  (perfect compliance assumption)
        next_expected = self._state_to_dict(current_state, actionable)
        for f, delta in action_t.items():
            if f in next_expected:
                next_expected[f] += delta

        self.expected_state_next = next_expected

        # r_hat_{t+1} = f(x_hat_{t+1})
        try:
            exp_df = current_state.copy()
            for f, v in next_expected.items():
                if f in exp_df.columns:
                    exp_df.iloc[0, exp_df.columns.get_loc(f)] = v
            self.expected_risk_next = float(self.risk_model.predict_risk(exp_df)[0])
        except Exception:
            self.expected_risk_next = None

    # ------------------------------------------------------------------
    # Normalised plan instability   I_t = sum_j w_j |dx_j / s_j|
    # ------------------------------------------------------------------

    def _plan_instability(self, new_target: Dict[str, float], actionable: list) -> float:
        """Point-3 fix: normalised, weighted L1 distance between plan targets."""
        if self.previous_plan_target is None:
            return 0.0
        return self._normalised_distance(new_target, self.previous_plan_target, actionable)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_action(
        self,
        current_state: pd.DataFrame,
        t: int,
        T_horizon: int,
        policy_threshold: float,
    ) -> Dict[str, Any]:
        """
        Executes one receding-horizon control step.

        Returns a dict with:
          risk_t            - current calibrated risk score
          replan_trigger    - bool: event condition fired (R_t = 1)
          replan_attempt    - bool: solver was actually called
          replan_success    - bool: solver returned a feasible new plan
          replan_failure    - bool: solver was called but failed
          reason            - string reason code
          action_t          - {feature: monthly_delta} to request this month
          plan_target       - {feature: absolute_target_value}
          instability_norm  - normalised weighted L1 between consecutive plan targets
        """
        actionable        = self._actionable_features(current_state)
        risk              = float(self.risk_model.predict_risk(current_state)[0])
        trigger, reason   = self._needs_replan(current_state, risk, policy_threshold, actionable)

        remaining_months  = max(1, T_horizon - t)
        instability_norm  = 0.0
        replan_attempt    = False
        replan_success    = False
        replan_failure    = False

        if trigger and self.solver_router is not None:
            replan_attempt = True
            target_tau = policy_threshold - self.delta_safety  # Bug-2 fix: explicit per-call

            res = self.solver_router.generate_recourse(
                current_state,
                target_threshold=target_tau,
                previous_plan=self.previous_plan_target,
                gamma_stability=self.gamma_stability,
            )

            if res.get("status") in ("success", "eligible"):
                new_target = self._derive_target_from_result(res, current_state, actionable)
                if new_target:
                    # Point-3: normalised instability
                    instability_norm = self._plan_instability(new_target, actionable)
                    self.previous_plan_target = new_target
                    replan_success = True
                else:
                    replan_failure = True
            else:
                replan_failure = True

        # Derive monthly action  a_t = (target - current) / remaining_months
        action_t = {}
        if self.previous_plan_target:
            for f, target_val in self.previous_plan_target.items():
                if f not in current_state.columns:
                    continue
                cur_val      = float(current_state.iloc[0][f])
                total_needed = target_val - cur_val
                if abs(total_needed) > 1e-9:
                    action_t[f] = total_needed / remaining_months

        # Advance expected trajectory for next step's trigger check
        self._update_expected_trajectory(current_state, action_t, actionable)

        return {
            "risk_t":           risk,
            "replan_trigger":   trigger,
            "replan_attempt":   replan_attempt,
            "replan_success":   replan_success,
            "replan_failure":   replan_failure,
            "reason":           reason,
            "action_t":         action_t,
            "plan_target":      self.previous_plan_target,
            "instability_norm": instability_norm,
        }
