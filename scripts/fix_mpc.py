import textwrap, os

# ── Fix 1+2+3+4: Rewrite mpc_controller.py entirely ──────────────────────────
mpc = textwrap.dedent('''
    """
    backend/engine/mpc_controller.py

    Closed-loop receding-horizon MPC controller.
    Execution principle:  a_t_exec = pi_t*[0]
    Safety margin:        target_threshold = policy_threshold - delta_safety
    Stability penalty:    J_stability = gamma * ||target_new - target_prev||_1
    Replanning:           event-triggered (see _needs_replan)
    """
    from __future__ import annotations
    import copy
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
        ):
            self.risk_model       = risk_model
            self.base_threshold   = base_threshold  # tracks current known policy threshold
            self.delta_safety     = delta_safety
            self.feature_contract = feature_contract
            self.solver_router    = solver_router
            self.gamma_stability  = gamma_stability

            # Persistent plan state (absolute target values per feature)
            self.previous_plan_target: Optional[Dict[str, float]] = None

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

        def _needs_replan(
            self,
            current_risk: float,
            policy_threshold: float,
            t: int,
        ) -> tuple[bool, str]:
            target_threshold = policy_threshold - self.delta_safety

            # 1. Risk exceeds safety buffer
            if current_risk > target_threshold:
                return True, "STATE_DEVIATION"

            # 2. Policy changed since last step
            if policy_threshold != self.base_threshold:
                self.base_threshold = policy_threshold
                return True, "POLICY_CHANGE"

            # 3. No plan yet
            if self.previous_plan_target is None:
                return True, "INITIAL_PLAN"

            return False, "ON_TRACK"

        def _derive_target_from_result(
            self,
            res: dict,
            current_state: pd.DataFrame,
        ) -> Optional[Dict[str, float]]:
            """
            Bug-1 fix: extract feature targets from solver\'s new_state,
            NOT from a non-existent \'roadmap\' key.
            """
            new_state_dict = res.get("new_state")
            if not new_state_dict:
                return None

            actionable = self._actionable_features(current_state)
            target = {}
            for f in actionable:
                if f in new_state_dict:
                    target[f] = float(new_state_dict[f])
                elif f in current_state.columns:
                    # No change recommended for this feature
                    target[f] = float(current_state.iloc[0][f])
            return target if target else None

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
              replan_attempt    - bool: was a replan attempted?
              replan_success    - bool: did the solver find a new feasible plan?
              replan_failure    - bool: did the solver fail despite attempt?
              reason            - string reason code
              action_t          - {feature: monthly_delta} to request this month
              plan_target       - {feature: absolute_target_value}
              instability_L1    - L1 distance between new and previous plan targets
            """
            risk = self.risk_model.predict_risk(current_state)[0]
            needs_replan, reason = self._needs_replan(risk, policy_threshold, t)

            remaining_months = max(1, T_horizon - t)
            instability      = 0.0
            replan_attempt   = False
            replan_success   = False
            replan_failure   = False

            if needs_replan and self.solver_router is not None:
                replan_attempt = True

                # Bug-2 fix: pass target_threshold explicitly instead of mutating router.threshold
                target_threshold = policy_threshold - self.delta_safety

                # Bug-3 fix: pass previous_plan_target and gamma so SLSQP can use it
                res = self.solver_router.generate_recourse(
                    current_state,
                    target_threshold=target_threshold,
                    previous_plan=self.previous_plan_target,
                    gamma_stability=self.gamma_stability,
                )

                if res.get("status") in ("success", "eligible"):
                    # Bug-1 fix: read new_state, not roadmap
                    new_target = self._derive_target_from_result(res, current_state)

                    if new_target:
                        # Measure plan instability (L1 vs previous target)
                        if self.previous_plan_target is not None:
                            all_feats = set(new_target) | set(self.previous_plan_target)
                            for f in all_feats:
                                v_new = new_target.get(f, current_state.iloc[0].get(f, 0))
                                v_old = self.previous_plan_target.get(f, current_state.iloc[0].get(f, 0))
                                instability += abs(v_new - v_old)

                        self.previous_plan_target = new_target
                        replan_success = True
                    else:
                        replan_failure = True
                else:
                    replan_failure = True
                    # Keep stale plan if it exists
            # else: ON_TRACK, no replan needed

            # Derive monthly action a_t = (target - current) / remaining_months
            action_t = {}
            if self.previous_plan_target:
                for f, target_val in self.previous_plan_target.items():
                    if f not in current_state.columns:
                        continue
                    current_val = float(current_state.iloc[0][f])
                    total_needed = target_val - current_val
                    if abs(total_needed) > 1e-9:
                        action_t[f] = total_needed / remaining_months

            return {
                "risk_t":          risk,
                "replan_attempt":  replan_attempt,
                "replan_success":  replan_success,
                "replan_failure":  replan_failure,
                "reason":          reason,
                "action_t":        action_t,
                "plan_target":     self.previous_plan_target,
                "instability_L1":  instability,
            }
''').lstrip()

with open("backend/engine/mpc_controller.py", "w", encoding="utf-8") as f:
    f.write(mpc)
print("OK  mpc_controller.py")
