import textwrap

content = textwrap.dedent('''
import copy
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

class MPCController:
    """
    Closed-loop orchestrator over a longitudinal trajectory.
    Implements receding horizon execution: a_t^{exec} = \pi_t^{*}[0].
    """
    def __init__(self, risk_model, base_threshold: float, feature_contract: dict, 
                 solver_router=None, delta_safety: float = 0.05, gamma_stability: float = 0.5):
        self.risk_model = risk_model
        self.base_threshold = base_threshold
        self.delta_safety = delta_safety
        self.feature_contract = feature_contract
        self.solver_router = solver_router
        self.gamma_stability = gamma_stability
        
        self.previous_plan_target = None
        
    def _needs_replan(self, state: pd.DataFrame, current_risk: float, policy_threshold: float, t: int) -> (bool, str):
        target_threshold = policy_threshold - self.delta_safety
        
        # 1. State deviation / Safety buffer violated
        if current_risk > target_threshold:
            return True, "STATE_DEVIATION"
            
        # 2. Policy change
        if policy_threshold != self.base_threshold:
            self.base_threshold = policy_threshold
            return True, "POLICY_CHANGE"
            
        # 3. If we don't have a plan yet
        if self.previous_plan_target is None:
            return True, "INITIAL_PLAN"
            
        # 4. Check if previous plan became infeasible - e.g. state drift made it physically impossible
        # (Could implement structural bounds check here, but for now we rely on risk > target)
        
        return False, "ON_TRACK"
        
    def get_action(self, current_state: pd.DataFrame, t: int, T_horizon: int, policy_threshold: float) -> Dict[str, Any]:
        """
        Executes one control step at time t.
        Returns:
            {
                'replan_triggered': bool,
                'reason': str,
                'action_t': dict (incremental delta for month t),
                'plan_target': dict (the final target state),
                'risk_t': float,
                'instability_L1': float
            }
        """
        risk = self.risk_model.predict_risk(current_state)[0]
        needs_replan, reason = self._needs_replan(current_state, risk, policy_threshold, t)
        
        instability = 0.0
        remaining_months = max(1, T_horizon - t)
        
        if needs_replan and self.solver_router is not None:
            orig_thresh = self.solver_router.threshold
            # Target safety margin
            self.solver_router.threshold = policy_threshold - self.delta_safety
            
            res = self.solver_router.generate_recourse(
                current_state, 
                previous_plan=self.previous_plan_target, 
                gamma_stability=self.gamma_stability
            )
            self.solver_router.threshold = orig_thresh
            
            if res.get('status') == 'success':
                new_roadmap = res.get('roadmap', {})
                new_target = {}
                # Calculate instability against previous plan target
                if self.previous_plan_target is not None:
                    for f in set(list(new_roadmap.keys()) + list(self.previous_plan_target.keys())):
                        val_new = new_roadmap.get(f, current_state.iloc[0].get(f, 0))
                        val_old = self.previous_plan_target.get(f, current_state.iloc[0].get(f, 0))
                        instability += abs(val_new - val_old)
                        
                # Re-calculate target states
                for f, offset in new_roadmap.items():
                    # The solver returns offset from current state. 
                    # new target state = current + offset
                    # BUT WAIT: In V3, SLSQP roadmap output was 'f': total_offset (where offset is usually negative for debt)
                    # Let's ensure it's a target state
                    orig_val = current_state.iloc[0].get(f, 0)
                    new_target[f] = orig_val + offset
                    
                self.previous_plan_target = new_target
            else:
                self.previous_plan_target = None
        else:
            reason = "ON_TRACK"
            
        # Now derive the incremental monthly action a_t
        action_t = {}
        if self.previous_plan_target:
            for f, target_val in self.previous_plan_target.items():
                orig_val = current_state.iloc[0].get(f, 0)
                total_needed = target_val - orig_val
                # Fractionate over remaining months
                if abs(total_needed) > 1e-9:
                    monthly_delta = total_needed / remaining_months
                    # For integers, we might want to round or handle carefully, 
                    # but the execution model expects floats and will handle.
                    action_t[f] = monthly_delta
                    
        return {
            'risk_t': risk,
            'replan_triggered': needs_replan,
            'reason': reason,
            'action_t': action_t,
            'plan_target': self.previous_plan_target,
            'instability_L1': instability
        }
''')

with open("backend/engine/mpc_controller.py", "w") as f:
    f.write(content)

print("Updated mpc_controller.py")
