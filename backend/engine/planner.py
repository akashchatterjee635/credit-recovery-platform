import math
from datetime import datetime, timedelta

# Define maximum allowable monthly change to ensure plausibility (anti-gaming & durability)
MONTHLY_LIMITS = {
    'AMT_CREDIT': 50000.0,      # Max  credit change per month
    'AMT_INCOME_TOTAL': 5000.0, # Max  verifiable income increase per month
    'AMT_ANNUITY': 2000.0       # Max  annuity payment change per month
}

class SequentialPlanner:
    def __init__(self, limits=None):
        self.limits = limits or MONTHLY_LIMITS
        
    def generate_timeline(self, original_state: dict, target_state: dict) -> dict:
        """
        Converts a one-shot counterfactual into a receding-horizon sequential plan.
        """
        deltas = {}
        months_required = 1
        
        # Find the bottleneck feature to determine the timeline horizon
        for feature, max_change in self.limits.items():
            if feature in original_state and feature in target_state:
                diff = target_state[feature] - original_state[feature]
                # Only consider actionable financial features that changed
                if abs(diff) > 0.01:
                    deltas[feature] = diff
                    req_months = math.ceil(abs(diff) / max_change)
                    if req_months > months_required:
                        months_required = req_months
                        
        # Cap at a 12-month horizon for practical recovery
        if months_required > 12:
            months_required = 12
            
        timeline = []
        current_date = datetime.now()
        
        for m in range(1, months_required + 1):
            step_state = original_state.copy()
            step_actions = []
            
            for feature, diff in deltas.items():
                # Linear progression toward the target
                step_val = original_state[feature] + (diff * (m / months_required))
                prev_val = original_state[feature] + (diff * ((m-1) / months_required))
                step_state[feature] = step_val
                
                monthly_diff = step_val - prev_val
                
                # Only log meaningful changes
                if abs(monthly_diff) > 0.01:
                    direction = "Increase" if monthly_diff > 0 else "Decrease"
                    step_actions.append(f"**{direction}** {feature} by **\** (Target: \)")
                
            reassessment_date = current_date + timedelta(days=30 * m)
            
            timeline.append({
                "month": m,
                "actions": step_actions,
                "intermediate_state": step_state,
                "reassessment_date": reassessment_date.strftime("%Y-%m-%d"),
                "is_final": m == months_required
            })
            
        return {
            "total_months": months_required,
            "timeline": timeline
        }
