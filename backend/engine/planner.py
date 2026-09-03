'''
backend/engine/planner.py
RecoveryTrajectoryPlanner
  – a trajectory DISCRETIZER, not an MPC closed-loop replanner.
  – converts a one-shot target state into a month-by-month action list.
  – respects monthly caps from the ConstraintRegistry.
  – returns infeasible_within_horizon if the required months exceed MAX_HORIZON.

NOTE: This will become a true closed-loop planner in a later phase when
borrower state refresh + risk re-scoring per month is implemented.
'''
import math
from datetime import datetime, timedelta
from backend.engine.constraint_registry import DEFAULT_REGISTRY


MAX_HORIZON = 12  # months


class RecoveryTrajectoryPlanner:
    def __init__(self, registry=None):
        self.registry = registry or DEFAULT_REGISTRY

    def generate_timeline(self, original_state: dict, target_state: dict) -> dict:
        deltas = {}
        months_required = 1

        for feat in list(original_state.keys()):
            if feat not in target_state:
                continue
            try:
                diff = float(target_state[feat]) - float(original_state[feat])
            except (TypeError, ValueError):
                continue
            if abs(diff) < 0.01:
                continue

            cap = self.registry.monthly_cap(feat)
            if cap and cap > 0:
                req = math.ceil(abs(diff) / cap)
                deltas[feat] = diff
                if req > months_required:
                    months_required = req

        # ── Horizon feasibility check (BUG FIX: no silent compression) ────────
        if months_required > MAX_HORIZON:
            return {
                'status': 'infeasible_within_horizon',
                'message': (
                    f'Recovery requires {months_required} months but max horizon '
                    f'is {MAX_HORIZON} months. The required changes exceed what is '
                    f'achievable without violating monthly capability constraints. '
                    f'Consider re-optimizing for a less aggressive target.'
                ),
                'months_required': months_required,
                'max_horizon': MAX_HORIZON,
                'total_months': None,
                'timeline': [],
            }

        timeline = []
        today = datetime.now()

        for m in range(1, months_required + 1):
            step_state = original_state.copy()
            step_actions = []

            for feat, diff in deltas.items():
                step_val = float(original_state[feat]) + diff * (m / months_required)
                prev_val = float(original_state[feat]) + diff * ((m - 1) / months_required)
                step_state[feat] = step_val
                monthly_delta = step_val - prev_val
                if abs(monthly_delta) > 0.01:
                    direction = 'Increase' if monthly_delta > 0 else 'Decrease'
                    step_actions.append({
                        'feature': feat,
                        'direction': direction,
                        'monthly_change': round(abs(monthly_delta), 2),
                        'cumulative_target': round(step_val, 2),
                        'label': f'{direction} {feat} by {abs(monthly_delta):,.2f} (target: {step_val:,.2f})',
                    })

            timeline.append({
                'month': m,
                'actions': step_actions,
                'intermediate_state': step_state,
                'reassessment_date': (today + timedelta(days=30 * m)).strftime('%Y-%m-%d'),
                'is_final': m == months_required,
            })

        return {
            'status': 'feasible',
            'total_months': months_required,
            'timeline': timeline,
        }


# Backwards-compatible alias
SequentialPlanner = RecoveryTrajectoryPlanner
