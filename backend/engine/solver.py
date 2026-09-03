'''
backend/engine/solver.py
SLSQPSolver – wrapped with FeasibilityGuard for full post-solve validation.
NOTE: SLSQP + tree model is acknowledged as a baseline only.
      The Solver Router (Wave 4) will add DiCE and binary-search tiers.
'''
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from backend.models.risk_model import RiskModelAdapter
from backend.engine.feature_contract import FEATURE_CONTRACT_V2, COMMON_FEATURE_CONTRACT
from backend.engine.constraint_registry import DEFAULT_REGISTRY, ConstraintRegistry
from backend.engine.validator import FeasibilityGuard


class CostAwareSolver:
    def __init__(self, risk_model: RiskModelAdapter, threshold: float = 0.3,
                 registry: ConstraintRegistry = None,
                 feature_contract: dict = None):
        self.risk_model = risk_model
        self.threshold = threshold
        self.registry = registry or DEFAULT_REGISTRY
        self.feature_contract = feature_contract or FEATURE_CONTRACT_V2
        self.guard = FeasibilityGuard(
            risk_model=risk_model,
            threshold=threshold,
            constraint_registry=self.registry,
            feature_contract=self.feature_contract,
            max_horizon=12,
        )

    def generate_recourse(self, original_applicant: pd.DataFrame) -> dict:
        if self.risk_model.model is None:
            self.risk_model.load()

        current_risk = float(self.risk_model.predict_risk(original_applicant)[0])
        if current_risk <= self.threshold:
            return {
                'status': 'eligible',
                'message': 'Risk already below threshold – no recourse needed.',
                'predicted_default_risk': current_risk,
                'risk_band': self._band(current_risk),
                'new_state': None,
            }

        # Only CONDITIONALLY_ACTIONABLE features are solver variables
        actionable_classes = ('CONDITIONALLY_ACTIONABLE', 'ACTIONABLE_STATE', 'ACTIONABLE_BEHAVIOUR')
        actionable_features = [
            f for f, d in self.feature_contract.items()
            if d.feature_class in actionable_classes
            and f in original_applicant.columns
        ]

        if not actionable_features:
            return {
                'status': 'failed',
                'message': 'No actionable features available for recourse.',
                'new_state': None,
            }

        x0 = original_applicant[actionable_features].iloc[0].values.astype(float)

        bounds = []
        for f in actionable_features:
            defn = self.feature_contract[f]
            bounds.append((defn.min_val if defn.min_val is not None else 0,
                           defn.max_val))

        def objective(x):
            cost = 0.0
            for i, f in enumerate(actionable_features):
                orig = x0[i]
                w = self.feature_contract[f].cost_weight
                scale = abs(orig) if orig != 0 else 1.0
                cost += w * ((x[i] - orig) / scale) ** 2
            return cost

        def constraint_risk(x):
            cand = original_applicant.copy()
            for i, f in enumerate(actionable_features):
                cand[f] = x[i]
            return self.threshold - float(self.risk_model.predict_risk(cand)[0])

        cons = [{'type': 'ineq', 'fun': constraint_risk}]

        # Structural constraints from registry
        feat_idx = {f: i for i, f in enumerate(actionable_features)}
        for c in self.registry.hard_constraints():
            if c.constraint_id == 'ANNUITY_CREDIT_MIN_001':
                if 'AMT_ANNUITY' in feat_idx and 'AMT_CREDIT' in feat_idx:
                    ia, ic = feat_idx['AMT_ANNUITY'], feat_idx['AMT_CREDIT']
                    r = c.params['min_ratio']
                    cons.append({'type': 'ineq', 'fun': lambda x, ia=ia, ic=ic, r=r: x[ia] - r * x[ic]})
            elif c.constraint_id == 'ANNUITY_CREDIT_MAX_001':
                if 'AMT_ANNUITY' in feat_idx and 'AMT_CREDIT' in feat_idx:
                    ia, ic = feat_idx['AMT_ANNUITY'], feat_idx['AMT_CREDIT']
                    r = c.params['max_ratio']
                    cons.append({'type': 'ineq', 'fun': lambda x, ia=ia, ic=ic, r=r: r * x[ic] - x[ia]})
            elif c.constraint_id == 'DTI_MAX_001':
                if 'AMT_ANNUITY' in feat_idx and 'AMT_INCOME_TOTAL' in feat_idx:
                    ia, ii = feat_idx['AMT_ANNUITY'], feat_idx['AMT_INCOME_TOTAL']
                    r = c.params['max_dti']
                    cons.append({'type': 'ineq', 'fun': lambda x, ia=ia, ii=ii, r=r: r * x[ii] - x[ia]})

        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                       constraints=cons, options={'maxiter': 200, 'ftol': 1e-8})

        # Build candidate df
        candidate = original_applicant.copy()
        for i, f in enumerate(actionable_features):
            candidate[f] = res.x[i]

        # ── Full 4-gate validation (replaces old success-or-risk-only check) ──
        validation = self.guard.validate(candidate, original_applicant)

        if validation.passed:
            new_risk = float(self.risk_model.predict_risk(candidate)[0])
            return {
                'status': 'success',
                'solver': 'SLSQP',
                'message': 'Feasible recourse path found. All validation gates passed.',
                'original_risk': current_risk,
                'new_risk': new_risk,
                'original_state': original_applicant.to_dict(orient='records')[0],
                'new_state': candidate.to_dict(orient='records')[0],
                'cost': float(res.fun),
                'validation_gates': validation.gate_results,
            }
        else:
            return {
                'status': 'failed',
                'solver': 'SLSQP',
                'message': 'Solver converged but candidate failed validation.',
                'violations': validation.violations,
                'gate_results': validation.gate_results,
                'new_state': None,
            }

    @staticmethod
    def _band(score: float) -> str:
        if score < 0.20: return 'LOW'
        if score < 0.30: return 'MODERATE'
        if score < 0.50: return 'ELEVATED'
        return 'HIGH'
