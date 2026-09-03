'''
SLSQP-based recourse solver.
Limitation: SLSQP expects smooth gradients; LightGBM is piecewise-constant.
This solver is kept as a BASELINE for comparison with DiCE and binary search.
'''
from __future__ import annotations
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from backend.engine.base_solver import BaseSolver, RecourseResult
from backend.engine.validator import FeasibilityGuard
from backend.engine.constraint_registry import DEFAULT_REGISTRY, ConstraintRegistry
from backend.engine.feature_contract import FEATURE_CONTRACT_V2


class SLSQPSolver(BaseSolver):
    solver_name = 'SLSQPSolver'

    def __init__(self, risk_model, threshold: float = 0.3,
                 registry: ConstraintRegistry = None,
                 feature_contract: dict = None):
        self.risk_model = risk_model
        self.threshold = threshold
        self.registry = registry or DEFAULT_REGISTRY
        self.feature_contract = feature_contract or FEATURE_CONTRACT_V2
        self.guard = FeasibilityGuard(risk_model, threshold, self.registry,
                                      self.feature_contract, max_horizon=12)

    def generate_recourse(self, applicant: pd.DataFrame) -> RecourseResult:
        if self.risk_model.model is None:
            self.risk_model.load()

        current_risk = float(self.risk_model.predict_risk(applicant)[0])
        if current_risk <= self.threshold:
            return RecourseResult(
                status='eligible', solver=self.solver_name,
                message='Risk already below threshold.',
                original_risk=current_risk)

        actionable_classes = ('CONDITIONALLY_ACTIONABLE', 'ACTIONABLE_STATE', 'ACTIONABLE_BEHAVIOUR')
        actionable = [f for f, d in self.feature_contract.items()
                      if d.feature_class in actionable_classes and f in applicant.columns]

        if not actionable:
            return RecourseResult(status='failed', solver=self.solver_name,
                                  message='No actionable features.')

        x0 = applicant[actionable].iloc[0].values.astype(float)
        bounds = [(self.feature_contract[f].min_val or 0,
                   self.feature_contract[f].max_val) for f in actionable]

        def objective(x):
            cost = 0.0
            for i, f in enumerate(actionable):
                orig, w = x0[i], self.feature_contract[f].cost_weight
                scale = abs(orig) if orig != 0 else 1.0
                cost += w * ((x[i] - orig) / scale) ** 2
            return cost

        def risk_con(x):
            cand = applicant.copy()
            for i, f in enumerate(actionable):
                cand[f] = x[i]
            return self.threshold - float(self.risk_model.predict_risk(cand)[0])

        cons = [{'type': 'ineq', 'fun': risk_con}]
        idx = {f: i for i, f in enumerate(actionable)}
        for c in self.registry.hard_constraints():
            p = c.params
            if c.constraint_id == 'ANNUITY_CREDIT_MIN_001':
                if 'AMT_ANNUITY' in idx and 'AMT_CREDIT' in idx:
                    ia, ic, r = idx['AMT_ANNUITY'], idx['AMT_CREDIT'], p['min_ratio']
                    cons.append({'type': 'ineq', 'fun': lambda x, ia=ia, ic=ic, r=r: x[ia] - r*x[ic]})
            elif c.constraint_id == 'ANNUITY_CREDIT_MAX_001':
                if 'AMT_ANNUITY' in idx and 'AMT_CREDIT' in idx:
                    ia, ic, r = idx['AMT_ANNUITY'], idx['AMT_CREDIT'], p['max_ratio']
                    cons.append({'type': 'ineq', 'fun': lambda x, ia=ia, ic=ic, r=r: r*x[ic] - x[ia]})
            elif c.constraint_id == 'DTI_MAX_001':
                if 'AMT_ANNUITY' in idx and 'AMT_INCOME_TOTAL' in idx:
                    ia, ii, r = idx['AMT_ANNUITY'], idx['AMT_INCOME_TOTAL'], p['max_dti']
                    cons.append({'type': 'ineq', 'fun': lambda x, ia=ia, ii=ii, r=r: r*x[ii] - x[ia]})

        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                       constraints=cons, options={'maxiter': 200, 'ftol': 1e-8})

        cand = applicant.copy()
        for i, f in enumerate(actionable):
            cand[f] = res.x[i]

        vr = self.guard.validate(cand, applicant)
        if vr.passed:
            return RecourseResult(
                status='success', solver=self.solver_name,
                message='All validation gates passed.',
                original_risk=current_risk,
                new_risk=float(self.risk_model.predict_risk(cand)[0]),
                cost=float(res.fun),
                original_state=applicant.to_dict(orient='records')[0],
                new_state=cand.to_dict(orient='records')[0],
                gate_results=vr.gate_results)
        return RecourseResult(
            status='failed', solver=self.solver_name,
            message='Solver converged but candidate failed validation.',
            violations=vr.violations, gate_results=vr.gate_results)
