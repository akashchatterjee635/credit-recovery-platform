"""
BinarySearchSolver
Best for single-feature monotonic cases (e.g., just reducing AMT_ANNUITY).
Searches along a single feature axis via bisection until risk <= threshold.

Bug-5 fix: effective_threshold is resolved per-call using target_threshold arg,
           never referencing an undefined variable. FeasibilityGuard created per-call.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from backend.engine.base_solver import BaseSolver, RecourseResult
from backend.engine.validator import FeasibilityGuard
from backend.engine.constraint_registry import DEFAULT_REGISTRY
from backend.engine.feature_contract import FEATURE_CONTRACT_V3


class BinarySearchSolver(BaseSolver):
    solver_name = 'BinarySearchSolver'

    def __init__(self, risk_model, threshold: float = None,
                 target_feature: str = 'AMT_ANNUITY',
                 registry=None, feature_contract=None,
                 n_iter: int = 50):
        self.risk_model = risk_model
        self.registry = registry or DEFAULT_REGISTRY
        self.threshold = threshold if threshold is not None else self.registry.recourse_threshold()
        self.target_feature = target_feature
        self.feature_contract = feature_contract or FEATURE_CONTRACT_V3
        self.n_iter = n_iter

    def generate_recourse(self, applicant: pd.DataFrame,
                          target_threshold: float = None,
                          **kwargs) -> RecourseResult:
        # Bug-5 fix: clean per-call resolution, no reference to undefined _threshold
        effective_threshold = target_threshold if target_threshold is not None else self.threshold

        if self.risk_model.model is None:
            self.risk_model.load()

        current_risk = float(self.risk_model.predict_risk(applicant)[0])
        if current_risk <= effective_threshold:
            return RecourseResult(status='eligible', solver=self.solver_name,
                                  message='Risk already below threshold.',
                                  original_risk=current_risk)

        f = self.target_feature
        orig_val = float(applicant.iloc[0][f])
        defn = self.feature_contract.get(f)
        lo = defn.min_val if defn and defn.min_val is not None else 0.0
        hi = orig_val

        best_cand, best_risk = None, current_risk
        for _ in range(self.n_iter):
            mid = (lo + hi) / 2.0
            cand = applicant.copy()
            cand[f] = mid
            risk = float(self.risk_model.predict_risk(cand)[0])
            if risk <= effective_threshold:
                best_cand, best_risk = cand, risk
                lo = mid    # can we do less change?
            else:
                hi = mid

        if best_cand is None:
            return RecourseResult(status='failed', solver=self.solver_name,
                                  message=f'Binary search on {f!r} could not reach threshold.')

        # Bug-5: FeasibilityGuard created per-call with effective_threshold
        guard = FeasibilityGuard(self.risk_model, effective_threshold, self.registry,
                                 self.feature_contract, max_horizon=12)
        vr = guard.validate(best_cand, applicant)
        cost = ((float(best_cand.iloc[0][f]) - orig_val) / (abs(orig_val) or 1)) ** 2
        if vr.passed:
            return RecourseResult(
                status='success', solver=self.solver_name,
                message=f'Binary search on {f!r} succeeded.',
                original_risk=current_risk, new_risk=best_risk, cost=cost,
                original_state=applicant.to_dict(orient='records')[0],
                new_state=best_cand.to_dict(orient='records')[0],
                gate_results=vr.gate_results)
        return RecourseResult(status='failed', solver=self.solver_name,
                              message='Candidate failed validation.',
                              violations=vr.violations, gate_results=vr.gate_results)
