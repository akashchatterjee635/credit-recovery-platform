'''
backend/engine/solver_router.py
Dispatcher that picks the appropriate solver tier based on the applicant profile.

Routing logic:
  1. Single-feature monotonic case  -> BinarySearchSolver (fast)
  2. Multi-feature continuous        -> SLSQPSolver (baseline)
  3. SLSQP fails validation          -> DiCESolver (tree-aware)
  4. All fail                        -> detailed failure report
'''
from __future__ import annotations
import numpy as np
import pandas as pd
from backend.engine.base_solver import RecourseResult
from backend.engine.solvers.slsqp_solver import SLSQPSolver
from backend.engine.solvers.binary_search_solver import BinarySearchSolver
from backend.engine.solvers.dice_solver import DiCESolver
from backend.engine.constraint_registry import DEFAULT_REGISTRY
from backend.engine.feature_contract import FEATURE_CONTRACT_V3


class SolverRouter:
    '''Routes each recourse request to the cheapest solver that can solve it.'''

    def __init__(self, risk_model, threshold: float = None,
                 registry=None, feature_contract=None,
                 training_data: pd.DataFrame = None):
        self.risk_model = risk_model
        self.registry = registry or DEFAULT_REGISTRY
        self.threshold = threshold if threshold is not None else self.registry.recourse_threshold()
        self.feature_contract = feature_contract or FEATURE_CONTRACT_V3
        self.training_data = training_data

        shared = dict(risk_model=risk_model, threshold=self.threshold,
                      registry=self.registry, feature_contract=self.feature_contract)

        self._binary = BinarySearchSolver(**shared)
        self._slsqp  = SLSQPSolver(**shared)
        self._dice   = DiCESolver(**shared, training_data=training_data)

    def _probe_monotonicity(self, applicant: pd.DataFrame, feature: str, n_probes: int = 10) -> bool:
        '''Samples risk at n_probes points between min_val and orig_val.
        Returns True only if diffs are all >= -1e-6 or all <= 1e-6 (monotonic).'''
        if self.risk_model.model is None:
            self.risk_model.load()

        defn = self.feature_contract.get(feature)
        min_val = float(defn.min_val) if (defn and defn.min_val is not None) else 0.0
        orig_val = float(applicant.iloc[0][feature])

        vals = np.linspace(min_val, orig_val, n_probes)
        cand_df = pd.concat([applicant.iloc[[0]]] * n_probes, ignore_index=True)
        cand_df[feature] = vals

        risks = np.asarray(self.risk_model.predict_risk(cand_df))
        diffs = np.diff(risks)
        if len(diffs) == 0:
            return True
        return bool(np.all(diffs >= -1e-6) or np.all(diffs <= 1e-6))

    def _is_single_monotonic(self, applicant: pd.DataFrame) -> bool:
        '''Checks if there is exactly 1 actionable feature and its response is monotonic.'''
        actionable_classes = ('CONDITIONALLY_ACTIONABLE', 'ACTIONABLE_STATE', 'ACTIONABLE_BEHAVIOUR')
        actionable = [f for f, d in self.feature_contract.items()
                      if (d.actionable or d.feature_class in actionable_classes) and f in applicant.columns]
        if len(actionable) != 1:
            return False
        return self._probe_monotonicity(applicant, actionable[0])

    def generate_recourse(self, applicant: pd.DataFrame, previous_plan: dict = None, gamma_stability: float = 0.0) -> dict:
        if self.risk_model.model is None:
            self.risk_model.load()

        actionable_classes = ('CONDITIONALLY_ACTIONABLE', 'ACTIONABLE_STATE', 'ACTIONABLE_BEHAVIOUR')
        actionable = [f for f, d in self.feature_contract.items()
                      if (d.actionable or d.feature_class in actionable_classes) and f in applicant.columns]

        tiers = []

        if len(actionable) == 1:
            feat = actionable[0]
            if self._probe_monotonicity(applicant, feat):
                self._binary.target_feature = feat
                tiers.append(('binary_search', self._binary))

        tiers.append(('slsqp', self._slsqp))
        tiers.append(('dice',  self._dice))

        attempted_tiers = []
        all_violations = []
        for tier_name, solver in tiers:
            attempted_tiers.append(tier_name)
            result = solver.generate_recourse(applicant)
            if result.status in ('success', 'eligible'):
                d = result.to_dict()
                d['solver_tier'] = tier_name
                d['tiers_attempted'] = list(attempted_tiers)
                return d
            all_violations.extend(result.violations or [f'[{tier_name}] {result.message}'])

        return {
            'status': 'failed',
            'message': 'All solver tiers exhausted. No feasible recourse found.',
            'tiers_attempted': attempted_tiers,
            'violations': all_violations,
            'new_state': None,
        }
