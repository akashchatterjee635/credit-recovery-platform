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
import pandas as pd
from backend.engine.base_solver import RecourseResult
from backend.engine.solvers.slsqp_solver import SLSQPSolver
from backend.engine.solvers.binary_search_solver import BinarySearchSolver
from backend.engine.solvers.dice_solver import DiCESolver
from backend.engine.constraint_registry import DEFAULT_REGISTRY
from backend.engine.feature_contract import FEATURE_CONTRACT_V2


class SolverRouter:
    '''Routes each recourse request to the cheapest solver that can solve it.'''

    def __init__(self, risk_model, threshold: float = 0.3,
                 registry=None, feature_contract=None,
                 training_data: pd.DataFrame = None):
        self.risk_model = risk_model
        self.threshold = threshold
        self.registry = registry or DEFAULT_REGISTRY
        self.feature_contract = feature_contract or FEATURE_CONTRACT_V2
        self.training_data = training_data

        shared = dict(risk_model=risk_model, threshold=threshold,
                      registry=self.registry, feature_contract=self.feature_contract)

        self._binary = BinarySearchSolver(**shared)
        self._slsqp  = SLSQPSolver(**shared)
        self._dice   = DiCESolver(**shared, training_data=training_data)

    def _is_single_monotonic(self, applicant: pd.DataFrame) -> bool:
        '''Heuristic: only one actionable feature changes risk monotonically.'''
        actionable_classes = ('CONDITIONALLY_ACTIONABLE', 'ACTIONABLE_STATE')
        actionable = [f for f, d in self.feature_contract.items()
                      if d.feature_class in actionable_classes and f in applicant.columns]
        return len(actionable) == 1

    def generate_recourse(self, applicant: pd.DataFrame) -> dict:
        if self.risk_model.model is None:
            self.risk_model.load()

        tiers = []

        if self._is_single_monotonic(applicant):
            tiers.append(('binary_search', self._binary))

        tiers.append(('slsqp', self._slsqp))
        tiers.append(('dice',  self._dice))

        all_violations = []
        for tier_name, solver in tiers:
            result = solver.generate_recourse(applicant)
            if result.status == 'success':
                d = result.to_dict()
                d['solver_tier'] = tier_name
                d['tiers_attempted'] = [t for t, _ in tiers[:tiers.index((tier_name, solver))+1]]
                return d
            all_violations.extend(result.violations or [f'[{tier_name}] {result.message}'])

        return {
            'status': 'failed',
            'message': 'All solver tiers exhausted. No feasible recourse found.',
            'tiers_attempted': [t for t, _ in tiers],
            'violations': all_violations,
            'new_state': None,
        }
