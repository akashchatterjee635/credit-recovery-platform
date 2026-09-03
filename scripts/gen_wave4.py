import os, textwrap

files = {}

# ── base_solver.py ────────────────────────────────────────────────────────────
files['backend/engine/base_solver.py'] = textwrap.dedent("""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import pandas as pd


@dataclass
class RecourseResult:
    status: str              # success | failed | eligible | infeasible_within_horizon
    solver: str
    message: str
    original_risk: Optional[float] = None
    new_risk: Optional[float] = None
    cost: Optional[float] = None
    original_state: Optional[Dict[str, Any]] = None
    new_state: Optional[Dict[str, Any]] = None
    violations: List[str] = field(default_factory=list)
    gate_results: Dict[str, bool] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class BaseSolver(ABC):
    '''Abstract interface every solver must implement.'''
    solver_name: str = 'BaseSolver'

    @abstractmethod
    def generate_recourse(self, applicant: pd.DataFrame) -> RecourseResult:
        ...
""").lstrip()

# ── solvers/slsqp_solver.py ───────────────────────────────────────────────────
files['backend/engine/solvers/slsqp_solver.py'] = textwrap.dedent("""
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
""").lstrip()

# ── solvers/binary_search_solver.py ──────────────────────────────────────────
files['backend/engine/solvers/binary_search_solver.py'] = textwrap.dedent("""
'''
BinarySearchSolver
Best for single-feature monotonic cases (e.g., just reducing AMT_ANNUITY).
Searches along a single feature axis via bisection until risk <= threshold.
'''
from __future__ import annotations
import pandas as pd
import numpy as np
from backend.engine.base_solver import BaseSolver, RecourseResult
from backend.engine.validator import FeasibilityGuard
from backend.engine.constraint_registry import DEFAULT_REGISTRY
from backend.engine.feature_contract import FEATURE_CONTRACT_V2


class BinarySearchSolver(BaseSolver):
    solver_name = 'BinarySearchSolver'

    def __init__(self, risk_model, threshold: float = 0.3,
                 target_feature: str = 'AMT_ANNUITY',
                 registry=None, feature_contract=None,
                 n_iter: int = 50):
        self.risk_model = risk_model
        self.threshold = threshold
        self.target_feature = target_feature
        self.registry = registry or DEFAULT_REGISTRY
        self.feature_contract = feature_contract or FEATURE_CONTRACT_V2
        self.n_iter = n_iter
        self.guard = FeasibilityGuard(risk_model, threshold, self.registry,
                                      self.feature_contract, max_horizon=12)

    def generate_recourse(self, applicant: pd.DataFrame) -> RecourseResult:
        if self.risk_model.model is None:
            self.risk_model.load()

        current_risk = float(self.risk_model.predict_risk(applicant)[0])
        if current_risk <= self.threshold:
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
            if risk <= self.threshold:
                best_cand, best_risk = cand, risk
                lo = mid    # can we do less change?
            else:
                hi = mid

        if best_cand is None:
            return RecourseResult(status='failed', solver=self.solver_name,
                                  message=f'Binary search on {f!r} could not reach threshold.')

        vr = self.guard.validate(best_cand, applicant)
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
""").lstrip()

# ── solvers/dice_solver.py ────────────────────────────────────────────────────
files['backend/engine/solvers/dice_solver.py'] = textwrap.dedent("""
'''
DiCE-based solver (Diverse Counterfactual Explanations).
Designed for tree models where SLSQP gradients are unreliable.
Requires: pip install dice-ml
'''
from __future__ import annotations
import pandas as pd
import numpy as np
from backend.engine.base_solver import BaseSolver, RecourseResult
from backend.engine.validator import FeasibilityGuard
from backend.engine.constraint_registry import DEFAULT_REGISTRY
from backend.engine.feature_contract import FEATURE_CONTRACT_V2

try:
    import dice_ml
    DICE_AVAILABLE = True
except ImportError:
    DICE_AVAILABLE = False


class DiCESolver(BaseSolver):
    solver_name = 'DiCESolver'

    def __init__(self, risk_model, threshold: float = 0.3,
                 registry=None, feature_contract=None,
                 training_data: pd.DataFrame = None,
                 n_counterfactuals: int = 3):
        self.risk_model = risk_model
        self.threshold = threshold
        self.registry = registry or DEFAULT_REGISTRY
        self.feature_contract = feature_contract or FEATURE_CONTRACT_V2
        self.training_data = training_data
        self.n_cf = n_counterfactuals
        self.guard = FeasibilityGuard(risk_model, threshold, self.registry,
                                      self.feature_contract, max_horizon=12)
        self._dice_exp = None

    def _build_dice(self, applicant: pd.DataFrame):
        if not DICE_AVAILABLE:
            return None
        if self.training_data is None:
            return None

        actionable_classes = ('CONDITIONALLY_ACTIONABLE', 'ACTIONABLE_STATE', 'ACTIONABLE_BEHAVIOUR')
        features_to_vary = [f for f, d in self.feature_contract.items()
                            if d.feature_class in actionable_classes and f in applicant.columns]

        d = dice_ml.Data(
            dataframe=self.training_data,
            continuous_features=[f for f in features_to_vary
                                  if self.training_data[f].dtype != 'object'],
            outcome_name='TARGET'
        )

        class _SklearnWrapper:
            def __init__(self, adapter):
                self._adapter = adapter
            def predict(self, X):
                return (self._adapter.predict_risk(X) > 0.5).astype(int)
            def predict_proba(self, X):
                p = self._adapter.predict_risk(X)
                return np.column_stack([1-p, p])

        m = dice_ml.Model(model=_SklearnWrapper(self.risk_model), backend='sklearn')
        return dice_ml.Dice(d, m, method='random'), features_to_vary

    def generate_recourse(self, applicant: pd.DataFrame) -> RecourseResult:
        if not DICE_AVAILABLE:
            return RecourseResult(
                status='failed', solver=self.solver_name,
                message='dice-ml not installed. Run: pip install dice-ml')

        if self.risk_model.model is None:
            self.risk_model.load()

        current_risk = float(self.risk_model.predict_risk(applicant)[0])
        if current_risk <= self.threshold:
            return RecourseResult(status='eligible', solver=self.solver_name,
                                  message='Risk already below threshold.',
                                  original_risk=current_risk)

        built = self._build_dice(applicant)
        if built is None:
            return RecourseResult(status='failed', solver=self.solver_name,
                                  message='DiCE could not be initialised (training data missing).')

        dice_exp, features_to_vary = built
        try:
            cf_result = dice_exp.generate_counterfactuals(
                applicant, total_CFs=self.n_cf,
                desired_class=0,
                features_to_vary=features_to_vary,
            )
            cfs_df = cf_result.cf_examples_list[0].final_cfs_df
        except Exception as ex:
            return RecourseResult(status='failed', solver=self.solver_name,
                                  message=f'DiCE generation error: {ex}')

        best_result = None
        for _, row in cfs_df.iterrows():
            cand = applicant.copy()
            for col in features_to_vary:
                if col in row.index:
                    cand[col] = row[col]
            vr = self.guard.validate(cand, applicant)
            if vr.passed:
                new_risk = float(self.risk_model.predict_risk(cand)[0])
                cost = sum(
                    ((float(cand.iloc[0][f]) - float(applicant.iloc[0][f])) /
                     (abs(float(applicant.iloc[0][f])) or 1)) ** 2
                    for f in features_to_vary if f in cand.columns
                )
                best_result = RecourseResult(
                    status='success', solver=self.solver_name,
                    message='DiCE found a validated counterfactual.',
                    original_risk=current_risk, new_risk=new_risk, cost=cost,
                    original_state=applicant.to_dict(orient='records')[0],
                    new_state=cand.to_dict(orient='records')[0],
                    gate_results=vr.gate_results)
                break

        if best_result:
            return best_result
        return RecourseResult(status='failed', solver=self.solver_name,
                              message='DiCE generated counterfactuals but none passed validation.',
                              violations=['All DiCE candidates failed FeasibilityGuard'])
""").lstrip()

# ── solver_router.py ──────────────────────────────────────────────────────────
files['backend/engine/solver_router.py'] = textwrap.dedent("""
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
""").lstrip()

for path, content in files.items():
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(content)
    print(f'  wrote {path}')

print('Wave 4 files written.')
