import os, textwrap

files = {}

# ── constraint_registry.py ──────────────────────────────────────────────────
files['backend/engine/constraint_registry.py'] = textwrap.dedent("""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Constraint:
    constraint_id: str
    description: str
    constraint_type: str
    source: str
    confidence: str
    hard_or_soft: str
    params: Dict[str, Any] = field(default_factory=dict)


_REGISTRY: List[Constraint] = [
    Constraint('DTI_MAX_001',
               'Annuity <= 40% of gross income (DTI cap)',
               'structural', 'prototype_assumption', 'LOW', 'hard',
               {'max_dti': 0.40}),
    Constraint('ANNUITY_CREDIT_MIN_001',
               'Annuity >= 3% of credit amount',
               'structural', 'prototype_assumption', 'LOW', 'hard',
               {'min_ratio': 0.03}),
    Constraint('ANNUITY_CREDIT_MAX_001',
               'Annuity <= 10% of credit amount',
               'structural', 'prototype_assumption', 'LOW', 'hard',
               {'max_ratio': 0.10}),
    Constraint('MONTHLY_INCOME_CAP_001',
               'Max verifiable income change per month',
               'plausibility', 'prototype_assumption', 'LOW', 'hard',
               {'max_monthly_change': 5000.0}),
    Constraint('MONTHLY_CREDIT_CAP_001',
               'Max credit change per month',
               'plausibility', 'prototype_assumption', 'LOW', 'hard',
               {'max_monthly_change': 50000.0}),
    Constraint('MONTHLY_ANNUITY_CAP_001',
               'Max annuity change per month',
               'plausibility', 'prototype_assumption', 'LOW', 'hard',
               {'max_monthly_change': 2000.0}),
]


class ConstraintRegistry:
    def __init__(self, constraints=None):
        self._constraints = constraints or _REGISTRY

    def all_constraints(self):
        return self._constraints

    def hard_constraints(self):
        return [c for c in self._constraints if c.hard_or_soft == 'hard']

    def get(self, constraint_id):
        for c in self._constraints:
            if c.constraint_id == constraint_id:
                return c
        return None

    def monthly_cap(self, feature):
        mapping = {
            'AMT_INCOME_TOTAL': 'MONTHLY_INCOME_CAP_001',
            'AMT_CREDIT':       'MONTHLY_CREDIT_CAP_001',
            'AMT_ANNUITY':      'MONTHLY_ANNUITY_CAP_001',
        }
        cid = mapping.get(feature)
        c = self.get(cid) if cid else None
        return c.params['max_monthly_change'] if c else None

    def summary(self):
        rows = [f'{c.constraint_id:<30} [{c.confidence:<6}] [{c.hard_or_soft}] {c.description}'
                for c in self._constraints]
        return 'Constraint Registry\\n' + '='*80 + '\\n' + '\\n'.join(rows)


DEFAULT_REGISTRY = ConstraintRegistry()
""").lstrip()

# ── feature_contract.py (V2, 10-class taxonomy) ─────────────────────────────
files['backend/engine/feature_contract.py'] = textwrap.dedent("""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FeatureDefinition:
    name: str
    feature_class: str   # see 10-class taxonomy below
    actionable: bool     # True only for ACTIONABLE_* and CONDITIONALLY_ACTIONABLE
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    allowed_categories: Optional[List[str]] = None
    cost_weight: float = 1.0
    description: str = ''

# 10-class taxonomy
# IMMUTABLE                – birth date, gender
# HISTORICAL_IMMUTABLE     – past default event, CCJ
# ACTIONABLE_STATE         – outstanding bureau debt (borrower can pay it down)
# ACTIONABLE_BEHAVIOUR     – repayment consistency (borrower can improve it)
# CONDITIONALLY_ACTIONABLE – credit utilisation (borrower can reduce it)
# DERIVED                  – DTI ratio (computed, not directly changeable)
# TIME_EVOLVING            – employment tenure (changes with time automatically)
# PLANNING_ONLY            – income target (a goal state, not a direct action)
# LENDER_HIDDEN            – fraud flags (not shown to borrower)
# LENDER_CONTROLLED        – interest rate, product terms

FEATURE_CONTRACT_V2 = {
    # ── Immutable ────────────────────────────────────────────────────────────
    'DAYS_BIRTH': FeatureDefinition(
        'DAYS_BIRTH', 'IMMUTABLE', False,
        description='Age in days (negative). Cannot be changed.'),

    # ── Time-evolving (improves automatically, not actionable directly) ───────
    'DAYS_EMPLOYED': FeatureDefinition(
        'DAYS_EMPLOYED', 'TIME_EVOLVING', False,
        description='Days at current employer. Grows with tenure; not directly actionable.'),

    # ── Immutable (credential, hard to change) ───────────────────────────────
    'NAME_EDUCATION_TYPE': FeatureDefinition(
        'NAME_EDUCATION_TYPE', 'IMMUTABLE', False,
        description='Highest education level. Treated as immutable for recourse purposes.'),

    # ── Lender-controlled (product terms set by lender, not borrower) ─────────
    'AMT_CREDIT': FeatureDefinition(
        'AMT_CREDIT', 'LENDER_CONTROLLED', False,
        min_val=10000, cost_weight=0.5,
        description='Total credit amount. Set by the lender, not a borrower action.'),

    # ── Planning-only (income is a TARGET STATE, not a direct action) ─────────
    'AMT_INCOME_TOTAL': FeatureDefinition(
        'AMT_INCOME_TOTAL', 'PLANNING_ONLY', False,
        min_val=0, cost_weight=2.0,
        description='Total income. A target state; borrower cannot directly execute '
                    'an income increase — it represents a planning goal.'),

    # ── Conditionally actionable (borrower can restructure repayment schedule) ─
    'AMT_ANNUITY': FeatureDefinition(
        'AMT_ANNUITY', 'CONDITIONALLY_ACTIONABLE', True,
        min_val=0, cost_weight=1.0,
        description='Monthly annuity payment. Borrower can negotiate repayment schedule '
                    'within structural bounds (3-10% of credit).'),
}

# Backwards-compatible alias used by legacy code
COMMON_FEATURE_CONTRACT = FEATURE_CONTRACT_V2
""").lstrip()

# ── validator.py ─────────────────────────────────────────────────────────────
files['backend/engine/validator.py'] = textwrap.dedent("""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    passed: bool
    violations: List[str] = field(default_factory=list)
    gate_results: Dict[str, bool] = field(default_factory=dict)


class FeasibilityGuard:
    '''
    Post-solver validator. A candidate x* is accepted ONLY if ALL 4 gates pass,
    independent of whether the optimizer reported success.

    Gates:
      V_risk         : predicted risk <= threshold
      V_structural   : all hard constraint-registry rules satisfied
      V_actionability: no immutable/lender-controlled/hidden feature changed
      V_plausibility : no feature changed beyond max_horizon * monthly_cap
    '''

    def __init__(self, risk_model, threshold: float, constraint_registry,
                 feature_contract: Dict[str, Any], max_horizon: int = 12):
        self.risk_model = risk_model
        self.threshold = threshold
        self.registry = constraint_registry
        self.feature_contract = feature_contract
        self.max_horizon = max_horizon

    def validate(self, candidate_df: pd.DataFrame,
                 original_df: pd.DataFrame) -> ValidationResult:
        violations: List[str] = []
        gates: Dict[str, bool] = {}

        gates['V_risk']          = self._check_risk(candidate_df, violations)
        gates['V_structural']    = self._check_structural(candidate_df, violations)
        gates['V_actionability'] = self._check_actionability(candidate_df, original_df, violations)
        gates['V_plausibility']  = self._check_plausibility(candidate_df, original_df, violations)

        return ValidationResult(passed=all(gates.values()),
                                violations=violations, gate_results=gates)

    # ── private gates ────────────────────────────────────────────────────────

    def _check_risk(self, cand: pd.DataFrame, v: list) -> bool:
        risk = float(self.risk_model.predict_risk(cand)[0])
        if risk > self.threshold:
            v.append(f'V_risk FAILED: risk {risk:.4f} > threshold {self.threshold}')
            return False
        return True

    def _check_structural(self, cand: pd.DataFrame, v: list) -> bool:
        row = cand.iloc[0]
        ok = True
        for c in self.registry.hard_constraints():
            p = c.params
            cid = c.constraint_id
            if cid == 'ANNUITY_CREDIT_MIN_001':
                if 'AMT_ANNUITY' in row.index and 'AMT_CREDIT' in row.index:
                    if row['AMT_CREDIT'] > 0 and row['AMT_ANNUITY'] < p['min_ratio'] * row['AMT_CREDIT']:
                        v.append(f'V_structural FAILED [{cid}]: annuity {row[\"AMT_ANNUITY\"]:.0f} '
                                 f'< {p[\"min_ratio\"]*100:.0f}% of credit {row[\"AMT_CREDIT\"]:.0f}')
                        ok = False
            elif cid == 'ANNUITY_CREDIT_MAX_001':
                if 'AMT_ANNUITY' in row.index and 'AMT_CREDIT' in row.index:
                    if row['AMT_CREDIT'] > 0 and row['AMT_ANNUITY'] > p['max_ratio'] * row['AMT_CREDIT']:
                        v.append(f'V_structural FAILED [{cid}]: annuity {row[\"AMT_ANNUITY\"]:.0f} '
                                 f'> {p[\"max_ratio\"]*100:.0f}% of credit {row[\"AMT_CREDIT\"]:.0f}')
                        ok = False
            elif cid == 'DTI_MAX_001':
                if 'AMT_ANNUITY' in row.index and 'AMT_INCOME_TOTAL' in row.index:
                    if row['AMT_INCOME_TOTAL'] > 0:
                        dti = row['AMT_ANNUITY'] / row['AMT_INCOME_TOTAL']
                        if dti > p['max_dti']:
                            v.append(f'V_structural FAILED [{cid}]: DTI {dti:.2%} > {p[\"max_dti\"]:.0%}')
                            ok = False
        return ok

    def _check_actionability(self, cand: pd.DataFrame, orig: pd.DataFrame, v: list) -> bool:
        NON_ACTIONABLE = ('IMMUTABLE', 'HISTORICAL_IMMUTABLE',
                          'LENDER_HIDDEN', 'LENDER_CONTROLLED')
        ok = True
        for feat, defn in self.feature_contract.items():
            if defn.feature_class in NON_ACTIONABLE:
                if feat in cand.columns and feat in orig.columns:
                    c_val = float(cand.iloc[0][feat]) if not isinstance(cand.iloc[0][feat], str) else cand.iloc[0][feat]
                    o_val = float(orig.iloc[0][feat]) if not isinstance(orig.iloc[0][feat], str) else orig.iloc[0][feat]
                    changed = (c_val != o_val)
                    if changed:
                        v.append(f'V_actionability FAILED: non-actionable feature '
                                 f'{feat!r} changed ({o_val} -> {c_val})')
                        ok = False
        return ok

    def _check_plausibility(self, cand: pd.DataFrame, orig: pd.DataFrame, v: list) -> bool:
        ok = True
        for feat in cand.columns:
            cap = self.registry.monthly_cap(feat)
            if cap is None:
                continue
            max_total = cap * self.max_horizon
            try:
                delta = abs(float(cand.iloc[0][feat]) - float(orig.iloc[0][feat]))
            except (TypeError, ValueError):
                continue
            if delta > max_total:
                v.append(f'V_plausibility FAILED: {feat!r} change {delta:,.0f} '
                         f'exceeds {max_total:,.0f} ({self.max_horizon}mo cap)')
                ok = False
        return ok
""").lstrip()

# ── planner.py (fixed + renamed) ─────────────────────────────────────────────
files['backend/engine/planner.py'] = textwrap.dedent("""
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
""").lstrip()

# ── main.py (fixed API terminology) ──────────────────────────────────────────
files['backend/main.py'] = textwrap.dedent("""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from backend.models.risk_model import RiskModelAdapter
from backend.engine.solver import CostAwareSolver
from backend.engine.planner import RecoveryTrajectoryPlanner
from backend.engine.constraint_registry import DEFAULT_REGISTRY
from backend.engine.feature_contract import FEATURE_CONTRACT_V2

app = FastAPI(title='Credit Recovery Intelligence API v2')

# ── Startup ────────────────────────────────────────────────────────────────
try:
    _risk_adapter = RiskModelAdapter()
    _risk_adapter.load()
    _solver = CostAwareSolver(_risk_adapter, threshold=0.3,
                              registry=DEFAULT_REGISTRY,
                              feature_contract=FEATURE_CONTRACT_V2)
    _planner = RecoveryTrajectoryPlanner(registry=DEFAULT_REGISTRY)
except Exception as e:
    print(f'WARNING: startup failed – {e}')
    _risk_adapter = _solver = _planner = None


def _risk_band(score: float) -> str:
    if score < 0.20:  return 'LOW'
    if score < 0.30:  return 'MODERATE'
    if score < 0.50:  return 'ELEVATED'
    return 'HIGH'


class ApplicantData(BaseModel):
    AMT_CREDIT: float
    AMT_INCOME_TOTAL: float
    AMT_ANNUITY: float
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    NAME_EDUCATION_TYPE: str


@app.get('/')
def root():
    return {'service': 'Credit Recovery Intelligence API v2', 'status': 'running'}


@app.get('/constraints')
def list_constraints():
    return {'constraints': [
        {'id': c.constraint_id, 'description': c.description,
         'confidence': c.confidence, 'hard_or_soft': c.hard_or_soft,
         'params': c.params}
        for c in DEFAULT_REGISTRY.all_constraints()
    ]}


@app.post('/predict')
def predict_risk(applicant: ApplicantData):
    if _solver is None:
        raise HTTPException(503, 'Risk model not loaded.')
    df = pd.DataFrame([applicant.model_dump()])
    score = float(_risk_adapter.predict_risk(df)[0])
    band = _risk_band(score)
    return {
        'predicted_default_risk': round(score, 4),
        'risk_band': band,
        'recovery_assessment_applicable': score > _solver.threshold,
        'threshold_used': _solver.threshold,
    }


@app.post('/generate_roadmap')
def generate_roadmap(applicant: ApplicantData):
    if _solver is None or _planner is None:
        raise HTTPException(503, 'Risk model or planner not loaded.')
    df = pd.DataFrame([applicant.model_dump()])
    result = _solver.generate_recourse(df)

    if result['status'] == 'success':
        plan = _planner.generate_timeline(result['original_state'], result['new_state'])
        result['sequential_plan'] = plan
        result['constraint_registry_version'] = 'v1.0'
        result['solver_version'] = 'SLSQPSolver-v1'

    return result
""").lstrip()

# ── solver.py (wired with FeasibilityGuard + registry) ───────────────────────
files['backend/engine/solver.py'] = textwrap.dedent("""
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
""").lstrip()

for path, content in files.items():
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(content)
    print(f'  wrote {path}')

print('All Wave 1+2 files written.')
