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
                        v.append(f'V_structural FAILED [{cid}]: annuity {row["AMT_ANNUITY"]:.0f} '
                                 f'< {p["min_ratio"]*100:.0f}% of credit {row["AMT_CREDIT"]:.0f}')
                        ok = False
            elif cid == 'ANNUITY_CREDIT_MAX_001':
                if 'AMT_ANNUITY' in row.index and 'AMT_CREDIT' in row.index:
                    if row['AMT_CREDIT'] > 0 and row['AMT_ANNUITY'] > p['max_ratio'] * row['AMT_CREDIT']:
                        v.append(f'V_structural FAILED [{cid}]: annuity {row["AMT_ANNUITY"]:.0f} '
                                 f'> {p["max_ratio"]*100:.0f}% of credit {row["AMT_CREDIT"]:.0f}')
                        ok = False
            elif cid == 'DTI_MAX_001':
                if 'AMT_ANNUITY' in row.index and 'AMT_INCOME_TOTAL' in row.index:
                    if row['AMT_INCOME_TOTAL'] > 0:
                        dti = row['AMT_ANNUITY'] / row['AMT_INCOME_TOTAL']
                        if dti > p['max_dti']:
                            v.append(f'V_structural FAILED [{cid}]: DTI {dti:.2%} > {p["max_dti"]:.0%}')
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
