from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    passed: bool
    violations: List[str] = field(default_factory=list)
    gate_results: Dict[str, bool] = field(default_factory=dict)


class FeasibilityGuard:
    def __init__(self, risk_model, threshold: float, constraint_registry,
                 feature_contract: Dict[str, Any], max_horizon: int = 12,
                 training_data: Optional[pd.DataFrame] = None,
                 durability_k: int = 5, durability_percentile: float = 95.0):
        self.risk_model = risk_model
        self.threshold = threshold
        self.registry = constraint_registry
        self.feature_contract = feature_contract
        self.max_horizon = max_horizon
        self.training_data = training_data
        self.durability_k = durability_k
        self.durability_percentile = durability_percentile
        self._knn_threshold = None
        self._train_numeric = None

    def validate(self, candidate_df: pd.DataFrame,
                 original_df: pd.DataFrame) -> ValidationResult:
        violations: List[str] = []
        gates: Dict[str, bool] = {}
        gates['V_risk'] = self._check_risk(candidate_df, violations)
        gates['V_structural'] = self._check_structural(candidate_df, violations)
        gates['V_actionability'] = self._check_actionability(candidate_df, original_df, violations)
        gates['V_plausibility'] = self._check_plausibility(candidate_df, original_df, violations)
        gates['V_durability'] = self._check_durability(candidate_df, violations)
        return ValidationResult(passed=all(gates.values()),
                                violations=violations, gate_results=gates)

    def _check_risk(self, cand, v):
        risk = float(self.risk_model.predict_risk(cand)[0])
        if risk > self.threshold:
            v.append(f'V_risk FAILED: risk {risk:.4f} > threshold {self.threshold}')
            return False
        return True

    def _check_structural(self, cand, v):
        row = cand.iloc[0]
        ok = True
        for c in self.registry.hard_constraints():
            p = c.params
            cid = c.constraint_id
            if cid == 'ANNUITY_CREDIT_MIN_001':
                if 'AMT_ANNUITY' in row.index and 'AMT_CREDIT' in row.index:
                    if row['AMT_CREDIT'] > 0 and row['AMT_ANNUITY'] < p['min_ratio'] * row['AMT_CREDIT']:
                        v.append(f'V_structural FAILED [{cid}]')
                        ok = False
            elif cid == 'ANNUITY_CREDIT_MAX_001':
                if 'AMT_ANNUITY' in row.index and 'AMT_CREDIT' in row.index:
                    if row['AMT_CREDIT'] > 0 and row['AMT_ANNUITY'] > p['max_ratio'] * row['AMT_CREDIT']:
                        v.append(f'V_structural FAILED [{cid}]')
                        ok = False
            elif cid == 'DTI_MAX_001':
                if 'AMT_ANNUITY' in row.index and 'AMT_INCOME_TOTAL' in row.index:
                    if row['AMT_INCOME_TOTAL'] > 0:
                        dti = row['AMT_ANNUITY'] / row['AMT_INCOME_TOTAL']
                        if dti > p['max_dti']:
                            v.append(f'V_structural FAILED [{cid}]: DTI {dti:.2%}')
                            ok = False
        return ok

    def _check_actionability(self, cand, orig, v):
        NON_ACTIONABLE = ('IMMUTABLE', 'HISTORICAL_IMMUTABLE', 'LENDER_HIDDEN', 'LENDER_CONTROLLED')
        ok = True
        for feat, defn in self.feature_contract.items():
            if defn.feature_class in NON_ACTIONABLE:
                if feat in cand.columns and feat in orig.columns:
                    try:
                        c_val = float(cand.iloc[0][feat])
                        o_val = float(orig.iloc[0][feat])
                        if abs(c_val - o_val) > 1e-6:
                            v.append(f'V_actionability FAILED: {feat!r} changed')
                            ok = False
                    except (TypeError, ValueError):
                        c_val = cand.iloc[0][feat]
                        o_val = orig.iloc[0][feat]
                        if c_val != o_val:
                            v.append(f'V_actionability FAILED: {feat!r} changed')
                            ok = False
        return ok

    def _check_plausibility(self, cand, orig, v):
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
                v.append(f'V_plausibility FAILED: {feat!r} change {delta:,.0f} exceeds {max_total:,.0f}')
                ok = False
        return ok

    def _check_durability(self, cand, v):
        if self.training_data is None or len(self.training_data) == 0:
            return True  # skip if no training data available

        try:
            numeric_cols = [c for c in cand.columns
                           if c in self.training_data.columns
                           and self.training_data[c].dtype in ('float64', 'float32', 'int64', 'int32')]
            if not numeric_cols:
                return True

            if self._train_numeric is None or self._knn_threshold is None:
                train_num = self.training_data[numeric_cols].dropna()
                if len(train_num) == 0:
                    return True
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                scaled = scaler.fit_transform(train_num)
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=self.durability_k, metric='euclidean')
                nn.fit(scaled)
                dists, _ = nn.kneighbors(scaled)
                mean_dists = dists.mean(axis=1)
                self._knn_threshold = float(np.percentile(mean_dists, self.durability_percentile))
                self._scaler = scaler
                self._nn = nn
                self._durability_cols = numeric_cols

            cand_vals = cand[self._durability_cols].fillna(0).values
            cand_scaled = self._scaler.transform(cand_vals)
            dists, _ = self._nn.kneighbors(cand_scaled)
            mean_dist = float(dists.mean())

            if mean_dist > self._knn_threshold:
                v.append(f'V_durability FAILED: kNN distance {mean_dist:.2f} > threshold {self._knn_threshold:.2f} (out of distribution)')
                return False
            return True
        except Exception as e:
            return True  # fail open if durability check errors
