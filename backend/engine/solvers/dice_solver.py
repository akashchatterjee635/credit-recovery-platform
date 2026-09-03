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
