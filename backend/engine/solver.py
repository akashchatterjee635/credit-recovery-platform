import pandas as pd
import numpy as np
from scipy.optimize import minimize
from backend.models.risk_model import RiskModelAdapter
from backend.engine.feature_contract import COMMON_FEATURE_CONTRACT

class CostAwareSolver:
    def __init__(self, risk_model: RiskModelAdapter, threshold: float = 0.3):
        self.risk_model = risk_model
        self.threshold = threshold
        
    def generate_recourse(self, original_applicant: pd.DataFrame) -> dict:
        if self.risk_model.model is None:
            self.risk_model.load()
            
        current_risk = self.risk_model.predict_risk(original_applicant)[0]
        if current_risk <= self.threshold:
            return {"status": "eligible", "message": "No recourse needed. Risk is already below threshold.", "new_state": None}
            
        # Extract recourse variables
        actionable_features = [f for f, d in COMMON_FEATURE_CONTRACT.items() if d.actionable]
        
        # Initial guess is the original values
        x0 = original_applicant[actionable_features].iloc[0].values
        
        # Define bounds based on feature contract
        bounds = []
        for f in actionable_features:
            min_v = COMMON_FEATURE_CONTRACT[f].min_val
            max_v = COMMON_FEATURE_CONTRACT[f].max_val
            bounds.append((min_v if min_v is not None else 0, max_v))
            
        def objective(x):
            cost = 0
            for i, f in enumerate(actionable_features):
                val = x[i]
                orig_val = x0[i]
                weight = COMMON_FEATURE_CONTRACT[f].cost_weight
                scale = abs(orig_val) if orig_val != 0 else 1.0
                cost += weight * ((val - orig_val)/scale)**2
            return cost
            
        def constraint_risk(x):
            candidate = original_applicant.copy()
            for i, f in enumerate(actionable_features):
                candidate[f] = x[i]
            risk = self.risk_model.predict_risk(candidate)[0]
            return self.threshold - risk
            
        cons = [{'type': 'ineq', 'fun': constraint_risk}]
        
        # Add Structural Constraints to enforce financial plausibility (Multicollinearity controls)
        if 'AMT_CREDIT' in actionable_features and 'AMT_ANNUITY' in actionable_features:
            idx_cred = actionable_features.index('AMT_CREDIT')
            idx_ann = actionable_features.index('AMT_ANNUITY')
            
            # Constraint 1: Annuity must be at least 3% of Total Credit Amount
            cons.append({'type': 'ineq', 'fun': lambda x: x[idx_ann] - 0.03 * x[idx_cred]})
            # Constraint 2: Annuity must be at most 10% of Total Credit Amount
            cons.append({'type': 'ineq', 'fun': lambda x: 0.10 * x[idx_cred] - x[idx_ann]})
            
        if 'AMT_INCOME_TOTAL' in actionable_features and 'AMT_ANNUITY' in actionable_features:
            idx_inc = actionable_features.index('AMT_INCOME_TOTAL')
            idx_ann = actionable_features.index('AMT_ANNUITY')
            
            # Constraint 3: Debt-to-Income (DTI) - Annuity should not exceed 40% of Total Income
            cons.append({'type': 'ineq', 'fun': lambda x: 0.40 * x[idx_inc] - x[idx_ann]})
        
        # Run optimization
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 50})
        
        if res.success or constraint_risk(res.x) >= 0:
            candidate = original_applicant.copy()
            for i, f in enumerate(actionable_features):
                candidate[f] = res.x[i]
            
            new_risk = self.risk_model.predict_risk(candidate)[0]
            return {
                "status": "success",
                "message": "Recourse path found with structural constraints enforced.",
                "original_risk": float(current_risk),
                "new_risk": float(new_risk),
                "original_state": original_applicant.to_dict(orient='records')[0],
                "new_state": candidate.to_dict(orient='records')[0],
                "cost": float(res.fun)
            }
        else:
            return {"status": "failed", "message": "Could not find a feasible recourse path within structural and risk constraints.", "new_state": None}
