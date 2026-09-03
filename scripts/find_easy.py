import sys, os
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.models.risk_model import RiskModelAdapter

adapter = RiskModelAdapter()
adapter.load()

test_df = pd.read_csv("data/test_reference.csv").dropna(subset=["TARGET"])
risks = adapter.predict_risk(test_df)
above = test_df[risks > 0.30].reset_index(drop=True)

print("Finding easily fixable applicants...")
found = []
for i in range(len(above)):
    app = above.iloc[[i]].copy()
    # Try just blasting debt and overdue to 0, and halving annuity
    if "BUREAU_TOTAL_DEBT" in app.columns: app["BUREAU_TOTAL_DEBT"] = 0
    if "BUREAU_MAX_OVERDUE" in app.columns: app["BUREAU_MAX_OVERDUE"] = 0
    if "AMT_ANNUITY" in app.columns: app["AMT_ANNUITY"] /= 2.0
    
    new_risk = adapter.predict_risk(app)[0]
    if new_risk <= 0.25:
        print(f"Index {i} risk drops from {risks[risks>0.30][i]:.3f} to {new_risk:.3f}")
        found.append(i)
        if len(found) >= 5: break

print("Done. Indices:", found)
