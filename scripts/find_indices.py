import sys, os
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.models.risk_model import RiskModelAdapter
from backend.engine.constraint_registry import ConstraintRegistry
from backend.engine.feature_contract import FEATURE_CONTRACT_V3
from backend.engine.solver_router import SolverRouter

adapter = RiskModelAdapter()
adapter.load()

test_df = pd.read_csv("data/test_reference.csv").dropna(subset=["TARGET"])
train_df = pd.read_csv("data/train_reference.csv").dropna(subset=["TARGET"]).head(500)
registry = ConstraintRegistry()
router = SolverRouter(adapter, 0.30, registry, FEATURE_CONTRACT_V3, train_df)

risks = adapter.predict_risk(test_df)
above = test_df[risks > 0.30].reset_index(drop=True)

print("Scanning for feasible applicants quickly using just SLSQP...")
found = []
router._dice = None
for i in range(len(above)):
    app = above.iloc[[i]]
    res = router.generate_recourse(app, target_threshold=0.25)
    if res.get("status") in ("success", "eligible"):
        print(f"Found one! Index {i}")
        found.append(i)
        if len(found) >= 5:
            break
print("Done. Feasible indices:", found)
