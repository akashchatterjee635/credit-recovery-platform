import os

path = "experiments/09_mpc_benchmark.py"
with open(path, "r") as f:
    content = f.read()

content = content.replace("adapter = RiskModelAdapter()", "print('init adapter...', flush=True)\n    adapter = RiskModelAdapter()")
content = content.replace("adapter.load()", "print('loading adapter...', flush=True)\n    adapter.load()")
content = content.replace("test_df = pd.read_csv", "print('loading test_df...', flush=True)\n    test_df = pd.read_csv")
content = content.replace("risks = adapter.predict_risk", "print('predicting risks...', flush=True)\n    risks = adapter.predict_risk")
content = content.replace("registry = ConstraintRegistry()", "print('init registry...', flush=True)\n    registry = ConstraintRegistry()")
content = content.replace("results = {'sequential'", "print('starting loop...', flush=True)\n    results = {'sequential'")

with open(path, "w") as f:
    f.write(content)
