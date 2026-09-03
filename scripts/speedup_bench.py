import re

path = "experiments/09_mpc_benchmark.py"
with open(path, "r", encoding="utf-8") as f:
    code = f.read()

# Disable DiCE in the benchmark's router to drastically speed up pre-filtering
code = code.replace(
    "shared_router = SolverRouter(",
    "shared_router = SolverRouter(\n        training_data=train_df, # dummy\n    )\n    shared_router._dice = None  # FAST PRE-FILTERING\n    shared_router_dummy = SolverRouter("
)

with open(path, "w", encoding="utf-8") as f:
    f.write(code)
print("Disabled DiCE in benchmark to speed up feasible applicant search.")
