import os

path = "backend/engine/solvers/slsqp_solver.py"
with open(path, "r") as f:
    content = f.read()

content = content.replace("def risk_constraint(x_opt):", "count = [0]\n        def risk_constraint(x_opt):\n            count[0] += 1\n            if count[0] % 100 == 0: print(f'Evaluations: {count[0]}', flush=True)")

with open(path, "w") as f:
    f.write(content)
