import os

path = "backend/engine/solvers/slsqp_solver.py"
with open(path, "r") as f:
    content = f.read()

content = content.replace("res = minimize(", "print('Starting minimize...', flush=True)\n            res = minimize(")
content = content.replace("if res.success:", "print('Finished minimize.', flush=True)\n            if res.success:")

with open(path, "w") as f:
    f.write(content)
