import os

path = "backend/engine/solvers/slsqp_solver.py"
with open(path, "r") as f:
    content = f.read()

content = content.replace("options={'ftol': 1e-4, 'disp': False}", "options={'ftol': 1e-4, 'disp': False, 'maxiter': 10}")

with open(path, "w") as f:
    f.write(content)
print("Added maxiter=10 to SLSQP")
