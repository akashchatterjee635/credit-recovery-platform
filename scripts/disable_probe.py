import os

path = "backend/engine/solver_router.py"
with open(path, "r") as f:
    content = f.read()

content = content.replace("if self._probe_monotonicity(applicant, feat):", "if False:")

with open(path, "w") as f:
    f.write(content)
print("Disabled monotonicity probing")
