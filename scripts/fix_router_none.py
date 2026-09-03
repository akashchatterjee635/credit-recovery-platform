import re

path = "backend/engine/solver_router.py"
with open(path, "r", encoding="utf-8") as f:
    code = f.read()

code = code.replace(
    "for tier_name, solver in tiers:",
    "for tier_name, solver in tiers:\n            if solver is None: continue"
)

with open(path, "w", encoding="utf-8") as f:
    f.write(code)
print("Updated solver_router to allow None solvers.")
