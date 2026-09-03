import os

path = "backend/engine/solver_router.py"
with open(path, "r") as f:
    content = f.read()

content = content.replace("attempted_tiers.append(tier_name)", "print(f'Attempting {tier_name}...', flush=True)\n            attempted_tiers.append(tier_name)")
content = content.replace("result = solver.generate_recourse", "print('calling generate_recourse...', flush=True)\n            result = solver.generate_recourse")
content = content.replace("return d", "print('returned d', flush=True)\n                return d")

with open(path, "w") as f:
    f.write(content)
