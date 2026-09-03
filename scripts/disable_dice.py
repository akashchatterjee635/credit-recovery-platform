import os

path = "backend/engine/solver_router.py"
with open(path, "r") as f:
    content = f.read()

content = content.replace("tiers.append(('dice',  self._dice))", "# tiers.append(('dice',  self._dice))")
content = content.replace("result = solver.generate_recourse(applicant)", "result = solver.generate_recourse(applicant, previous_plan=previous_plan, gamma_stability=gamma_stability) if tier_name == 'slsqp' else solver.generate_recourse(applicant)")

with open(path, "w") as f:
    f.write(content)
print("Disabled DiCE and fixed slsqp arguments")
