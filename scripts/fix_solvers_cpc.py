import re

solver_path = "backend/engine/solvers/slsqp_solver.py"
with open(solver_path, "r") as f:
    content = f.read()

# Add previous_plan and gamma_stability to generate_recourse signature
sig_replace = "def generate_recourse(self, applicant: pd.DataFrame, previous_plan: dict = None, gamma_stability: float = 0.0) -> RecourseResult:"
content = re.sub(r"def generate_recourse\(self, applicant: pd.DataFrame\) -> RecourseResult:", sig_replace, content)

# Modify objective function
obj_replace = """
        def objective(x_opt):
            # C_action: distance from orig_vals weighted by cost
            cost = sum(weight * abs(x_opt[i] - orig) for i, (orig, weight) in enumerate(zip(orig_vals, weights)))
            
            # C_plan_change: distance from previous_plan (if provided)
            if previous_plan and gamma_stability > 0:
                # previous_plan contains the target feature values for actionable features
                # if an actionable feature isn't in previous_plan, its target was its orig val
                stability_penalty = 0.0
                for i, feat in enumerate(actionable):
                    prev_target = previous_plan.get(feat, orig_vals[i])
                    # Absolute distance to previous target
                    stability_penalty += abs(x_opt[i] - prev_target)
                cost += gamma_stability * stability_penalty
                
            return cost
"""
content = re.sub(r"        def objective\(x_opt\):.*?return cost\n", obj_replace.lstrip(), content, flags=re.DOTALL)

with open(solver_path, "w") as f:
    f.write(content)

# Update solver_router.py to forward previous_plan and gamma
router_path = "backend/engine/solver_router.py"
with open(router_path, "r") as f:
    router_content = f.read()

router_sig_replace = "def generate_recourse(self, applicant: pd.DataFrame, previous_plan: dict = None, gamma_stability: float = 0.0) -> dict:"
router_content = re.sub(r"def generate_recourse\(self, applicant: pd.DataFrame\) -> dict:", router_sig_replace, router_content)

router_content = router_content.replace(
    "res = solver.generate_recourse(applicant)",
    "res = solver.generate_recourse(applicant, previous_plan=previous_plan, gamma_stability=gamma_stability) if isinstance(solver, SLSQPSolver) else solver.generate_recourse(applicant)"
)

with open(router_path, "w") as f:
    f.write(router_content)

print("Updated slsqp and router for C_plan-change")
