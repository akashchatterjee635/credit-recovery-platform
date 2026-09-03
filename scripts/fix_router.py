import re

# ── Fix 2 (continued): solver_router must accept + propagate target_threshold ─
path = "backend/engine/solver_router.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace generate_recourse signature
old_sig = "def generate_recourse(self, applicant: pd.DataFrame, previous_plan: dict = None, gamma_stability: float = 0.0) -> dict:"
new_sig = "def generate_recourse(self, applicant: pd.DataFrame, target_threshold: float = None, previous_plan: dict = None, gamma_stability: float = 0.0) -> dict:"
content = content.replace(old_sig, new_sig)

# Add effective_threshold resolution at top of method body (after the def line)
old_body_start = "        if self.risk_model.model is None:\n            self.risk_model.load()"
new_body_start = """        # Bug-2 fix: use per-call target_threshold if supplied, else fall back to self.threshold
        effective_threshold = target_threshold if target_threshold is not None else self.threshold
        if self.risk_model.model is None:
            self.risk_model.load()"""
content = content.replace(old_body_start, new_body_start)

# Pass effective_threshold into each solver call
old_call = "result = solver.generate_recourse(applicant, previous_plan=previous_plan, gamma_stability=gamma_stability) if tier_name == 'slsqp' else solver.generate_recourse(applicant)"
new_call = """result = solver.generate_recourse(
                    applicant,
                    target_threshold=effective_threshold,
                    previous_plan=previous_plan,
                    gamma_stability=gamma_stability,
                ) if tier_name == 'slsqp' else solver.generate_recourse(applicant, target_threshold=effective_threshold)"""
content = content.replace(old_call, new_call)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("OK  solver_router.py (threshold propagation)")
