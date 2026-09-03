import re

# ── Fix 2+3: Update SLSQPSolver to accept target_threshold + implement C_stability ──
path = "backend/engine/solvers/slsqp_solver.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Update generate_recourse signature
old_sig = "def generate_recourse(self, applicant: pd.DataFrame, previous_plan: dict = None, gamma_stability: float = 0.0) -> RecourseResult:"
new_sig = "def generate_recourse(self, applicant: pd.DataFrame, target_threshold: float = None, previous_plan: dict = None, gamma_stability: float = 0.0) -> RecourseResult:"
content = content.replace(old_sig, new_sig)

# Inject effective_threshold at top of the method, replacing the fixed use of self.threshold
old_thresh_line = "        threshold = self.threshold"
new_thresh_block = """        # Bug-2 fix: use per-call target_threshold if supplied
        threshold = target_threshold if target_threshold is not None else self.threshold"""
content = content.replace(old_thresh_line, new_thresh_block)

# Bug-3: replace the objective function with one that includes C_stability
old_obj = """        def objective(x_opt):
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
                
            return cost"""
new_obj = """        # Compute per-feature scale for stability penalty (std of feature or 1.0 fallback)
        feat_scales = []
        for feat in actionable:
            vals_col = applicant[feat].values if feat in applicant.columns else np.array([1.0])
            s = float(np.std(vals_col)) if len(vals_col) > 1 else 1.0
            feat_scales.append(max(s, 1.0))

        def objective(x_opt):
            # C_action: normalized weighted distance from original
            cost = sum(
                w * abs(x_opt[i] - orig)
                for i, (orig, w) in enumerate(zip(orig_vals, weights))
            )
            # C_stability: Bug-3 fix — weighted squared distance from previous plan target
            # J_stability = gamma * sum_j w_j * ((x_j - x_prev_j) / s_j)^2
            if previous_plan and gamma_stability > 0:
                stab = 0.0
                for i, feat in enumerate(actionable):
                    x_prev = previous_plan.get(feat, orig_vals[i])
                    stab += weights[i] * ((x_opt[i] - x_prev) / feat_scales[i]) ** 2
                cost += gamma_stability * stab
            return cost"""
content = content.replace(old_obj, new_obj)

# Also update base solvers (BinarySearchSolver, DiCESolver) to accept target_threshold gracefully
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("OK  slsqp_solver.py (threshold + C_stability)")
