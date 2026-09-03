import re
path = "experiments/09_mpc_benchmark.py"
with open(path, "r", encoding="utf-8") as f:
    code = f.read()

# Fix the bug where MPC is seeded with the final target as expected_state_next
# which caused an immediate STATE_DEVIATION trigger at t=0
old_code = '''        mpc.previous_plan_target = dict(shared_target)
        # Seed the expected trajectory using the SLSQP target as t=0 expectation
        af = _actionable(applicant_orig)
        mpc.expected_state_next = {
            f: shared_target.get(f, float(applicant_orig.iloc[0].get(f, 0)))
            for f in af
        }
        # Pre-compute expected risk at x_hat_1
        try:
            exp_df = applicant_orig.copy()
            for f, v in mpc.expected_state_next.items():
                if f in exp_df.columns:
                    exp_df.iloc[0, exp_df.columns.get_loc(f)] = v
            mpc.expected_risk_next = float(adapter.predict_risk(exp_df)[0])
        except Exception:
            mpc.expected_risk_next = None'''

new_code = '''        mpc.previous_plan_target = dict(shared_target)
        # expected_state_next is left as None at t=0, so the first get_action() 
        # doesn't falsely trigger a STATE_DEVIATION against the final target.'''

code = code.replace(old_code, new_code)
with open(path, "w", encoding="utf-8") as f:
    f.write(code)
print("Removed false expected_state_next seeding.")
