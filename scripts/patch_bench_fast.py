import re
path = "experiments/09_mpc_benchmark.py"
with open(path, "r", encoding="utf-8") as f:
    code = f.read()

# Replace the pre-filtering loop with a fast brute-force one
old_code = '''        res = shared_router.generate_recourse(applicant, target_threshold=TAU_TARGET)
        if res.get("status") in ("success", "eligible") and res.get("new_state"):
            af = _actionable(applicant)
            target = {f: float(res["new_state"][f]) for f in af if f in res["new_state"]}'''

new_code = '''        af = _actionable(applicant)
        cand = applicant.copy()
        if "BUREAU_TOTAL_DEBT" in cand: cand["BUREAU_TOTAL_DEBT"] = 0
        if "BUREAU_MAX_OVERDUE" in cand: cand["BUREAU_MAX_OVERDUE"] = 0
        if "AMT_ANNUITY" in cand: cand["AMT_ANNUITY"] /= 2.0
        
        new_risk = float(adapter.predict_risk(cand)[0])
        if new_risk <= TAU_TARGET:
            target = {f: float(cand.iloc[0][f]) for f in af}'''

code = code.replace(old_code, new_code)
with open(path, "w", encoding="utf-8") as f:
    f.write(code)
print("Patched benchmark to use brute-force pre-filtering for speed.")
