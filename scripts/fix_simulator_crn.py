import re

sim_path = "backend/engine/simulator.py"
with open(sim_path, "r") as f:
    content = f.read()

# Make it draw noise for ALL actionable features from FEATURE_CONTRACT_V3
fix = """
        realized: Dict[str, float] = {}
        # To guarantee CRN synchronization across different control regimes,
        # we MUST draw random variates for ALL potentially actionable features in fixed order,
        # regardless of whether the current regime requested an action for them.
        
        from backend.engine.feature_contract import FEATURE_CONTRACT_V3
        actionable_feats = sorted([k for k, v in FEATURE_CONTRACT_V3.items() if v.actionable])
        
        for feat in actionable_feats:
            # Draw variates unconditionally
            is_miss = bool(effective_rng.binomial(1, config.p_miss) == 1)
            completion_ratio = float(effective_rng.beta(config.beta_alpha, config.beta_beta))
            
            requested_delta = parsed_actions.get(feat, 0.0)
            if abs(requested_delta) < 1e-9:
                realized[feat] = 0.0
            else:
                if is_miss:
                    realized[feat] = 0.0
                else:
                    realized[feat] = float(requested_delta * completion_ratio)

        return realized
"""
content = re.sub(
    r"        realized: Dict\[str, float\] = \{\}.*?return realized",
    fix.strip() + "\n\n        return realized",
    content, flags=re.DOTALL
)

with open(sim_path, "w") as f:
    f.write(content)
print("Patched simulator for robust CRN.")
