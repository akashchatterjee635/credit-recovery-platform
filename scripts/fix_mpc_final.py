import re

path = "experiments/09_mpc_benchmark.py"
with open(path, "r", encoding="utf-8") as f:
    code = f.read()

# Fix Bug 1: router._dice = None inside run_regime
old_router_logic = '''        training_data=train_df,
    )
    # Disable DiCE to speed up the pre-filter search dramatically
    shared_router._dice = None'''

new_router_logic = '''        training_data=train_df,
    )
    # Disable DiCE to speed up the pre-filter search dramatically
    router._dice = None'''
    
# Only replace the first occurrence (which is inside run_regime)
code = code.replace(old_router_logic, new_router_logic, 1)

# Fix Bug 2: Pass T_HORIZON to run_regime
old_def = "def run_regime(\n    regime_name, applicant_orig, adapter, registry, train_df,\n    config, seed, shared_target, fixed_actions, is_deterministic\n):"
new_def = "def run_regime(\n    regime_name, applicant_orig, adapter, registry, train_df,\n    config, seed, shared_target, fixed_actions, is_deterministic, T_HORIZON=12\n):"
code = code.replace(old_def, new_def)

# Actually, the original def might not have line breaks exactly like that.
# Let's use regex.
code = re.sub(r'def run_regime\([^)]*\):', 
'''def run_regime(
    regime_name, applicant_orig, adapter, registry, train_df,
    config, seed, shared_target, fixed_actions, is_deterministic, T_HORIZON
):''', code)

# Update the call to run_regime inside run_benchmark
old_call = '''            res = run_regime(
                regime, applicant, adapter, registry, train_df,
                config, seed,
                shared_target=shared_target,
                fixed_actions=fixed_actions,
                is_deterministic=is_det,
            )'''
            
new_call = '''            res = run_regime(
                regime, applicant, adapter, registry, train_df,
                config, seed,
                shared_target=shared_target,
                fixed_actions=fixed_actions,
                is_deterministic=is_det,
                T_HORIZON=T_HORIZON,
            )'''
            
code = code.replace(old_call, new_call)

with open(path, "w", encoding="utf-8") as f:
    f.write(code)
print("Fixed 09_mpc_benchmark.py bugs.")
