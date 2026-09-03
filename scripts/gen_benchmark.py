import os, textwrap

files = {}
files['experiments/09_mpc_benchmark.py'] = textwrap.dedent('''
import sys, os, copy
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.models.risk_model import RiskModelAdapter
from backend.engine.constraint_registry import ConstraintRegistry
from backend.engine.feature_contract import FEATURE_CONTRACT_V3
from backend.engine.solver_router import SolverRouter
from backend.engine.simulator import EnvironmentSimulator, DisturbanceConfig
from backend.engine.mpc_controller import MPCController

N_APPLICANTS = 50 # Start with 50 for speed. Final can be 1000.
T_HORIZON = 12
DELTA_SAFETY = 0.05
BASE_TAU = 0.30

def run_regime(regime_type, applicant_orig, adapter, registry, config_params, crn_seed):
    \"\"\"
    regime_type: 'one-shot', 'sequential', 'mpc'
    \"\"\"
    # Re-initialize simulator completely to ensure identical sequence
    config = DisturbanceConfig(**config_params)
    sim = EnvironmentSimulator()
    rng = np.random.RandomState(crn_seed)
    
    router = SolverRouter(risk_model=adapter, threshold=BASE_TAU, registry=registry, 
                          feature_contract=FEATURE_CONTRACT_V3, training_data=None) # Mock training data if needed
                          
    mpc = MPCController(risk_model=adapter, base_threshold=BASE_TAU, 
                        feature_contract=FEATURE_CONTRACT_V3, solver_router=router, 
                        delta_safety=DELTA_SAFETY, gamma_stability=0.5)
                        
    current_state = applicant_orig.copy()
    current_tau = BASE_TAU
    
    logs = []
    
    # 1. Initial Plan
    if regime_type in ['one-shot', 'sequential']:
        res = mpc.get_action(current_state, 0, T_HORIZON, current_tau)
        static_target = res['plan_target']
        
    for t in range(T_HORIZON):
        current_tau = sim.policy_environment.step(t, config, rng=rng)
        
        # 2. Decide Action
        if regime_type == 'mpc':
            res = mpc.get_action(current_state, t, T_HORIZON, current_tau)
            a_t = res['action_t']
            is_replan = res['replan_triggered']
            instability = res['instability_L1']
        elif regime_type == 'sequential':
            a_t = {}
            if static_target is not None:
                remaining_months = max(1, T_HORIZON - t)
                for f, targ in static_target.items():
                    orig = current_state.iloc[0].get(f, 0)
                    a_t[f] = (targ - orig) / remaining_months
            is_replan = False
            instability = 0.0
        else: # one-shot (execute once if needed, but standard one-shot means no ongoing execution, 
              # or means blind execution of the exact static amount. We will map one-shot to "attempt to execute all at once", 
              # or just execute sequential but fail immediately if off track. 
              # For fairness, One-Shot usually means generating the counterfactual and doing nothing else, 
              # but here let's say One-Shot = just executing the static plan with no feedback).
              # Let's align one-shot with fixed sequential for now, or just say one-shot attempts to jump to the state immediately.
              # Let's use fixed sequential for both or treat them identically for now.
            a_t = {}
            if static_target is not None:
                remaining_months = max(1, T_HORIZON - t)
                for f, targ in static_target.items():
                    orig = current_state.iloc[0].get(f, 0)
                    a_t[f] = (targ - orig) / remaining_months
            is_replan = False
            instability = 0.0
            
        # 3. Simulate Environment Step (CRN synchronized)
        next_state, log_t = sim.step(current_state, a_t, t, config, rng=rng)
        
        log_t['t'] = t
        log_t['risk_t'] = adapter.predict_risk(current_state)[0]
        log_t['is_replan'] = is_replan
        log_t['instability'] = instability
        
        # Calculate cost
        cost_t = 0.0
        for f, delta in log_t.get('realized_action', {}).items():
            # mock economic cost: scaled by income
            inc = current_state.iloc[0].get('AMT_INCOME_TOTAL', 1)
            cost_t += abs(delta) / (inc + 1e-5)
        log_t['cost_t'] = cost_t
        
        logs.append(log_t)
        current_state = next_state
        
    # Terminal eval
    terminal_risk = adapter.predict_risk(current_state)[0]
    is_valid = terminal_risk <= current_tau
    
    return is_valid, logs


if __name__ == '__main__':
    adapter = RiskModelAdapter()
    adapter.load()
    
    test_df = pd.read_csv('data/test_reference.csv').dropna(subset=['TARGET'])
    train_df = pd.read_csv('data/train_reference.csv').dropna(subset=['TARGET']).head(500)
    
    registry = ConstraintRegistry()
    
    # Extract applicants that need recourse
    risks = adapter.predict_risk(test_df)
    above_mask = risks > BASE_TAU
    sample_df = test_df[above_mask].head(N_APPLICANTS)
    
    config_params = {
        'p_miss': 0.10,
        'p_income_shock': 0.05,
        'p_debt_shock': 0.05,
        'policy_shift': True
    }
    
    results = {'sequential': [], 'mpc': []}
    
    for i in range(len(sample_df)):
        applicant = sample_df.iloc[[i]]
        seed = 42 + i
        
        v_seq, log_seq = run_regime('sequential', applicant, adapter, registry, config_params, seed)
        v_mpc, log_mpc = run_regime('mpc', applicant, adapter, registry, config_params, seed)
        
        def agg(v, log):
            costs = sum(l['cost_t'] for l in log)
            replans = sum(l['is_replan'] for l in log)
            instab = sum(l['instability'] for l in log)
            return {'valid': v, 'cost': costs, 'replans': replans, 'instability': instab}
            
        results['sequential'].append(agg(v_seq, log_seq))
        results['mpc'].append(agg(v_mpc, log_mpc))
        
    print('\\n============================================================')
    print(f'MPC BENCHMARK RESULTS (N={N_APPLICANTS}, T={T_HORIZON})')
    print('============================================================')
    
    for regime in ['sequential', 'mpc']:
        df_res = pd.DataFrame(results[regime])
        v_rate = df_res['valid'].mean()
        avg_cost = df_res['cost'].mean()
        avg_replan = df_res['replans'].mean()
        avg_instab = df_res['instability'].mean()
        
        print(f'Regime: {regime.upper()}')
        print(f'  Terminal Validity (Survival): {v_rate:.1%}')
        print(f'  Avg Cumulative Cost:          {avg_cost:.4f}')
        print(f'  Avg Replans:                  {avg_replan:.2f}')
        print(f'  Avg Plan Instability:         {avg_instab:.2f}\\n')
        
''').lstrip()

for path, content in files.items():
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(content)
    print(f'wrote {path}')
