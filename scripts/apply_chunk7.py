import os, textwrap

files = {}
files['experiments/08_threshold_sensitivity.py'] = textwrap.dedent('''
import sys, os, time
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.models.risk_model import RiskModelAdapter
from backend.engine.constraint_registry import ConstraintRegistry, DEFAULT_REGISTRY
from backend.engine.feature_contract import FEATURE_CONTRACT_V3
from backend.engine.solver_router import SolverRouter

if __name__ == '__main__':
    print('Loading model and reference data...')
    adapter = RiskModelAdapter()
    adapter.load()
    
    test_df = pd.read_csv('data/test_reference.csv').dropna(subset=['TARGET'])
    train_df = pd.read_csv('data/train_reference.csv').dropna(subset=['TARGET']).head(2000)
    
    risks = adapter.predict_risk(test_df)
    
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30]
    
    print('='*80)
    print('THRESHOLD SENSITIVITY ANALYSIS')
    print('='*80)
    
    results = []
    
    for tau in thresholds:
        print(f'\\nEvaluating threshold tau = {tau}')
        
        above_mask = risks > tau
        eligible_count = above_mask.sum()
        
        # sample up to 50 applicants for speed in this experiment
        sample_df = test_df[above_mask]
        if len(sample_df) > 50:
            sample_df = sample_df.sample(50, random_state=42)
            
        registry = ConstraintRegistry()
        for c in registry._constraints:
            if c.constraint_id == 'RECOURSE_THRESHOLD_001':
                c.params['threshold'] = tau
                
        router = SolverRouter(risk_model=adapter, threshold=tau, registry=registry,
                              feature_contract=FEATURE_CONTRACT_V3, training_data=train_df)
        
        successes = 0
        costs = []
        
        for i in range(len(sample_df)):
            applicant = sample_df.iloc[[i]]
            res = router.generate_recourse(applicant)
            if res.get('status') == 'success':
                successes += 1
                costs.append(res.get('cost', 0))
                
        success_rate = successes / len(sample_df) if len(sample_df) > 0 else 0
        avg_cost = np.mean(costs) if costs else 0
        
        results.append({
            'Threshold': tau,
            'Eligible Count': eligible_count,
            'Recourse Availability': success_rate,
            'Average Cost': avg_cost
        })
        
    print('\\n' + '='*80)
    print(f'{"Threshold":<15} {"Eligible Count":<20} {"Availability":<15} {"Avg Cost":<15}')
    print('-'*80)
    for r in results:
        print(f'{r["Threshold"]:<15.2f} {r["Eligible Count"]:<20} {r["Recourse Availability"]:<15.1%} {r["Average Cost"]:<15.4f}')
    print('='*80)
''').lstrip()

for path, content in files.items():
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(content)
    print(f'  wrote {path}')
print('Done.')
