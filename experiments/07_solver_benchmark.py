import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from backend.models.risk_model import RiskModelAdapter
from backend.models.feature_engineering import build_enriched_dataset
from backend.engine.constraint_registry import DEFAULT_REGISTRY
from backend.engine.feature_contract import FEATURE_CONTRACT_V3
from backend.engine.solvers.slsqp_solver import SLSQPSolver
from backend.engine.solvers.binary_search_solver import BinarySearchSolver
from backend.engine.solvers.dice_solver import DiCESolver
from backend.engine.solver_router import SolverRouter
from backend.engine.validator import FeasibilityGuard

DATA_DIR = 'data'
N_SAMPLE = 1000

def get_held_out_applicants(adapter, n=1000):
    df = pd.read_csv('data/test_reference.csv')
    df = df.dropna(subset=['TARGET'])
    risks = adapter.predict_risk(df)
    threshold = DEFAULT_REGISTRY.recourse_threshold()
    above_thresh = df[risks > threshold].copy()
    if len(above_thresh) > n:
        sample = above_thresh.sample(n=n, random_state=42)
    else:
        sample = above_thresh
    return sample


def bootstrap_ci(metric_list, n_bootstraps=1000, ci=95):
    if not metric_list:
        return 0.0, 0.0, 0.0
    arr = np.array(metric_list)
    boot_means = []
    for _ in range(n_bootstraps):
        boot_sample = np.random.choice(arr, size=len(arr), replace=True)
        boot_means.append(np.mean(boot_sample))

    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return np.mean(arr), lower, upper


if __name__ == '__main__':
    print('Loading model...')
    adapter = RiskModelAdapter()
    adapter.load()

    print(f'Extracting {N_SAMPLE} high-risk held-out applicants...')
    test_sample = get_held_out_applicants(adapter, n=N_SAMPLE)

    print(f'Loading training data for DiCE/durability...')
    # Load 5000 train rows
    df_all = build_enriched_dataset(DATA_DIR)
    X_train = pd.read_csv('data/train_reference.csv')
    train_sample = X_train.head(5000)

    kwargs = dict(risk_model=adapter, threshold=DEFAULT_REGISTRY.recourse_threshold(),
                  registry=DEFAULT_REGISTRY, feature_contract=FEATURE_CONTRACT_V3)

    slsqp = SLSQPSolver(**kwargs)
    bsearch = BinarySearchSolver(**kwargs)
    dice = DiCESolver(**kwargs, training_data=train_sample)
    router = SolverRouter(**kwargs, training_data=train_sample)

    solvers = [
        ('BinarySearch', bsearch),
        ('SLSQP', slsqp),
        ('DiCE', dice),
        ('Router', router)
    ]

    results_by_solver = {name: [] for name, _ in solvers}

    print(f'Benchmarking on {len(test_sample)} applicants...')

    for i in range(len(test_sample)):
        if (i+1) % 100 == 0:
            print(f'  Processed {i+1}/{len(test_sample)}...')

        applicant_row = test_sample.iloc[[i]]

        for name, solver in solvers:
            t0 = time.time()
            if name == 'Router':
                res_dict = solver.generate_recourse(applicant_row)
                status = res_dict.get('status')
                cost = res_dict.get('cost')
                gates = res_dict.get('validation') or res_dict.get('gate_results')
                is_success = (status == 'success')
            else:
                res = solver.generate_recourse(applicant_row)
                status = res.status
                cost = res.cost
                gates = res.gate_results
                is_success = (status == 'success')
                
            elapsed = time.time() - t0
            
            results_by_solver[name].append({
                'validity': 1 if is_success else 0,
                'structural_fail': 1 if gates and not gates.get('V_structural', True) else 0,
                'actionability_fail': 1 if gates and not gates.get('V_actionability', True) else 0,
                'plausibility_fail': 1 if gates and not gates.get('V_plausibility', True) else 0,
                'durability_fail': 1 if gates and not gates.get('V_durability', True) else 0,
                'cost': cost if cost is not None else float('nan'),
                'latency': elapsed
            })

    # Output metrics
    print('\n' + '='*80)
    print(f'SOLVER BENCHMARK RESULTS (N={len(test_sample)} held-out)')
    print('='*80)

    cols = ['Metric', 'BinarySearch', 'SLSQP', 'DiCE', 'Router']
    print(f'{cols[0]:<30} {cols[1]:<15} {cols[2]:<15} {cols[3]:<15} {cols[4]:<15}')
    print('-'*80)

    def format_ci(mean, lower, upper, is_pct=True):
        if is_pct:
            return f'{mean:.1%} [{lower:.1%}-{upper:.1%}]'
        else:
            return f'{mean:.3f} [{lower:.3f}-{upper:.3f}]'

    metrics = [
        ('Full feasible validity', 'validity', True),
        ('Structural violation rate', 'structural_fail', True),
        ('Actionability violation rate', 'actionability_fail', True),
        ('Plausibility violation rate', 'plausibility_fail', True),
        ('Durability violation rate', 'durability_fail', True),
    ]

    for label, key, is_pct in metrics:
        row_str = f'{label:<30}'
        for name, _ in solvers:
            vals = [r[key] for r in results_by_solver[name]]
            mean, lower, upper = bootstrap_ci(vals)
            row_str += f' {format_ci(mean, lower, upper, is_pct):<15}'
        print(row_str)

    # Latency and cost
    row_str = f'{"Median action cost":<30}'
    for name, _ in solvers:
        valid_costs = [r['cost'] for r in results_by_solver[name] if not np.isnan(r['cost'])]
        if valid_costs:
            mean = np.median(valid_costs)
            row_str += f' {mean:<15.4f}'
        else:
            row_str += f' {"-":<15}'
    print(row_str)

    row_str = f'{"P50 latency (s)":<30}'
    for name, _ in solvers:
        lats = [r['latency'] for r in results_by_solver[name]]
        p50 = np.percentile(lats, 50)
        row_str += f' {p50:<15.3f}'
    print(row_str)

    row_str = f'{"P95 latency (s)":<30}'
    for name, _ in solvers:
        lats = [r['latency'] for r in results_by_solver[name]]
        p95 = np.percentile(lats, 95)
        row_str += f' {p95:<15.3f}'
    print(row_str)
    print('='*80)
