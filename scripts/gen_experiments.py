import os, textwrap

files = {}

# ── experiments/02_calibration.py ─────────────────────────────────────────────
files['experiments/02_calibration.py'] = textwrap.dedent('''
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np

def compute_ece(y_true, y_prob, n_bins=10):
    y_true, y_prob = np.array(y_true), np.array(y_prob)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bins_data = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            bins_data.append((lo, hi, 0, 0, 0))
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        count = mask.sum()
        ece += count / len(y_true) * abs(bin_acc - bin_conf)
        bins_data.append((lo, hi, bin_acc, bin_conf, count))
    return ece, bins_data

if __name__ == '__main__':
    data_path = 'experiments/outputs/calibration_data.json'
    if not os.path.exists(data_path):
        print(f'No calibration data found at {data_path}. Run risk_model.py train first.')
        sys.exit(1)

    with open(data_path) as f:
        data = json.load(f)

    y_true = np.array(data['y_true'])
    y_raw = np.array(data['y_prob_raw'])
    y_cal = np.array(data['y_prob_cal'])

    from sklearn.metrics import brier_score_loss

    ece_raw, bins_raw = compute_ece(y_true, y_raw)
    ece_cal, bins_cal = compute_ece(y_true, y_cal)
    brier_raw = brier_score_loss(y_true, y_raw)
    brier_cal = brier_score_loss(y_true, y_cal)

    print('='*60)
    print('CALIBRATION COMPARISON')
    print('='*60)
    print(f'{"Metric":<20} {"Raw LightGBM":>15} {"Isotonic Cal":>15}')
    print('-'*50)
    print(f'{"Brier Score":<20} {brier_raw:>15.4f} {brier_cal:>15.4f}')
    print(f'{"ECE (10 bins)":<20} {ece_raw:>15.4f} {ece_cal:>15.4f}')
    print('='*60)

    print('\\nReliability Diagram (Calibrated):')
    print(f'{"Bin":<12} {"Predicted":>10} {"Observed":>10} {"Count":>8}')
    print('-'*42)
    for lo, hi, acc, conf, count in bins_cal:
        if count > 0:
            print(f'[{lo:.1f}-{hi:.1f}]   {conf:>10.3f} {acc:>10.3f} {count:>8}')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, probs, label in [(axes[0], y_raw, 'Raw LightGBM'),
                                  (axes[1], y_cal, 'Isotonic Calibrated')]:
            ece_val, bins = compute_ece(y_true, probs)
            predicted = [b[3] for b in bins if b[4] > 0]
            observed = [b[2] for b in bins if b[4] > 0]
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
            ax.plot(predicted, observed, 'o-', label=f'{label} (ECE={ece_val:.4f})')
            ax.set_xlabel('Mean predicted probability')
            ax.set_ylabel('Fraction of positives')
            ax.set_title(label)
            ax.legend()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        plt.tight_layout()
        out_path = 'experiments/outputs/reliability_curve.png'
        plt.savefig(out_path, dpi=150)
        print(f'\\nReliability diagram saved to {out_path}')
    except ImportError:
        print('matplotlib not available, skipping diagram.')
''').lstrip()


# ── experiments/07_solver_benchmark.py (Upgraded) ────────────────────────────
files['experiments/07_solver_benchmark.py'] = textwrap.dedent('''
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
    df = build_enriched_dataset(DATA_DIR)
    target = 'TARGET'
    df = df.dropna(subset=[target])
    
    # 60/20/20 split used in risk_model.py
    X_trainval, X_test, _, y_test = train_test_split(
        df, df[target].astype(int), test_size=0.2, random_state=42, stratify=df[target].astype(int)
    )
    
    # Predict risk on test set
    risks = adapter.predict_risk(X_test)
    
    # Filter above threshold
    threshold = DEFAULT_REGISTRY.recourse_threshold()
    above_thresh = X_test[risks > threshold].copy()
    
    # Sample n rows
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
    trainval, _, _, _ = train_test_split(
        df_all, df_all['TARGET'].astype(int), test_size=0.2, random_state=42, stratify=df_all['TARGET'].astype(int)
    )
    X_train, _, _, _ = train_test_split(
        trainval, trainval['TARGET'].astype(int), test_size=0.25, random_state=42, stratify=trainval['TARGET'].astype(int)
    )
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
                res = RecourseResult(**{k:v for k,v in res_dict.items() if hasattr(RecourseResult, k) and k != 'validation'})
                if 'validation' in res_dict:
                    res.gate_results = res_dict['validation']
            else:
                res = solver.generate_recourse(applicant_row)
                
            elapsed = time.time() - t0
            
            results_by_solver[name].append({
                'validity': 1 if res.status == 'success' else 0,
                'structural_fail': 1 if res.gate_results and not res.gate_results.get('V_structural', True) else 0,
                'actionability_fail': 1 if res.gate_results and not res.gate_results.get('V_actionability', True) else 0,
                'plausibility_fail': 1 if res.gate_results and not res.gate_results.get('V_plausibility', True) else 0,
                'durability_fail': 1 if res.gate_results and not res.gate_results.get('V_durability', True) else 0,
                'cost': res.cost if res.cost is not None else float('nan'),
                'latency': elapsed
            })
            
    # Output metrics
    print('\\n' + '='*80)
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
            mean, lower, upper = bootstrap_ci(vals, is_pct=is_pct)
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
''').lstrip()

for path, content in files.items():
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(content)
    print(f'  wrote {path}')
print('Done.')
