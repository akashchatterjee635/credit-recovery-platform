'''
experiments/07_solver_benchmark.py
Compares SLSQP vs BinarySearch vs SolverRouter on:
  validity (did it cross the threshold?),
  structural violations,
  action cost,
  runtime (seconds)

Usage:
    .\venv\Scripts\python experiments/07_solver_benchmark.py
'''
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from backend.models.risk_model import RiskModelAdapter
from backend.engine.constraint_registry import DEFAULT_REGISTRY
from backend.engine.feature_contract import FEATURE_CONTRACT_V2
from backend.engine.solvers.slsqp_solver import SLSQPSolver
from backend.engine.solvers.binary_search_solver import BinarySearchSolver
from backend.engine.solver_router import SolverRouter

TEST_CASES = [
    {'AMT_CREDIT': 500000.0, 'AMT_INCOME_TOTAL': 60000.0, 'AMT_ANNUITY': 45000.0,
     'DAYS_BIRTH': -15000, 'DAYS_EMPLOYED': -500, 'NAME_EDUCATION_TYPE': 'Secondary / secondary special'},
    {'AMT_CREDIT': 800000.0, 'AMT_INCOME_TOTAL': 90000.0, 'AMT_ANNUITY': 55000.0,
     'DAYS_BIRTH': -12000, 'DAYS_EMPLOYED': -200, 'NAME_EDUCATION_TYPE': 'Incomplete higher'},
]


def run(solver, df, label):
    t0 = time.time()
    result = solver.generate_recourse(df)
    elapsed = time.time() - t0
    return {
        'Solver': label,
        'Status': result.get('status', '?'),
        'Validity': result.get('status') == 'success',
        'Cost': round(result.get('cost', float('nan')), 4),
        'Time(s)': round(elapsed, 3),
        'Violations': len(result.get('violations', [])),
    }


if __name__ == '__main__':
    print('Loading model...')
    adapter = RiskModelAdapter()
    adapter.load()

    kwargs = dict(risk_model=adapter, threshold=0.3,
                  registry=DEFAULT_REGISTRY, feature_contract=FEATURE_CONTRACT_V2)
    slsqp   = SLSQPSolver(**kwargs)
    bsearch = BinarySearchSolver(**kwargs)
    router  = SolverRouter(**kwargs)

    rows = []
    for i, case in enumerate(TEST_CASES):
        df = pd.DataFrame([case])
        print(f'\nTest case {i+1}: risk={adapter.predict_risk(df)[0]:.3f}')
        for solver, label in [(slsqp, 'SLSQP'), (bsearch, 'BinarySearch'), (router, 'SolverRouter')]:
            r = run(solver, df, label)
            r['Case'] = i + 1
            rows.append(r)

    print('\n' + '='*75)
    print('SOLVER BENCHMARK RESULTS')
    print('='*75)
    print(pd.DataFrame(rows).to_string(index=False))
    print('='*75)
