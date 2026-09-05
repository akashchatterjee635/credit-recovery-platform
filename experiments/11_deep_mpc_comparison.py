import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib
mpc = importlib.import_module("experiments.09_mpc_benchmark")
run_benchmark = mpc.run_benchmark

def main():
    print("="*60)
    print("RUNNING RECOURSE COMPARISON WITH DEEP RISK ADAPTER")
    print("="*60)
    print("Stage 5 requires measuring if downstream recourse improves.")
    
    # Run the deep version
    run_benchmark(N_APPLICANTS=25, T_HORIZON=6, use_deep=True)

if __name__ == '__main__':
    main()
