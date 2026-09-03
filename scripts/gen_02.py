import os, sys, json
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

if __name__ == '__main__':
    data_path = 'experiments/outputs/calibration_data.json'
    if not os.path.exists(data_path):
        print(f'No calibration data found at {data_path}. Run risk_model.py train first.')
        sys.exit(1)

    with open(data_path) as f:
        data = json.load(f)

    unc_metrics = data['uncalibrated']['metrics']
    cal_metrics = data['calibrated']['metrics']

    print('='*60)
    print('CALIBRATION COMPARISON')
    print('='*60)
    print(f'{"Metric":<20} {"Raw LightGBM":>15} {"Isotonic Cal":>15}')
    print('-'*50)
    print(f'{"Brier Score":<20} {unc_metrics.get("Brier", 0):>15.4f} {cal_metrics.get("Brier", 0):>15.4f}')
    print(f'{"ECE":<20} {unc_metrics.get("ECE", 0):>15.4f} {cal_metrics.get("ECE", 0):>15.4f}')
    print('='*60)

    unc_curve = data['uncalibrated']['curve']
    cal_curve = data['calibrated']['curve']

    if plt is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Uncalibrated
        ax = axes[0]
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax.plot(unc_curve['prob_pred'], unc_curve['prob_true'], 'o-', label=f'Raw (ECE={unc_metrics.get("ECE", 0):.4f})')
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of positives')
        ax.set_title('Raw LightGBM')
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Calibrated
        ax = axes[1]
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax.plot(cal_curve['prob_pred'], cal_curve['prob_true'], 'o-', label=f'Calibrated (ECE={cal_metrics.get("ECE", 0):.4f})')
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of positives')
        ax.set_title('Isotonic Calibrated')
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        out_path = 'experiments/outputs/reliability_curve.png'
        plt.savefig(out_path, dpi=150)
        print(f'\\nReliability diagram saved to {out_path}')
    else:
        print('matplotlib not available, skipping diagram.')

