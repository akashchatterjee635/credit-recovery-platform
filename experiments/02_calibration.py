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

    print('\nReliability Diagram (Calibrated):')
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
        print(f'\nReliability diagram saved to {out_path}')
    except ImportError:
        print('matplotlib not available, skipping diagram.')
