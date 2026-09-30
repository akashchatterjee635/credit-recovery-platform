"""Evaluate a saved static artifact and persist a traceable run."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.models.risk_model import RiskModelAdapter, compute_ece
from experiments.reporting import save_benchmark_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset", default="data/test_reference.csv")
    args = parser.parse_args()
    frame = pd.read_csv(args.dataset).dropna(subset=["TARGET"])
    adapter = RiskModelAdapter()
    adapter.load()
    scores = adapter.predict_risk(frame)
    labels = frame["TARGET"].astype(int).to_numpy()
    results = pd.DataFrame(
        {
            "applicant_id": frame.get("SK_ID_CURR", pd.Series(range(len(frame)))),
            "label": labels,
            "risk": scores,
        }
    )
    metrics = {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "brier_score": float(brier_score_loss(labels, scores)),
        "ece": float(compute_ece(labels, scores)),
        "n": len(frame),
    }
    output = save_benchmark_run(
        args.run_id,
        config={"dataset": args.dataset},
        applicant_results=results,
        summary=metrics,
        report_markdown="# Model evaluation\n\nMetrics are stored in `summary.json`.\n",
    )
    print(output)


if __name__ == "__main__":
    main()
