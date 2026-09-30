"""Compare all recourse solvers on identical held-out applicants."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.engine.constraint_registry import DEFAULT_REGISTRY
from backend.engine.feature_contract import FEATURE_CONTRACT_V3
from backend.engine.solver_router import SolverRouter
from backend.engine.solvers.binary_search_solver import BinarySearchSolver
from backend.engine.solvers.dice_solver import DiCESolver
from backend.engine.solvers.slsqp_solver import SLSQPSolver
from backend.models.risk_model import RiskModelAdapter
from experiments.bootstrap import bootstrap_metric
from experiments.reporting import save_benchmark_run


def _as_dict(result) -> dict:
    return result if isinstance(result, dict) else result.to_dict()


def _changed_features(result: dict) -> int:
    original = result.get("original_state") or {}
    updated = result.get("new_state") or {}
    return sum(
        1
        for name, value in updated.items()
        if name in original and value != original[name]
    )


def evaluate_solver(name: str, solver, applicant: pd.DataFrame, applicant_id: int) -> dict:
    started = time.perf_counter()
    try:
        result = _as_dict(solver.generate_recourse(applicant))
        error = None
    except Exception as exc:
        result = {"status": "failed", "message": str(exc)}
        error = type(exc).__name__
    latency = time.perf_counter() - started
    gates = result.get("gate_results") or result.get("validation") or {}
    success = result.get("status") in {"success", "eligible"}
    attempted = result.get("tiers_attempted", [])
    return {
        "applicant_id": applicant_id,
        "solver": name,
        "success": int(success),
        "validity": int(success and all(gates.values())) if gates else int(success),
        "structural_violation": int(not gates.get("V_structural", True)),
        "actionability_violation": int(not gates.get("V_actionability", True)),
        "plausibility_violation": int(not gates.get("V_plausibility", True)),
        "manifold_failure": int(not gates.get("V_manifold", True)),
        "action_cost": result.get("cost", np.nan),
        "latency_seconds": latency,
        "features_changed": _changed_features(result),
        "fallback": int(name == "Router" and len(attempted) > 1),
        "failure_reason": None if success else error or result.get("message", "unknown"),
    }


def _metric(values, metric=np.mean, seed=42) -> dict:
    return bootstrap_metric(values, metric=metric, seed=seed)


def summarize_population(results: pd.DataFrame, applicant_ids: set[int], seed: int) -> dict:
    population = results[results["applicant_id"].isin(applicant_ids)]
    summary = {}
    for solver, rows in population.groupby("solver"):
        successful = rows[rows["success"] == 1]
        summary[solver] = {
            "recourse_coverage": _metric(rows["success"].tolist(), seed=seed),
            "validity_rate": _metric(rows["validity"].tolist(), seed=seed),
            "structural_violation_rate": _metric(rows["structural_violation"].tolist(), seed=seed),
            "actionability_violation_rate": _metric(rows["actionability_violation"].tolist(), seed=seed),
            "plausibility_violation_rate": _metric(rows["plausibility_violation"].tolist(), seed=seed),
            "manifold_failure_rate": _metric(rows["manifold_failure"].tolist(), seed=seed),
            "mean_action_cost": _metric(successful["action_cost"].tolist(), seed=seed),
            "median_action_cost": _metric(successful["action_cost"].tolist(), metric=np.median, seed=seed),
            "p50_latency_seconds": _metric(rows["latency_seconds"].tolist(), metric=np.median, seed=seed),
            "p95_latency_seconds": _metric(
                rows["latency_seconds"].tolist(), metric=lambda data: np.percentile(data, 95), seed=seed
            ),
            "mean_features_changed": _metric(successful["features_changed"].tolist(), seed=seed),
            "fallback_frequency": _metric(rows["fallback"].tolist(), seed=seed),
            "failure_reasons": Counter(rows["failure_reason"].dropna()).most_common(),
        }
    return summary


def run(n_applicants: int = 1000, seed: int = 42, run_id: str = "solver-benchmark") -> Path:
    adapter = RiskModelAdapter()
    adapter.load()
    test = pd.read_csv("data/test_reference.csv")
    risk = adapter.predict_risk(test)
    candidates = test[risk > DEFAULT_REGISTRY.recourse_threshold()].head(n_applicants)
    training = pd.read_csv("data/train_reference.csv").head(5000)
    shared = {
        "risk_model": adapter,
        "threshold": DEFAULT_REGISTRY.recourse_threshold(),
        "registry": DEFAULT_REGISTRY,
        "feature_contract": FEATURE_CONTRACT_V3,
    }
    solvers = {
        "BinarySearch": BinarySearchSolver(**shared),
        "SLSQP": SLSQPSolver(**shared),
        "DiCE": DiCESolver(**shared, training_data=training),
        "Router": SolverRouter(**shared, training_data=training),
    }
    rows = []
    for index, (_, applicant) in enumerate(candidates.iterrows()):
        applicant_id = int(applicant.get("SK_ID_CURR", index))
        frame = applicant.to_frame().T
        for name, solver in solvers.items():
            rows.append(evaluate_solver(name, solver, frame, applicant_id))
    results = pd.DataFrame(rows)
    all_ids = set(results["applicant_id"].unique())
    solvable_ids = set(
        results.groupby("applicant_id")["success"].max().loc[lambda series: series == 1].index
    )
    summary = {
        "all_above_recourse_threshold": summarize_population(results, all_ids, seed),
        "solvable_by_at_least_one_solver": summarize_population(results, solvable_ids, seed),
        "bootstrap_seed": seed,
        "n_all": len(all_ids),
        "n_solvable": len(solvable_ids),
    }
    report = (
        "# Solver benchmark\n\n"
        f"Applicants above threshold: {len(all_ids)}. Applicants solvable by at least one solver: "
        f"{len(solvable_ids)}. Machine-readable estimates and 95% applicant-level bootstrap "
        "intervals are in `summary.json`.\n"
    )
    return save_benchmark_run(
        run_id,
        config={"n_applicants": n_applicants, "bootstrap_seed": seed},
        applicant_results=results,
        summary=summary,
        report_markdown=report,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-applicants", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default="solver-benchmark")
    args = parser.parse_args()
    print(run(args.n_applicants, args.seed, args.run_id))


if __name__ == "__main__":
    main()
