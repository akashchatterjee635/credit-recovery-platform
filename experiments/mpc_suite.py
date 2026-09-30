"""Scenario orchestration for one-shot, fixed sequential, and event-triggered MPC."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.bootstrap import bootstrap_metric, bootstrap_paired_difference
from experiments.reporting import save_benchmark_run

MPC_SCENARIOS = ("zero", "mild", "moderate", "severe", "policy_shift")
MPC_STRATEGIES = ("one_shot", "fixed_sequential", "event_triggered_mpc")
MPC_METRICS = (
    "terminal_validity",
    "valid_state_occupancy",
    "recovery_month",
    "trajectory_survival",
    "cumulative_action_cost",
    "replanning_frequency",
    "replanning_success",
    "plan_instability",
    "solver_failure_rate",
)


def require_deep_benchmark_artifacts(
    *,
    sequence_path: str | Path,
    model_path: str | Path,
    preprocessor_path: str | Path,
    comparable_population_ids: Iterable[int] | None,
) -> None:
    missing = [
        str(path)
        for path in (sequence_path, model_path, preprocessor_path)
        if not Path(path).exists()
    ]
    if missing:
        raise RuntimeError(f"Deep benchmark prerequisites missing: {', '.join(missing)}")
    if not comparable_population_ids:
        raise RuntimeError("Deep/static comparison requires an explicit shared applicant population")


def run_mpc_suite(
    applicant_ids: Iterable[int],
    runner: Callable[[int, str, str, int], dict[str, Any]],
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Run each strategy with common random numbers per applicant and scenario."""
    rows = []
    for scenario_index, scenario in enumerate(MPC_SCENARIOS):
        for applicant_index, applicant_id in enumerate(applicant_ids):
            common_seed = seed + scenario_index * 100_000 + applicant_index
            for strategy in MPC_STRATEGIES:
                result = runner(int(applicant_id), scenario, strategy, common_seed)
                rows.append(
                    {
                        "applicant_id": int(applicant_id),
                        "scenario": scenario,
                        "strategy": strategy,
                        "common_random_seed": common_seed,
                        **result,
                    }
                )
    return pd.DataFrame(rows)


def summarize_mpc_suite(results: pd.DataFrame, *, seed: int = 42) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for scenario, scenario_rows in results.groupby("scenario"):
        summary[scenario] = {}
        for strategy, rows in scenario_rows.groupby("strategy"):
            summary[scenario][strategy] = {
                metric: bootstrap_metric(rows[metric].dropna().tolist(), seed=seed)
                for metric in MPC_METRICS
                if metric in rows
            }
        baseline = scenario_rows[scenario_rows["strategy"] == "one_shot"].sort_values("applicant_id")
        mpc = scenario_rows[scenario_rows["strategy"] == "event_triggered_mpc"].sort_values("applicant_id")
        if baseline["applicant_id"].tolist() == mpc["applicant_id"].tolist():
            summary[scenario]["paired_mpc_minus_one_shot"] = {
                metric: bootstrap_paired_difference(
                    mpc[metric].tolist(), baseline[metric].tolist(), seed=seed
                )
                for metric in MPC_METRICS
                if metric in baseline and metric in mpc
            }
    return summary


def persist_mpc_suite(
    run_id: str,
    results: pd.DataFrame,
    config: dict[str, Any],
    *,
    seed: int = 42,
) -> Path:
    summary = summarize_mpc_suite(results, seed=seed)
    return save_benchmark_run(
        run_id,
        config=config,
        applicant_results=results,
        summary=summary,
        report_markdown=(
            "# MPC benchmark\n\nAll strategies use the same common-random-number seed for each "
            "applicant/scenario pair. Estimates and paired 95% bootstrap intervals are in "
            "`summary.json`.\n"
        ),
    )
