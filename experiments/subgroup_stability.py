"""Subgroup risk/recourse metrics and recommendation-stability diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

from backend.models.risk_model import compute_ece
from experiments.bootstrap import bootstrap_metric

STABILITY_DIMENSIONS = (
    "random_seed",
    "train_calibration_split",
    "operating_threshold",
    "small_feature_perturbation",
    "missing_features",
    "income_shock",
    "debt_shock",
    "model_retraining",
    "constraint_registry_update",
    "distribution_drift",
)


def derive_audit_subgroups(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive standard audit groups without adding them to model inputs."""
    audited = frame.copy()
    if "DAYS_BIRTH" in audited:
        age = -pd.to_numeric(audited["DAYS_BIRTH"], errors="coerce") / 365.25
        audited["audit_age_band"] = pd.cut(
            age,
            bins=[0, 25, 35, 45, 55, 65, np.inf],
            labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"],
        )
    if "AMT_INCOME_TOTAL" in audited:
        audited["audit_income_quantile"] = pd.qcut(
            audited["AMT_INCOME_TOTAL"], 4, duplicates="drop"
        )
    if "BUREAU_ACTIVE_COUNT" in audited:
        count = pd.to_numeric(audited["BUREAU_ACTIVE_COUNT"], errors="coerce").fillna(0)
        audited["audit_credit_file"] = np.where(count <= 1, "thin_file", "thick_file")
    if "BUREAU_TOTAL_DEBT" in audited:
        audited["audit_credit_exposure"] = pd.qcut(
            audited["BUREAU_TOTAL_DEBT"], 4, duplicates="drop"
        )
    return audited


def _safe_auc(metric: Callable, labels: np.ndarray, scores: np.ndarray) -> float:
    return float(metric(labels, scores)) if np.unique(labels).size > 1 else float("nan")


def subgroup_risk_metrics(
    frame: pd.DataFrame,
    *,
    group_column: str,
    label_column: str,
    score_column: str,
    threshold: float,
    minimum_group_size: int = 100,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    report = {}
    for group, rows in frame.groupby(group_column, dropna=False):
        labels = rows[label_column].to_numpy(dtype=int)
        scores = rows[score_column].to_numpy(dtype=float)
        predictions = scores >= threshold
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        if len(rows) < minimum_group_size:
            warnings.warn(f"Subgroup {group!r} has only {len(rows)} rows", stacklevel=2)
        report[str(group)] = {
            "n": int(len(rows)),
            "small_group_warning": len(rows) < minimum_group_size,
            "default_prevalence": float(labels.mean()),
            "roc_auc": _safe_auc(roc_auc_score, labels, scores),
            "pr_auc": _safe_auc(average_precision_score, labels, scores),
            "brier_score": float(brier_score_loss(labels, scores)),
            "ece": float(compute_ece(labels, scores)),
            "tpr": tp / (tp + fn) if tp + fn else None,
            "fpr": fp / (fp + tn) if fp + tn else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "score_ci": bootstrap_metric(scores.tolist(), seed=bootstrap_seed),
        }
    return report


def subgroup_recourse_metrics(
    frame: pd.DataFrame,
    *,
    group_column: str,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    metric_columns = {
        "feasible_recourse_rate": ("feasible", np.mean),
        "median_recourse_cost": ("recourse_cost", np.median),
        "median_actions": ("number_of_actions", np.median),
        "median_recovery_duration": ("recovery_duration", np.median),
        "solver_failure_rate": ("solver_failed", np.mean),
        "infeasible_within_horizon_rate": ("infeasible_within_horizon", np.mean),
        "plan_instability": ("plan_instability", np.mean),
        "manifold_rejection_rate": ("manifold_rejected", np.mean),
    }
    report = {}
    for group, rows in frame.groupby(group_column, dropna=False):
        report[str(group)] = {
            name: bootstrap_metric(
                rows[column].dropna().tolist(), metric=metric, seed=bootstrap_seed
            )
            for name, (column, metric) in metric_columns.items()
            if column in rows
        }
        report[str(group)]["n"] = int(len(rows))
    return report


def plan_instability(
    plan_a: Mapping[str, float],
    plan_b: Mapping[str, float],
    *,
    scales: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
) -> float:
    weights = weights or {}
    features = set(plan_a) | set(plan_b)
    return float(
        sum(
            weights.get(feature, 1.0)
            * abs(plan_a.get(feature, 0.0) - plan_b.get(feature, 0.0))
            / max(abs(scales.get(feature, 1.0)), 1e-12)
            for feature in features
        )
    )


def recommendation_churn(plan_a: Mapping[str, float], plan_b: Mapping[str, float]) -> float:
    shared = set(plan_a) & set(plan_b)
    if not shared:
        return 0.0
    changed = sum(np.sign(plan_a[name]) != np.sign(plan_b[name]) for name in shared)
    return changed / len(shared)


def perturbation_stability(
    applicant: pd.DataFrame,
    planner: Callable[[pd.DataFrame], Mapping[str, float]],
    perturbations: Mapping[str, float],
    scales: Mapping[str, float],
) -> dict[str, float]:
    base = dict(planner(applicant.copy()))
    scores = []
    churn = []
    for feature, delta in perturbations.items():
        if feature not in applicant:
            continue
        perturbed = applicant.copy()
        perturbed[feature] = perturbed[feature] + delta
        changed = dict(planner(perturbed))
        scores.append(plan_instability(base, changed, scales=scales))
        churn.append(recommendation_churn(base, changed))
    return {
        "mean_plan_instability": float(np.mean(scores)) if scores else 0.0,
        "recommendation_churn": float(np.mean(churn)) if churn else 0.0,
    }


def run_stability_suite(
    evaluator: Callable[[str], Mapping[str, float]],
    dimensions: tuple[str, ...] = STABILITY_DIMENSIONS,
) -> dict[str, Mapping[str, float]]:
    """Evaluate every predeclared stability dimension through a supplied runner."""
    missing = set(STABILITY_DIMENSIONS) - set(dimensions)
    if missing:
        raise ValueError(f"Required stability dimensions omitted: {sorted(missing)}")
    return {dimension: evaluator(dimension) for dimension in dimensions}
