"""Applicant-level bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


def bootstrap_metric(
    values: Sequence[float],
    *,
    metric: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return {"estimate": None, "ci_95": [None, None], "n": 0, "bootstrap_seed": seed}
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sample = data[rng.integers(0, data.size, size=data.size)]
        estimates[index] = metric(sample)
    return {
        "estimate": float(metric(data)),
        "ci_95": [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))],
        "n": int(data.size),
        "bootstrap_seed": seed,
    }


def bootstrap_paired_difference(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Paired solver comparisons require identical applicant rows")
    valid = np.isfinite(a) & np.isfinite(b)
    return bootstrap_metric(
        (a[valid] - b[valid]).tolist(), n_bootstrap=n_bootstrap, seed=seed
    )


def bootstrap_ci(data, confidence=0.95, n_resamples=2000, seed=42):
    """Backward-compatible tuple interface."""
    if confidence != 0.95:
        raise ValueError("Use bootstrap_metric for non-95% intervals")
    result = bootstrap_metric(data, n_bootstrap=n_resamples, seed=seed)
    return result["estimate"], *result["ci_95"]
