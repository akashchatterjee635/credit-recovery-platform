import numpy as np

from experiments.bootstrap import bootstrap_metric, bootstrap_paired_difference
from experiments.subgroup_stability import plan_instability, recommendation_churn


def test_bootstrap_is_reproducible_and_reports_lineage():
    first = bootstrap_metric([0, 1, 1, 0], n_bootstrap=200, seed=7)
    second = bootstrap_metric([0, 1, 1, 0], n_bootstrap=200, seed=7)
    assert first == second
    assert first["n"] == 4
    assert first["bootstrap_seed"] == 7
    assert first["estimate"] == 0.5


def test_paired_bootstrap_preserves_applicant_pairing():
    result = bootstrap_paired_difference([1, 2, 3], [0, 1, 2], n_bootstrap=100)
    assert result["estimate"] == 1.0
    assert result["ci_95"] == [1.0, 1.0]


def test_plan_instability_and_churn():
    instability = plan_instability(
        {"debt": -100.0, "annuity": -10.0},
        {"debt": -50.0, "annuity": 10.0},
        scales={"debt": 100.0, "annuity": 10.0},
        weights={"debt": 2.0, "annuity": 1.0},
    )
    assert np.isclose(instability, 3.0)
    assert recommendation_churn(
        {"debt": -100.0, "annuity": -10.0},
        {"debt": -50.0, "annuity": 10.0},
    ) == 0.5
