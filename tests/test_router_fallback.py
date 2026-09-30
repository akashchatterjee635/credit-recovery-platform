from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from backend.engine.base_solver import RecourseResult
from backend.engine.solver_router import SolverRouter


def _router():
    model = MagicMock()
    model.model = object()
    model.predict_risk.return_value = np.array([0.5])
    router = SolverRouter(model)
    return router


def test_router_falls_back_after_slsqp_failure():
    router = _router()
    router._slsqp.generate_recourse = MagicMock(
        return_value=RecourseResult(status="failed", solver="slsqp", message="no gradient")
    )
    router._dice.generate_recourse = MagicMock(
        return_value=RecourseResult(
            status="success",
            solver="dice",
            message="ok",
            original_state={"AMT_ANNUITY": 10_000, "BUREAU_TOTAL_DEBT": 20_000},
            new_state={"AMT_ANNUITY": 8_000, "BUREAU_TOTAL_DEBT": 10_000},
        )
    )
    applicant = pd.DataFrame([{"AMT_ANNUITY": 10_000.0, "BUREAU_TOTAL_DEBT": 20_000.0}])
    result = router.generate_recourse(applicant)
    assert result["status"] == "success"
    assert result["solver_tier"] == "dice"
    assert result["tiers_attempted"] == ["slsqp", "dice"]


def test_router_returns_failure_reasons_from_every_tier():
    router = _router()
    router._slsqp.generate_recourse = MagicMock(
        return_value=RecourseResult(status="failed", solver="slsqp", message="first")
    )
    router._dice.generate_recourse = MagicMock(
        return_value=RecourseResult(status="failed", solver="dice", message="second")
    )
    applicant = pd.DataFrame([{"AMT_ANNUITY": 10_000.0, "BUREAU_TOTAL_DEBT": 20_000.0}])
    result = router.generate_recourse(applicant)
    assert result["status"] == "failed"
    assert result["tiers_attempted"] == ["slsqp", "dice"]
    assert any("first" in reason for reason in result["violations"])
    assert any("second" in reason for reason in result["violations"])
