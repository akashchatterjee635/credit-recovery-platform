"""FastAPI entry point for risk scoring, recovery planning, and reassessment."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import math
import os
import uuid
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.database.schema import (
    Borrower,
    BorrowerSnapshot,
    ModelDecision,
    RecoveryAction,
    RecoveryJourney,
    RecoveryPlan,
    get_db,
    init_db,
)
from backend.engine.constraint_registry import DEFAULT_REGISTRY
from backend.engine.explainer import RiskExplainer
from backend.engine.feature_contract import FEATURE_CONTRACT_V3
from backend.engine.mpc_controller import MPCController
from backend.engine.planner import RecoveryTrajectoryPlanner
from backend.engine.solver_router import SolverRouter
from backend.models.risk_model import RiskModelAdapter

logger = logging.getLogger(__name__)

_THRESHOLD = DEFAULT_REGISTRY.recourse_threshold()
MODEL_VERSION = "lgbm-v3-calibrated"
FC_VERSION = "feature-contract-v3"
CR_VERSION = "constraint-registry-v2"
SOLVER_VERSION = "SolverRouter-v2"

_services: dict[str, Any] = {
    "risk_adapter": None,
    "router": None,
    "planner": None,
    "explainer": None,
    "startup_error": None,
}


def _record_startup_error(component: str, exc: Exception) -> None:
    _services["startup_error"] = {
        "component": component,
        "error_type": type(exc).__name__,
        "message": str(exc),
    }
    logger.exception("Service startup failed", extra={"component": component})


def _initialize_services() -> None:
    """Initialize artifacts once at lifespan startup; never at module import."""
    _services.update(
        risk_adapter=None,
        router=None,
        planner=None,
        explainer=None,
        startup_error=None,
    )
    try:
        init_db()
    except Exception as exc:
        _record_startup_error("database", exc)
        return

    try:
        adapter = RiskModelAdapter()
        adapter.load()
        _services["risk_adapter"] = adapter
    except Exception as exc:
        _record_startup_error("risk_model", exc)
        return

    try:
        train_path = os.path.join("data", "train_reference.csv")
        training_sample = pd.read_csv(train_path) if os.path.exists(train_path) else None
        router = SolverRouter(
            adapter,
            threshold=_THRESHOLD,
            registry=DEFAULT_REGISTRY,
            feature_contract=FEATURE_CONTRACT_V3,
            training_data=training_sample,
        )
        _services.update(
            router=router,
            planner=RecoveryTrajectoryPlanner(registry=DEFAULT_REGISTRY),
            explainer=RiskExplainer(adapter, FEATURE_CONTRACT_V3),
        )
    except Exception as exc:
        _record_startup_error("solver_router", exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _initialize_services()
    yield
    _services.update(
        risk_adapter=None,
        router=None,
        planner=None,
        explainer=None,
    )


app = FastAPI(title="Credit Recovery Intelligence API v3", lifespan=lifespan)


def _require_service(name: str):
    service = _services.get(name)
    if service is None:
        detail = {
            "code": "ARTIFACT_UNAVAILABLE",
            "component": name,
            "startup_error": _services.get("startup_error"),
        }
        raise HTTPException(status_code=503, detail=detail)
    return service


def _band(score: float) -> str:
    if score < 0.20:
        return "LOW"
    if score < 0.30:
        return "MODERATE"
    if score < 0.50:
        return "ELEVATED"
    return "HIGH"


def _next_snapshot_index(db: Session, journey_id: int) -> int:
    latest = (
        db.query(BorrowerSnapshot)
        .filter_by(journey_id=journey_id)
        .order_by(BorrowerSnapshot.snapshot_index.desc())
        .first()
    )
    return 0 if latest is None else latest.snapshot_index + 1


def _write_prediction(
    db: Session,
    features: dict[str, Any],
    risk: float,
    threshold: float,
    borrower_id: int | None = None,
    journey_id: int | None = None,
) -> tuple[int, int, BorrowerSnapshot]:
    if journey_id is not None:
        journey = db.get(RecoveryJourney, journey_id)
        if journey is None:
            raise HTTPException(404, "Journey not found")
        if borrower_id is not None and journey.borrower_id != borrower_id:
            raise HTTPException(403, "Journey does not belong to borrower")
        borrower_id = journey.borrower_id
    else:
        borrower = db.get(Borrower, borrower_id) if borrower_id is not None else None
        if borrower is None:
            borrower = Borrower(external_id=str(uuid.uuid4()))
            db.add(borrower)
            db.flush()
        borrower_id = borrower.id
        journey = RecoveryJourney(borrower_id=borrower_id, status="active")
        db.add(journey)
        db.flush()
        journey_id = journey.id

    snapshot = BorrowerSnapshot(
        journey_id=journey_id,
        snapshot_index=_next_snapshot_index(db, journey_id),
        features_json=features,
    )
    db.add(snapshot)
    db.flush()
    db.add(
        ModelDecision(
            journey_id=journey_id,
            snapshot_id=snapshot.id,
            predicted_default_risk=risk,
            risk_band=_band(risk),
            threshold_used=threshold,
            recovery_applicable=risk > threshold,
            model_version=MODEL_VERSION,
            feature_contract_version=FC_VERSION,
        )
    )
    db.flush()
    return borrower_id, journey_id, snapshot


def _write_plan(
    db: Session,
    journey_id: int,
    result: dict[str, Any],
    plan: dict[str, Any],
    *,
    replan_reason: str | None = None,
) -> RecoveryPlan:
    previous = (
        db.query(RecoveryPlan)
        .filter_by(journey_id=journey_id)
        .order_by(RecoveryPlan.plan_version.desc())
        .first()
    )
    version = 1 if previous is None else previous.plan_version + 1
    row = RecoveryPlan(
        journey_id=journey_id,
        plan_version=version,
        solver_used=result.get("solver", result.get("solver_tier", "unknown")),
        solver_version=SOLVER_VERSION,
        constraint_registry_version=CR_VERSION,
        original_risk=result.get("original_risk"),
        target_risk=result.get("new_risk"),
        total_months=plan.get("total_months"),
        status=plan.get("status", result.get("status", "failed")),
        replan_reason=replan_reason,
    )
    db.add(row)
    db.flush()
    for month in plan.get("timeline", []):
        for action in month.get("actions", []):
            db.add(
                RecoveryAction(
                    plan_id=row.id,
                    month=month.get("month"),
                    feature_name=action.get("feature"),
                    direction=action.get("direction"),
                    monthly_change=action.get("monthly_change"),
                    cumulative_target=action.get("cumulative_target"),
                    reassessment_date=month.get("reassessment_date"),
                )
            )
    db.flush()
    return row


class ApplicantData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    AMT_CREDIT: float
    AMT_INCOME_TOTAL: float
    AMT_ANNUITY: float
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    NAME_EDUCATION_TYPE: str
    BUREAU_TOTAL_DEBT: float | None = None
    BUREAU_MAX_OVERDUE: float | None = None
    BUREAU_ACTIVE_COUNT: float | None = None
    INST_LATE_RATIO: float | None = None
    INST_AVG_DAYS_LATE: float | None = None
    PREV_REFUSED_RATIO: float | None = None


class RoadmapRequest(ApplicantData):
    journey_id: int | None = None
    borrower_id: int | None = None


class ReassessmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observed_feature_updates: dict[str, float]
    policy_threshold: float | None = Field(default=None, gt=0.0, lt=1.0)

    @field_validator("observed_feature_updates")
    @classmethod
    def validate_updates(cls, updates: dict[str, float]) -> dict[str, float]:
        if not updates:
            raise ValueError("At least one observed feature update is required")
        if any(not math.isfinite(value) for value in updates.values()):
            raise ValueError("Observed feature updates must be finite numbers")
        return updates


IMMUTABLE_FEATURE_CLASSES = {"IMMUTABLE", "HISTORICAL_IMMUTABLE", "DERIVED"}
OBSERVABLE_FEATURE_CLASSES = {
    "TIME_EVOLVING",
    "ACTIONABLE_STATE",
    "ACTIONABLE_BEHAVIOUR",
    "CONDITIONALLY_ACTIONABLE",
    "LENDER_CONTROLLED",
    "PLANNING_ONLY",
}


def _merge_observed_updates(current: dict[str, Any], updates: dict[str, float]) -> dict[str, Any]:
    merged = dict(current)
    for feature, value in updates.items():
        definition = FEATURE_CONTRACT_V3.get(feature)
        if definition is None:
            raise HTTPException(422, f"Unknown feature: {feature}")
        if definition.feature_class in IMMUTABLE_FEATURE_CLASSES:
            raise HTTPException(422, f"Feature cannot be reassessed: {feature}")
        if definition.feature_class not in OBSERVABLE_FEATURE_CLASSES:
            raise HTTPException(422, f"Feature is not an allowed observation: {feature}")
        if definition.min_val is not None and value < definition.min_val:
            raise HTTPException(422, f"{feature} must be >= {definition.min_val}")
        if definition.max_val is not None and value > definition.max_val:
            raise HTTPException(422, f"{feature} must be <= {definition.max_val}")
        merged[feature] = value
    return merged


def _plan_targets(db: Session, plan: RecoveryPlan, through_month: int | None = None) -> dict[str, float]:
    query = db.query(RecoveryAction).filter_by(plan_id=plan.id)
    if through_month is not None:
        query = query.filter(RecoveryAction.month <= through_month)
    actions = query.order_by(RecoveryAction.month.asc(), RecoveryAction.id.asc()).all()
    targets: dict[str, float] = {}
    for action in actions:
        if action.feature_name and action.cumulative_target is not None:
            targets[action.feature_name] = float(action.cumulative_target)
    return targets


@app.get("/")
def root():
    return {"service": "Credit Recovery Intelligence API v3", "status": "running"}


@app.get("/health/live")
def health_live():
    return {"status": "alive"}


@app.get("/health/ready")
def health_ready(db: Session = Depends(get_db)):
    if _services["startup_error"] is not None:
        raise HTTPException(503, detail=_services["startup_error"])
    _require_service("risk_adapter")
    _require_service("router")
    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(
            503,
            detail={"code": "DATABASE_UNAVAILABLE", "message": str(exc)},
        ) from exc
    return {"status": "ready", "database": "ready", "model": "ready", "router": "ready"}


@app.get("/constraints")
def list_constraints():
    return {
        "version": CR_VERSION,
        "constraints": [
            {
                "id": constraint.constraint_id,
                "description": constraint.description,
                "confidence": constraint.confidence,
                "type": constraint.hard_or_soft,
                "params": constraint.params,
            }
            for constraint in DEFAULT_REGISTRY.all_constraints()
        ],
    }


@app.post("/predict")
def predict_risk(applicant: ApplicantData, db: Session = Depends(get_db)):
    adapter = _require_service("risk_adapter")
    features = applicant.model_dump()
    frame = pd.DataFrame([features])
    score = float(adapter.predict_risk(frame)[0])
    borrower_id, journey_id, _ = _write_prediction(db, features, score, _THRESHOLD)
    explanation = _services["explainer"].explain(frame) if _services["explainer"] else {}
    response = {
        "predicted_default_risk": round(score, 4),
        "risk_band": _band(score),
        "recovery_program_applicable": score > _THRESHOLD,
        "threshold_used": _THRESHOLD,
        "model_version": MODEL_VERSION,
        "borrower_id": borrower_id,
        "journey_id": journey_id,
        "explanation_available": bool(explanation.get("available")),
    }
    if explanation.get("available"):
        response["top_risk_drivers"] = explanation["top_risk_drivers"]
    return response


@app.post("/generate_roadmap")
def generate_roadmap(request: RoadmapRequest, db: Session = Depends(get_db)):
    adapter = _require_service("risk_adapter")
    router = _require_service("router")
    planner = _require_service("planner")
    features = request.model_dump(exclude={"journey_id", "borrower_id"})
    frame = pd.DataFrame([features])
    score = float(adapter.predict_risk(frame)[0])
    borrower_id, journey_id, _ = _write_prediction(
        db,
        features,
        score,
        _THRESHOLD,
        borrower_id=request.borrower_id,
        journey_id=request.journey_id,
    )
    result = router.generate_recourse(frame)
    if result.get("status") in {"success", "eligible"}:
        result.setdefault("original_state", features)
        result.setdefault("new_state", features)
        result.setdefault("original_risk", score)
        result.setdefault("new_risk", score)
        plan = planner.generate_timeline(result["original_state"], result["new_state"])
        result["sequential_plan"] = plan
        plan_row = _write_plan(db, journey_id, result, plan)
        result["plan_version"] = plan_row.plan_version
    result.update(
        borrower_id=borrower_id,
        journey_id=journey_id,
        constraint_registry_version=CR_VERSION,
        solver_version=SOLVER_VERSION,
    )
    if "gate_results" in result:
        result["validation"] = result.pop("gate_results")
    return result


@app.post("/journeys/{journey_id}/reassess")
def reassess_journey(
    journey_id: int,
    request: ReassessmentRequest,
    db: Session = Depends(get_db),
    x_borrower_id: int | None = Header(default=None, alias="X-Borrower-ID"),
):
    adapter = _require_service("risk_adapter")
    router = _require_service("router")
    planner = _require_service("planner")

    journey = db.get(RecoveryJourney, journey_id)
    if journey is None:
        raise HTTPException(404, "Journey not found")
    if x_borrower_id is not None and journey.borrower_id != x_borrower_id:
        raise HTTPException(403, "Journey does not belong to borrower")
    if journey.status != "active":
        raise HTTPException(409, "Only active journeys can be reassessed")

    latest_snapshot = (
        db.query(BorrowerSnapshot)
        .filter_by(journey_id=journey_id)
        .order_by(BorrowerSnapshot.snapshot_index.desc())
        .first()
    )
    if latest_snapshot is None:
        raise HTTPException(409, "Journey has no borrower snapshot")

    observed = _merge_observed_updates(
        latest_snapshot.features_json or {}, request.observed_feature_updates
    )
    observed_frame = pd.DataFrame([observed])
    risk = float(adapter.predict_risk(observed_frame)[0])
    threshold = request.policy_threshold or _THRESHOLD

    active_plan = (
        db.query(RecoveryPlan)
        .filter_by(journey_id=journey_id)
        .order_by(RecoveryPlan.plan_version.desc())
        .first()
    )
    previous_version = active_plan.plan_version if active_plan else None

    last_decision = (
        db.query(ModelDecision)
        .filter_by(journey_id=journey_id)
        .order_by(ModelDecision.id.desc())
        .first()
    )
    previous_threshold = (
        last_decision.threshold_used
        if last_decision and last_decision.threshold_used is not None
        else _THRESHOLD
    )

    controller = MPCController(
        risk_model=adapter,
        base_threshold=previous_threshold,
        feature_contract=FEATURE_CONTRACT_V3,
        solver_router=router,
    )
    actionable = controller._actionable_features(observed_frame)
    if active_plan is not None:
        controller.previous_plan_target = _plan_targets(db, active_plan) or {
            feature: float(observed[feature])
            for feature in actionable
            if feature in observed and observed[feature] is not None
        }
        expected = dict(latest_snapshot.features_json or {})
        expected.update(
            _plan_targets(db, active_plan, through_month=latest_snapshot.snapshot_index + 1)
        )
        expected_frame = pd.DataFrame([expected])
        controller.expected_state_next = {
            feature: float(expected[feature])
            for feature in actionable
            if feature in expected and expected[feature] is not None
        }
        controller.expected_risk_next = float(adapter.predict_risk(expected_frame)[0])

    trigger, reason = controller._needs_replan(
        observed_frame, risk, threshold, actionable
    )

    _, _, snapshot = _write_prediction(
        db,
        observed,
        risk,
        threshold,
        borrower_id=journey.borrower_id,
        journey_id=journey_id,
    )

    new_version = previous_version
    response: dict[str, Any] = {
        "status": "on_track",
        "journey_id": journey_id,
        "snapshot_index": snapshot.snapshot_index,
        "calibrated_risk": risk,
        "replan_triggered": trigger,
        "trigger_reason": reason,
        "previous_plan_version": previous_version,
        "new_plan_version": new_version,
    }
    if not trigger:
        return response

    target_threshold = max(0.0, threshold - controller.delta_safety)
    result = router.generate_recourse(
        observed_frame,
        target_threshold=target_threshold,
        previous_plan=controller.previous_plan_target,
        gamma_stability=controller.gamma_stability,
    )
    if result.get("status") in {"success", "eligible"}:
        result.setdefault("original_state", observed)
        result.setdefault("new_state", observed)
        result.setdefault("original_risk", risk)
        result.setdefault("new_risk", risk)
        plan = planner.generate_timeline(result["original_state"], result["new_state"])
        response["status"] = "replanned"
        response["new_plan"] = plan
    else:
        plan = {"status": "failed", "total_months": None, "timeline": []}
        response["status"] = "replan_failed"
        response["violations"] = result.get("violations", [])

    plan_row = _write_plan(db, journey_id, result, plan, replan_reason=reason)
    response["new_plan_version"] = plan_row.plan_version
    return response
