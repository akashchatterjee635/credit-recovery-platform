import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database.schema import Base, RecoveryPlan
from backend.engine.planner import RecoveryTrajectoryPlanner
from backend.main import (
    ApplicantData,
    ReassessmentRequest,
    RoadmapRequest,
    _merge_observed_updates,
    _services,
    generate_roadmap,
    health_live,
    predict_risk,
    reassess_journey,
)

pytestmark = pytest.mark.integration

APPLICANT = {
    "AMT_CREDIT": 300_000.0,
    "AMT_INCOME_TOTAL": 100_000.0,
    "AMT_ANNUITY": 15_000.0,
    "DAYS_BIRTH": -12_000,
    "DAYS_EMPLOYED": -2_000,
    "NAME_EDUCATION_TYPE": "Higher education",
    "BUREAU_TOTAL_DEBT": 80_000.0,
    "BUREAU_MAX_OVERDUE": 5_000.0,
    "BUREAU_ACTIVE_COUNT": 3.0,
}


class FakeAdapter:
    model = object()

    def predict_risk(self, frame):
        return np.repeat(0.42, len(frame))


class FakeRouter:
    def generate_recourse(self, frame, **_):
        original = frame.iloc[0].to_dict()
        updated = dict(original)
        updated["BUREAU_TOTAL_DEBT"] = max(0, original["BUREAU_TOTAL_DEBT"] - 10_000)
        return {
            "status": "success",
            "solver": "fake",
            "original_risk": 0.42,
            "new_risk": 0.24,
            "original_state": original,
            "new_state": updated,
        }


@pytest.fixture()
def db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    yield session
    session.close()


@pytest.fixture(autouse=True)
def fake_services():
    previous = dict(_services)
    _services.update(
        risk_adapter=FakeAdapter(),
        router=FakeRouter(),
        planner=RecoveryTrajectoryPlanner(),
        explainer=None,
        startup_error=None,
    )
    yield
    _services.clear()
    _services.update(previous)


def test_health_live():
    assert health_live() == {"status": "alive"}


def test_predict_and_generate_roadmap_end_to_end(db):
    prediction = predict_risk(ApplicantData(**APPLICANT), db)
    assert prediction["journey_id"]
    roadmap = generate_roadmap(
        RoadmapRequest(
            **APPLICANT,
            borrower_id=prediction["borrower_id"],
            journey_id=prediction["journey_id"],
        ),
        db,
    )
    assert roadmap["status"] == "success"
    assert roadmap["plan_version"] == 1
    assert roadmap["sequential_plan"]["timeline"]


def test_reassessment_versions_plan_and_records_reason(db):
    prediction = predict_risk(ApplicantData(**APPLICANT), db)
    initial = generate_roadmap(
        RoadmapRequest(
            **APPLICANT,
            borrower_id=prediction["borrower_id"],
            journey_id=prediction["journey_id"],
        ),
        db,
    )
    response = reassess_journey(
        prediction["journey_id"],
        ReassessmentRequest(
            observed_feature_updates={"BUREAU_TOTAL_DEBT": 90_000.0},
            policy_threshold=0.25,
        ),
        db,
        x_borrower_id=prediction["borrower_id"],
    )
    assert initial["plan_version"] == 1
    assert response["previous_plan_version"] == 1
    assert response["new_plan_version"] == 2
    assert response["replan_triggered"] is True
    stored = db.query(RecoveryPlan).filter_by(plan_version=2).one()
    assert stored.replan_reason == "POLICY_CHANGE"


def test_immutable_variables_cannot_change():
    with pytest.raises(Exception) as error:
        _merge_observed_updates(APPLICANT, {"DAYS_BIRTH": -9_000})
    assert getattr(error.value, "status_code", None) == 422


def test_ownership_is_enforced(db):
    prediction = predict_risk(ApplicantData(**APPLICANT), db)
    with pytest.raises(Exception) as error:
        reassess_journey(
            prediction["journey_id"],
            ReassessmentRequest(observed_feature_updates={"BUREAU_TOTAL_DEBT": 70_000.0}),
            db,
            x_borrower_id=prediction["borrower_id"] + 1,
        )
    assert getattr(error.value, "status_code", None) == 403
