from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from backend.models.risk_model import RiskModelAdapter
from backend.engine.solver_router import SolverRouter
from backend.engine.planner import RecoveryTrajectoryPlanner
from backend.engine.constraint_registry import DEFAULT_REGISTRY
from backend.engine.feature_contract import FEATURE_CONTRACT_V2
from backend.database.schema import SessionLocal, init_db, Borrower, RecoveryJourney
from backend.database.schema import BorrowerSnapshot, ModelDecision, RecoveryPlan, RecoveryAction
import uuid, datetime

app = FastAPI(title='Credit Recovery Intelligence API v2')

# ── Startup ────────────────────────────────────────────────────────────────
try:
    init_db()
    _risk_adapter = RiskModelAdapter()
    _risk_adapter.load()
    _router = SolverRouter(_risk_adapter, threshold=0.3,
                           registry=DEFAULT_REGISTRY,
                           feature_contract=FEATURE_CONTRACT_V2)
    _planner = RecoveryTrajectoryPlanner(registry=DEFAULT_REGISTRY)
    print('API startup complete.')
except Exception as e:
    print(f'WARNING: startup failed – {e}')
    _risk_adapter = _router = _planner = None

MODEL_VERSION = 'lgbm-v2'
FC_VERSION = 'feature-contract-v2'
CR_VERSION = 'constraint-registry-v1'
SOLVER_VERSION = 'SolverRouter-v1'


def _band(score: float) -> str:
    if score < 0.20: return 'LOW'
    if score < 0.30: return 'MODERATE'
    if score < 0.50: return 'ELEVATED'
    return 'HIGH'


def _write_prediction(features: dict, risk: float, band: str, applicable: bool):
    '''Persist prediction to DB (non-blocking; errors are logged, not raised).'''
    try:
        db = SessionLocal()
        ext_id = str(uuid.uuid4())
        borrower = Borrower(external_id=ext_id)
        db.add(borrower); db.flush()
        journey = RecoveryJourney(borrower_id=borrower.id)
        db.add(journey); db.flush()
        snap = BorrowerSnapshot(journey_id=journey.id, snapshot_index=0,
                                features_json=features)
        db.add(snap); db.flush()
        decision = ModelDecision(
            journey_id=journey.id, snapshot_id=snap.id,
            predicted_default_risk=risk, risk_band=band,
            threshold_used=0.3, recovery_applicable=applicable,
            model_version=MODEL_VERSION, feature_contract_version=FC_VERSION)
        db.add(decision)
        db.commit()
        return journey.id
    except Exception as exc:
        print(f'[DB] write_prediction failed: {exc}')
        return None
    finally:
        db.close()


def _write_plan(journey_id, result: dict, plan: dict):
    try:
        db = SessionLocal()
        rp = RecoveryPlan(
            journey_id=journey_id,
            solver_used=result.get('solver', 'unknown'),
            solver_version=SOLVER_VERSION,
            constraint_registry_version=CR_VERSION,
            original_risk=result.get('original_risk'),
            target_risk=result.get('new_risk'),
            total_months=plan.get('total_months'),
            status=plan.get('status', 'unknown'))
        db.add(rp); db.flush()
        for step in plan.get('timeline', []):
            for action in step.get('actions', []):
                db.add(RecoveryAction(
                    plan_id=rp.id, month=step['month'],
                    feature_name=action.get('feature'),
                    direction=action.get('direction'),
                    monthly_change=action.get('monthly_change'),
                    cumulative_target=action.get('cumulative_target'),
                    reassessment_date=step.get('reassessment_date')))
        db.commit()
    except Exception as exc:
        print(f'[DB] write_plan failed: {exc}')
    finally:
        db.close()


class ApplicantData(BaseModel):
    AMT_CREDIT: float
    AMT_INCOME_TOTAL: float
    AMT_ANNUITY: float
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    NAME_EDUCATION_TYPE: str


@app.get('/')
def root():
    return {'service': 'Credit Recovery Intelligence API v2', 'status': 'running'}


@app.get('/constraints')
def list_constraints():
    return {'constraints': [
        {'id': c.constraint_id, 'description': c.description,
         'confidence': c.confidence, 'type': c.hard_or_soft, 'params': c.params}
        for c in DEFAULT_REGISTRY.all_constraints()]}


@app.post('/predict')
def predict_risk(applicant: ApplicantData):
    if _risk_adapter is None:
        raise HTTPException(503, 'Risk model not loaded.')
    df = pd.DataFrame([applicant.model_dump()])
    score = float(_risk_adapter.predict_risk(df)[0])
    band = _band(score)
    applicable = score > _router.threshold if _router else score > 0.3
    features = applicant.model_dump()
    _write_prediction(features, score, band, applicable)
    return {
        'predicted_default_risk': round(score, 4),
        'risk_band': band,
        'recovery_assessment_applicable': applicable,
        'threshold_used': 0.3,
        'model_version': MODEL_VERSION,
    }


@app.post('/generate_roadmap')
def generate_roadmap(applicant: ApplicantData):
    if _router is None or _planner is None:
        raise HTTPException(503, 'Router or planner not loaded.')
    df = pd.DataFrame([applicant.model_dump()])

    # Persist initial state
    score = float(_risk_adapter.predict_risk(df)[0])
    band = _band(score)
    journey_id = _write_prediction(applicant.model_dump(), score, band, score > 0.3)

    result = _router.generate_recourse(df)

    if result.get('status') == 'success':
        plan = _planner.generate_timeline(result['original_state'], result['new_state'])
        result['sequential_plan'] = plan
        result['constraint_registry_version'] = CR_VERSION
        result['solver_version'] = SOLVER_VERSION
        if journey_id:
            _write_plan(journey_id, result, plan)

    return result
