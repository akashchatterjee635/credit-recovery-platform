from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from backend.models.risk_model import RiskModelAdapter
from backend.engine.solver_router import SolverRouter
from backend.engine.planner import RecoveryTrajectoryPlanner
from backend.engine.constraint_registry import DEFAULT_REGISTRY
from backend.engine.feature_contract import FEATURE_CONTRACT_V3
from backend.engine.explainer import RiskExplainer
from backend.database.schema import SessionLocal, init_db, Borrower, RecoveryJourney
from backend.database.schema import BorrowerSnapshot, ModelDecision, RecoveryPlan, RecoveryAction
import uuid, datetime, os as _os

app = FastAPI(title='Credit Recovery Intelligence API v3')

_THRESHOLD = DEFAULT_REGISTRY.recourse_threshold()
MODEL_VERSION = 'lgbm-v3-calibrated'
FC_VERSION = 'feature-contract-v3'
CR_VERSION = 'constraint-registry-v2'
SOLVER_VERSION = 'SolverRouter-v2'

_risk_adapter = _router = _planner = _explainer = None
_training_sample = None

try:
    init_db()
    _risk_adapter = RiskModelAdapter()
    _risk_adapter.load()

    train_path = 'data/train_reference.csv'
    if _os.path.exists(train_path):
        _training_sample = pd.read_csv(train_path)
        print(f'Loaded {len(_training_sample)} training samples for DiCE.')

    _router = SolverRouter(_risk_adapter, threshold=_THRESHOLD,
                           registry=DEFAULT_REGISTRY,
                           feature_contract=FEATURE_CONTRACT_V3,
                           training_data=_training_sample)
    _planner = RecoveryTrajectoryPlanner(registry=DEFAULT_REGISTRY)
    _explainer = RiskExplainer(_risk_adapter, FEATURE_CONTRACT_V3)
    print('API v3 startup complete.')
except Exception as e:
    print(f'WARNING: startup failed: {e}')


def _band(score):
    if score < 0.20: return 'LOW'
    if score < 0.30: return 'MODERATE'
    if score < 0.50: return 'ELEVATED'
    return 'HIGH'


def _write_prediction(features, risk, band, applicable, borrower_id=None, journey_id=None):
    try:
        db = SessionLocal()
        if borrower_id and journey_id:
            journey = db.query(RecoveryJourney).filter_by(id=journey_id).first()
            if journey:
                snap_count = db.query(BorrowerSnapshot).filter_by(journey_id=journey.id).count()
                snap = BorrowerSnapshot(journey_id=journey.id, snapshot_index=snap_count, features_json=features)
                db.add(snap); db.flush()
                decision = ModelDecision(journey_id=journey.id, snapshot_id=snap.id,
                    predicted_default_risk=risk, risk_band=band,
                    threshold_used=_THRESHOLD, recovery_applicable=applicable,
                    model_version=MODEL_VERSION, feature_contract_version=FC_VERSION)
                db.add(decision); db.commit()
                return journey.borrower_id, journey.id
        ext_id = str(uuid.uuid4())
        borrower = Borrower(external_id=ext_id)
        db.add(borrower); db.flush()
        journey = RecoveryJourney(borrower_id=borrower.id)
        db.add(journey); db.flush()
        snap = BorrowerSnapshot(journey_id=journey.id, snapshot_index=0, features_json=features)
        db.add(snap); db.flush()
        decision = ModelDecision(journey_id=journey.id, snapshot_id=snap.id,
            predicted_default_risk=risk, risk_band=band,
            threshold_used=_THRESHOLD, recovery_applicable=applicable,
            model_version=MODEL_VERSION, feature_contract_version=FC_VERSION)
        db.add(decision); db.commit()
        return borrower.id, journey.id
    except Exception as exc:
        print(f'[DB] write_prediction failed: {exc}')
        return None, None
    finally:
        db.close()


def _write_plan(journey_id, result, plan):
    try:
        db = SessionLocal()
        rp = RecoveryPlan(journey_id=journey_id,
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
                db.add(RecoveryAction(plan_id=rp.id, month=step['month'],
                    feature_name=action.get('feature'), direction=action.get('direction'),
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
    BUREAU_TOTAL_DEBT: Optional[float] = None
    BUREAU_MAX_OVERDUE: Optional[float] = None
    BUREAU_ACTIVE_COUNT: Optional[float] = None
    INST_LATE_RATIO: Optional[float] = None
    INST_AVG_DAYS_LATE: Optional[float] = None
    PREV_REFUSED_RATIO: Optional[float] = None


class RoadmapRequest(BaseModel):
    AMT_CREDIT: float
    AMT_INCOME_TOTAL: float
    AMT_ANNUITY: float
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    NAME_EDUCATION_TYPE: str
    BUREAU_TOTAL_DEBT: Optional[float] = None
    BUREAU_MAX_OVERDUE: Optional[float] = None
    BUREAU_ACTIVE_COUNT: Optional[float] = None
    INST_LATE_RATIO: Optional[float] = None
    INST_AVG_DAYS_LATE: Optional[float] = None
    PREV_REFUSED_RATIO: Optional[float] = None
    journey_id: Optional[int] = None
    borrower_id: Optional[int] = None


@app.get('/')
def root():
    return {'service': 'Credit Recovery Intelligence API v3', 'status': 'running'}


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
    applicable = score > _THRESHOLD
    features = applicant.model_dump()

    borrower_id, journey_id = _write_prediction(features, score, band, applicable)

    explanation = {}
    if _explainer:
        explanation = _explainer.explain(df)

    resp = {
        'predicted_default_risk': round(score, 4),
        'risk_band': band,
        'recovery_assessment_applicable': applicable,
        'threshold_used': _THRESHOLD,
        'model_version': MODEL_VERSION,
        'borrower_id': borrower_id,
        'journey_id': journey_id,
    }
    if explanation.get('available'):
        resp['top_risk_drivers'] = explanation['top_risk_drivers']
    return resp


@app.post('/generate_roadmap')
def generate_roadmap(req: RoadmapRequest):
    if _router is None or _planner is None:
        raise HTTPException(503, 'Router or planner not loaded.')
    features = {k: v for k, v in req.model_dump().items() if k not in ('journey_id', 'borrower_id')}
    df = pd.DataFrame([features])

    score = float(_risk_adapter.predict_risk(df)[0])
    band = _band(score)
    borrower_id, journey_id = _write_prediction(
        features, score, band, score > _THRESHOLD,
        borrower_id=req.borrower_id, journey_id=req.journey_id)

    result = _router.generate_recourse(df)

    if result.get('status') == 'success':
        plan = _planner.generate_timeline(result['original_state'], result['new_state'])
        result['sequential_plan'] = plan
        result['constraint_registry_version'] = CR_VERSION
        result['solver_version'] = SOLVER_VERSION
        # Standardise validation key
        if 'gate_results' in result:
            result['validation'] = result.pop('gate_results')
        if journey_id:
            _write_plan(journey_id, result, plan)

    result['borrower_id'] = borrower_id
    result['journey_id'] = journey_id
    return result


@app.post("/journeys/{journey_id}/reassess")
def reassess_journey(journey_id: int, request: RoadmapRequest):
    # This is a stub for the closed-loop endpoint
    # 1. Merges observed_feature_updates into a new snapshot
    # 2. Checks risk via MPCController._needs_replan
    # 3. Creates plan version v+1 if needed
    
    applicant_df = pd.DataFrame([request.model_dump(exclude={'borrower_id', 'journey_id'})])
    risk_prob = risk_model.predict_risk(applicant_df)[0]
    
    # Mock behavior for API wrapper: Just generate a new plan
    res = solver_router.generate_recourse(applicant_df)
    
    # Save snapshot
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT MAX(snapshot_index) FROM snapshots WHERE journey_id=?", (journey_id,))
        row = c.fetchone()
        next_idx = (row[0] + 1) if row and row[0] is not None else 1
        
        c.execute("INSERT INTO snapshots (journey_id, snapshot_index, state_data) VALUES (?, ?, ?)",
                  (journey_id, next_idx, applicant_df.to_json()))
                  
        if res.get('status') == 'success':
            c.execute("SELECT MAX(plan_version) FROM recovery_plans WHERE journey_id=?", (journey_id,))
            p_row = c.fetchone()
            next_p_idx = (p_row[0] + 1) if p_row and p_row[0] is not None else 1
            
            c.execute("INSERT INTO recovery_plans (journey_id, plan_version, replan_reason, roadmap_data) VALUES (?, ?, ?, ?)",
                      (journey_id, next_p_idx, "STATE_DEVIATION", json.dumps(res.get('roadmap', {}))))
                      
    return {
        "status": "success",
        "journey_id": journey_id,
        "snapshot_index": next_idx,
        "replan_triggered": True,
        "roadmap": res.get('roadmap', {})
    }
