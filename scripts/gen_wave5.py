import os, textwrap

files = {}

# ── database/schema.py (redesigned audit hierarchy) ──────────────────────────
files['backend/database/schema.py'] = textwrap.dedent("""
'''
backend/database/schema.py  – V2 Audit Schema
Hierarchy:
  Borrower -> RecoveryJourney -> BorrowerSnapshot (many)
                              -> ModelDecision (many)
                              -> RecoveryPlan (versioned)
                                  -> RecoveryAction (many)

All entities carry versioning fields for full audit traceability.
'''
import datetime
from sqlalchemy import (Column, String, Integer, Float, Boolean,
                        DateTime, ForeignKey, JSON, create_engine)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///./credit_recovery.db')
engine = create_engine(DATABASE_URL, connect_args={'check_same_thread': False}
                        if 'sqlite' in DATABASE_URL else {})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

import os


class Borrower(Base):
    __tablename__ = 'borrower'
    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    journeys = relationship('RecoveryJourney', back_populates='borrower')


class RecoveryJourney(Base):
    __tablename__ = 'recovery_journey'
    id = Column(Integer, primary_key=True, index=True)
    borrower_id = Column(Integer, ForeignKey('borrower.id'), nullable=False)
    started_at = Column(DateTime, default=datetime.datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)
    status = Column(String, default='active')  # active | completed | abandoned
    borrower = relationship('Borrower', back_populates='journeys')
    snapshots = relationship('BorrowerSnapshot', back_populates='journey')
    decisions = relationship('ModelDecision', back_populates='journey')
    plans = relationship('RecoveryPlan', back_populates='journey')


class BorrowerSnapshot(Base):
    '''One row per observed state of the borrower (x_0, x_1, ..., x_T).'''
    __tablename__ = 'borrower_snapshot'
    id = Column(Integer, primary_key=True, index=True)
    journey_id = Column(Integer, ForeignKey('recovery_journey.id'), nullable=False)
    snapshot_index = Column(Integer, nullable=False)   # 0 = initial
    observed_at = Column(DateTime, default=datetime.datetime.utcnow)
    features_json = Column(JSON)
    journey = relationship('RecoveryJourney', back_populates='snapshots')


class ModelDecision(Base):
    __tablename__ = 'model_decision'
    id = Column(Integer, primary_key=True, index=True)
    journey_id = Column(Integer, ForeignKey('recovery_journey.id'), nullable=False)
    snapshot_id = Column(Integer, ForeignKey('borrower_snapshot.id'))
    predicted_default_risk = Column(Float)
    risk_band = Column(String)
    threshold_used = Column(Float)
    recovery_applicable = Column(Boolean)
    model_version = Column(String)
    feature_contract_version = Column(String)
    decided_at = Column(DateTime, default=datetime.datetime.utcnow)
    journey = relationship('RecoveryJourney', back_populates='decisions')


class RecoveryPlan(Base):
    __tablename__ = 'recovery_plan'
    id = Column(Integer, primary_key=True, index=True)
    journey_id = Column(Integer, ForeignKey('recovery_journey.id'), nullable=False)
    plan_version = Column(Integer, default=1)
    solver_used = Column(String)
    solver_version = Column(String)
    constraint_registry_version = Column(String)
    original_risk = Column(Float)
    target_risk = Column(Float)
    total_months = Column(Integer)
    status = Column(String)   # feasible | infeasible_within_horizon | failed
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    actions = relationship('RecoveryAction', back_populates='plan')
    journey = relationship('RecoveryJourney', back_populates='plans')


class RecoveryAction(Base):
    __tablename__ = 'recovery_action'
    id = Column(Integer, primary_key=True, index=True)
    plan_id = Column(Integer, ForeignKey('recovery_plan.id'), nullable=False)
    month = Column(Integer)
    feature_name = Column(String)
    direction = Column(String)
    monthly_change = Column(Float)
    cumulative_target = Column(Float)
    reassessment_date = Column(String)
    plan = relationship('RecoveryPlan', back_populates='actions')


def init_db():
    Base.metadata.create_all(bind=engine)
""").lstrip()

# ── main.py updated to use SolverRouter + wire DB ────────────────────────────
files['backend/main.py'] = textwrap.dedent("""
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
""").lstrip()

# ── tests/test_validator.py ───────────────────────────────────────────────────
files['tests/test_validator.py'] = textwrap.dedent("""
'''Unit tests for FeasibilityGuard (4 gates).'''
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock
from backend.engine.validator import FeasibilityGuard
from backend.engine.constraint_registry import DEFAULT_REGISTRY
from backend.engine.feature_contract import FEATURE_CONTRACT_V2


def make_mock_model(risk_value):
    m = MagicMock()
    m.predict_risk = MagicMock(return_value=np.array([risk_value]))
    return m


BASE_APPLICANT = pd.DataFrame([{
    'AMT_CREDIT': 300000.0,
    'AMT_INCOME_TOTAL': 100000.0,
    'AMT_ANNUITY': 15000.0,
    'DAYS_BIRTH': -12000,
    'DAYS_EMPLOYED': -2000,
    'NAME_EDUCATION_TYPE': 'Higher education',
}])


def make_guard(risk_value):
    model = make_mock_model(risk_value)
    return FeasibilityGuard(model, 0.3, DEFAULT_REGISTRY, FEATURE_CONTRACT_V2)


def test_v_risk_passes_when_below_threshold():
    guard = make_guard(0.20)
    result = guard.validate(BASE_APPLICANT, BASE_APPLICANT)
    assert result.gate_results['V_risk'] is True


def test_v_risk_fails_when_above_threshold():
    guard = make_guard(0.45)
    result = guard.validate(BASE_APPLICANT, BASE_APPLICANT)
    assert result.gate_results['V_risk'] is False
    assert any('V_risk' in v for v in result.violations)


def test_v_structural_dti_violation():
    guard = make_guard(0.20)
    # Annuity = 50% of income -> violates DTI_MAX_001 (40%)
    high_ann = BASE_APPLICANT.copy()
    high_ann['AMT_ANNUITY'] = 50000.0
    result = guard.validate(high_ann, BASE_APPLICANT)
    assert result.gate_results['V_structural'] is False
    assert any('DTI' in v for v in result.violations)


def test_v_actionability_immutable_changed():
    guard = make_guard(0.20)
    changed = BASE_APPLICANT.copy()
    changed['DAYS_BIRTH'] = -9000   # IMMUTABLE changed
    result = guard.validate(changed, BASE_APPLICANT)
    assert result.gate_results['V_actionability'] is False


def test_v_plausibility_income_cap_exceeded():
    guard = make_guard(0.20)
    # Income jump of 200k over 12 months -> max allowed = 5000 * 12 = 60000
    big_jump = BASE_APPLICANT.copy()
    big_jump['AMT_INCOME_TOTAL'] = 300000.0
    result = guard.validate(big_jump, BASE_APPLICANT)
    assert result.gate_results['V_plausibility'] is False


def test_all_gates_pass_for_valid_candidate():
    guard = make_guard(0.20)
    # Small valid annuity reduction within caps
    valid = BASE_APPLICANT.copy()
    valid['AMT_ANNUITY'] = 10000.0   # within 3-10% of 300k credit
    result = guard.validate(valid, BASE_APPLICANT)
    assert result.gate_results['V_risk'] is True
    assert result.gate_results['V_structural'] is True
    assert result.gate_results['V_actionability'] is True


def test_passed_is_conjunction_of_gates():
    guard = make_guard(0.45)  # risk gate will fail
    result = guard.validate(BASE_APPLICANT, BASE_APPLICANT)
    assert result.passed is False
""").lstrip()

# ── tests/test_constraint_registry.py ────────────────────────────────────────
files['tests/test_constraint_registry.py'] = textwrap.dedent("""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.engine.constraint_registry import DEFAULT_REGISTRY


def test_registry_has_six_constraints():
    assert len(DEFAULT_REGISTRY.all_constraints()) == 6


def test_hard_constraints_are_all_hard():
    for c in DEFAULT_REGISTRY.hard_constraints():
        assert c.hard_or_soft == 'hard'


def test_monthly_cap_returns_correct_values():
    assert DEFAULT_REGISTRY.monthly_cap('AMT_INCOME_TOTAL') == 5000.0
    assert DEFAULT_REGISTRY.monthly_cap('AMT_CREDIT') == 50000.0
    assert DEFAULT_REGISTRY.monthly_cap('AMT_ANNUITY') == 2000.0


def test_monthly_cap_returns_none_for_unknown():
    assert DEFAULT_REGISTRY.monthly_cap('DAYS_BIRTH') is None


def test_get_by_id():
    c = DEFAULT_REGISTRY.get('DTI_MAX_001')
    assert c is not None
    assert c.params['max_dti'] == 0.40


def test_summary_is_string():
    s = DEFAULT_REGISTRY.summary()
    assert isinstance(s, str)
    assert 'DTI_MAX_001' in s
""").lstrip()

# ── tests/test_planner.py ─────────────────────────────────────────────────────
files['tests/test_planner.py'] = textwrap.dedent("""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.engine.planner import RecoveryTrajectoryPlanner, MAX_HORIZON
from backend.engine.constraint_registry import DEFAULT_REGISTRY


def make_planner():
    return RecoveryTrajectoryPlanner(registry=DEFAULT_REGISTRY)


BASE = {'AMT_CREDIT': 300000.0, 'AMT_INCOME_TOTAL': 100000.0,
        'AMT_ANNUITY': 15000.0}


def test_no_change_returns_one_month():
    p = make_planner()
    result = p.generate_timeline(BASE, BASE)
    assert result['status'] == 'feasible'
    assert result['total_months'] == 1


def test_small_annuity_change_within_horizon():
    p = make_planner()
    target = BASE.copy()
    target['AMT_ANNUITY'] = 13000.0   # delta = 2000, cap = 2000/mo -> 1 month
    result = p.generate_timeline(BASE, target)
    assert result['status'] == 'feasible'
    assert result['total_months'] == 1


def test_large_income_change_infeasible():
    p = make_planner()
    target = BASE.copy()
    # income jump of 300k, cap=5000/mo -> 60 months needed -> infeasible
    target['AMT_INCOME_TOTAL'] = 400000.0
    result = p.generate_timeline(BASE, target)
    assert result['status'] == 'infeasible_within_horizon'
    assert result['months_required'] > MAX_HORIZON


def test_timeline_length_matches_total_months():
    p = make_planner()
    target = BASE.copy()
    target['AMT_ANNUITY'] = 9000.0   # delta=6000, cap=2000/mo -> 3 months
    result = p.generate_timeline(BASE, target)
    assert result['status'] == 'feasible'
    assert len(result['timeline']) == result['total_months']


def test_final_step_is_marked():
    p = make_planner()
    target = BASE.copy()
    target['AMT_ANNUITY'] = 9000.0
    result = p.generate_timeline(BASE, target)
    assert result['timeline'][-1]['is_final'] is True
""").lstrip()

# ── experiments/01_baseline_models.py ────────────────────────────────────────
files['experiments/01_baseline_models.py'] = textwrap.dedent("""
'''
experiments/01_baseline_models.py
Compares Logistic Regression vs LightGBM on the Home Credit dataset.
Outputs a metrics table (ROC-AUC, PR-AUC, Recall, Precision, F1, Brier).

Usage:
    cd credit_recovery_platform
    .\\venv\\Scripts\\python experiments/01_baseline_models.py
'''
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              recall_score, precision_score, f1_score,
                              brier_score_loss)
import lightgbm as lgb

from backend.models.feature_engineering import build_enriched_dataset, CreditDataTransformer

DATA_DIR = 'data'
NUMERIC = ['AMT_CREDIT', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY',
           'DAYS_BIRTH', 'DAYS_EMPLOYED']
CAT = ['NAME_EDUCATION_TYPE']


def build_pipeline(classifier):
    pre = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                          ('sc', StandardScaler())]), NUMERIC),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                          ('ohe', OneHotEncoder(handle_unknown='ignore'))]), CAT),
    ])
    return Pipeline([('prep', pre), ('clf', classifier)])


def evaluate(name, model, X_test, y_test):
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        'Model': name,
        'ROC-AUC':   round(roc_auc_score(y_test, prob), 4),
        'PR-AUC':    round(average_precision_score(y_test, prob), 4),
        'Recall':    round(recall_score(y_test, pred), 4),
        'Precision': round(precision_score(y_test, pred, zero_division=0), 4),
        'F1':        round(f1_score(y_test, pred, zero_division=0), 4),
        'Brier':     round(brier_score_loss(y_test, prob), 4),
    }


if __name__ == '__main__':
    print('Loading data...')
    df = build_enriched_dataset(DATA_DIR)
    df = df.dropna(subset=['TARGET'])

    use_cols = NUMERIC + CAT
    X = df[[c for c in use_cols if c in df.columns]]
    y = df['TARGET'].astype(int)

    tf = CreditDataTransformer()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    X_train = tf.fit_transform(X_train)
    X_test  = tf.transform(X_test)

    results = []

    lr = build_pipeline(LogisticRegression(max_iter=1000, class_weight='balanced',
                                            random_state=42))
    print('Training Logistic Regression...')
    lr.fit(X_train, y_train)
    results.append(evaluate('Logistic Regression', lr, X_test, y_test))

    lgbm = build_pipeline(lgb.LGBMClassifier(n_estimators=300, class_weight='balanced',
                                               random_state=42, verbosity=-1))
    print('Training LightGBM...')
    lgbm.fit(X_train, y_train)
    results.append(evaluate('LightGBM', lgbm, X_test, y_test))

    print('\\n' + '='*70)
    print('BASELINE MODEL COMPARISON')
    print('='*70)
    res_df = pd.DataFrame(results).set_index('Model')
    print(res_df.to_string())
    print('='*70)
""").lstrip()

# ── experiments/07_solver_benchmark.py ───────────────────────────────────────
files['experiments/07_solver_benchmark.py'] = textwrap.dedent("""
'''
experiments/07_solver_benchmark.py
Compares SLSQP vs BinarySearch vs SolverRouter on:
  validity (did it cross the threshold?),
  structural violations,
  action cost,
  runtime (seconds)

Usage:
    .\\venv\\Scripts\\python experiments/07_solver_benchmark.py
'''
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from backend.models.risk_model import RiskModelAdapter
from backend.engine.constraint_registry import DEFAULT_REGISTRY
from backend.engine.feature_contract import FEATURE_CONTRACT_V2
from backend.engine.solvers.slsqp_solver import SLSQPSolver
from backend.engine.solvers.binary_search_solver import BinarySearchSolver
from backend.engine.solver_router import SolverRouter

TEST_CASES = [
    {'AMT_CREDIT': 500000.0, 'AMT_INCOME_TOTAL': 60000.0, 'AMT_ANNUITY': 45000.0,
     'DAYS_BIRTH': -15000, 'DAYS_EMPLOYED': -500, 'NAME_EDUCATION_TYPE': 'Secondary / secondary special'},
    {'AMT_CREDIT': 800000.0, 'AMT_INCOME_TOTAL': 90000.0, 'AMT_ANNUITY': 55000.0,
     'DAYS_BIRTH': -12000, 'DAYS_EMPLOYED': -200, 'NAME_EDUCATION_TYPE': 'Incomplete higher'},
]


def run(solver, df, label):
    t0 = time.time()
    result = solver.generate_recourse(df)
    elapsed = time.time() - t0
    return {
        'Solver': label,
        'Status': result.get('status', '?'),
        'Validity': result.get('status') == 'success',
        'Cost': round(result.get('cost', float('nan')), 4),
        'Time(s)': round(elapsed, 3),
        'Violations': len(result.get('violations', [])),
    }


if __name__ == '__main__':
    print('Loading model...')
    adapter = RiskModelAdapter()
    adapter.load()

    kwargs = dict(risk_model=adapter, threshold=0.3,
                  registry=DEFAULT_REGISTRY, feature_contract=FEATURE_CONTRACT_V2)
    slsqp   = SLSQPSolver(**kwargs)
    bsearch = BinarySearchSolver(**kwargs)
    router  = SolverRouter(**kwargs)

    rows = []
    for i, case in enumerate(TEST_CASES):
        df = pd.DataFrame([case])
        print(f'\\nTest case {i+1}: risk={adapter.predict_risk(df)[0]:.3f}')
        for solver, label in [(slsqp, 'SLSQP'), (bsearch, 'BinarySearch'), (router, 'SolverRouter')]:
            r = run(solver, df, label)
            r['Case'] = i + 1
            rows.append(r)

    print('\\n' + '='*75)
    print('SOLVER BENCHMARK RESULTS')
    print('='*75)
    print(pd.DataFrame(rows).to_string(index=False))
    print('='*75)
""").lstrip()

# ── frontend/app.py (updated terminology + router-aware UI) ───────────────────
files['frontend/app.py'] = textwrap.dedent("""
import streamlit as st
import requests
import pandas as pd

API_URL = 'http://127.0.0.1:8000'

st.set_page_config(page_title='Credit Recovery Platform', layout='wide')
st.title('Credit Recovery Intelligence Dashboard')
st.caption('Post-decision algorithmic recourse platform — NOT a lender approval engine.')

BAND_COLOR = {'LOW': 'green', 'MODERATE': 'blue', 'ELEVATED': 'orange', 'HIGH': 'red'}

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header('Applicant Simulator')
amt_income  = st.sidebar.number_input('Total Income (AMT_INCOME_TOTAL)', value=80000.0, step=5000.0)
amt_credit  = st.sidebar.number_input('Credit Amount (AMT_CREDIT)',      value=800000.0, step=10000.0)
amt_annuity = st.sidebar.number_input('Annuity Amount (AMT_ANNUITY)',    value=45000.0, step=1000.0)
days_birth  = st.sidebar.number_input('Age in Days (negative)',  value=-15000, max_value=0)
days_emp    = st.sidebar.number_input('Days Employed (negative)', value=-2000, max_value=0)
education   = st.sidebar.selectbox('Education Level',
    ['Secondary / secondary special', 'Higher education',
     'Incomplete higher', 'Lower secondary'])

payload = {'AMT_CREDIT': amt_credit, 'AMT_INCOME_TOTAL': amt_income,
           'AMT_ANNUITY': amt_annuity, 'DAYS_BIRTH': days_birth,
           'DAYS_EMPLOYED': days_emp, 'NAME_EDUCATION_TYPE': education}

# ── Section 1: Risk Assessment ─────────────────────────────────────────────────
st.subheader('1. Default Risk Assessment')
if st.button('Evaluate Risk'):
    try:
        r = requests.post(f'{API_URL}/predict', json=payload)
        if r.status_code == 200:
            data = r.json()
            risk = data['predicted_default_risk']
            band = data['risk_band']
            applicable = data['recovery_assessment_applicable']

            col1, col2, col3 = st.columns(3)
            col1.metric('Predicted Default Risk', f'{risk:.1%}')
            col2.metric('Risk Band', band)
            col3.metric('Model Version', data.get('model_version', 'N/A'))

            color = BAND_COLOR.get(band, 'gray')
            st.markdown(f'**Risk Band: :{color}[{band}]**')

            if applicable:
                st.info('Recovery Assessment Applicable — this applicant may benefit from a recourse roadmap.')
                st.session_state['show_recovery'] = True
                st.session_state['payload'] = payload
            else:
                st.success('Risk below recourse threshold. No recovery roadmap required.')
                st.session_state['show_recovery'] = False
        else:
            st.error(f'API error: {r.text}')
    except requests.exceptions.ConnectionError:
        st.error('Cannot connect to API. Start the FastAPI server first.')

# ── Section 2: Recovery Roadmap ────────────────────────────────────────────────
st.markdown('---')
st.subheader('2. Sequential Recovery Roadmap')

if st.session_state.get('show_recovery', False):
    if st.button('Generate Recovery Roadmap'):
        with st.spinner('Running Solver Router + Trajectory Planner...'):
            try:
                r = requests.post(f'{API_URL}/generate_roadmap',
                                  json=st.session_state['payload'])
                if r.status_code == 200:
                    data = r.json()
                    status = data.get('status')

                    if status == 'success':
                        st.success(f'Solver: **{data.get(\"solver_tier\", data.get(\"solver\", \"?\"))}** — Feasible recourse path found.')
                        col1, col2, col3 = st.columns(3)
                        col1.metric('Original Risk', f'{data[\"original_risk\"]:.1%}')
                        col2.metric('Target Risk',   f'{data[\"new_risk\"]:.1%}')
                        col3.metric('Action Cost',   f'{data[\"cost\"]:.4f}')

                        plan = data.get('sequential_plan', {})
                        plan_status = plan.get('status', '')
                        if plan_status == 'infeasible_within_horizon':
                            st.warning(plan['message'])
                        elif plan.get('timeline'):
                            st.markdown(f'### Recovery Timeline — {plan[\"total_months\"]} months')
                            tiers = ', '.join(data.get('tiers_attempted', []))
                            if tiers:
                                st.caption(f'Solver tiers attempted: {tiers}')
                            for step in plan['timeline']:
                                label = f'Month {step[\"month\"]} — {step[\"reassessment_date\"]}'
                                if step['is_final']:
                                    label += ' ✅ Final'
                                with st.expander(label, expanded=(step['month'] == 1)):
                                    for a in step['actions']:
                                        st.markdown(f'- {a[\"label\"]}')

                        st.markdown('### Validation Gates')
                        gates = data.get('validation_gates', {})
                        for gate, passed in gates.items():
                            icon = '✅' if passed else '❌'
                            st.write(f'{icon} {gate}')

                        st.markdown('### Audit View (Original vs Target State)')
                        st.dataframe(pd.DataFrame([data['original_state'],
                                                    data['new_state']],
                                                   index=['Original', 'Target']))

                    elif status == 'eligible':
                        st.success('Risk already below threshold — no recourse needed.')
                    else:
                        st.error(data.get('message', 'Recourse failed.'))
                        for v in data.get('violations', []):
                            st.write(f'  ⚠️ {v}')
                else:
                    st.error(f'API error: {r.text}')
            except requests.exceptions.ConnectionError:
                st.error('Cannot connect to API.')
else:
    st.info('First run the Risk Assessment above. The roadmap generator activates for ELEVATED / HIGH risk applicants.')

# ── Section 3: Constraint Registry ────────────────────────────────────────────
st.markdown('---')
with st.expander('View Constraint Registry (all active rules)'):
    try:
        r = requests.get(f'{API_URL}/constraints')
        if r.status_code == 200:
            constraints = r.json()['constraints']
            st.dataframe(pd.DataFrame(constraints))
    except Exception:
        st.write('API not reachable.')
""").lstrip()

for path, content in files.items():
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(content)
    print(f'  wrote {path}')

print('Wave 5 files written.')
