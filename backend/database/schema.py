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
