from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./credit_recovery.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class BorrowerState(Base):
    __tablename__ = "borrower_states"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String, index=True, unique=True)
    current_risk_score = Column(Float)
    is_eligible = Column(String)
    original_features = Column(JSON)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class RecoveryRoadmap(Base):
    __tablename__ = "recovery_roadmaps"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String, index=True)
    milestones = Column(JSON)
    target_risk_score = Column(Float)
    action_cost = Column(Float)
    status = Column(String) # 'active', 'completed', 'abandoned'
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)

if __name__ == '__main__':
    print("Creating tables...")
    try:
        init_db()
        print("Database initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize database: {e}")
