from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from backend.models.risk_model import RiskModelAdapter
from backend.engine.solver import CostAwareSolver
from backend.engine.planner import SequentialPlanner
import sys

app = FastAPI(title="Credit Recovery Intelligence API")

# Initialize models
try:
    risk_adapter = RiskModelAdapter()
    risk_adapter.load()
    solver = CostAwareSolver(risk_adapter, threshold=0.3)
    planner = SequentialPlanner()
except Exception as e:
    print(f"Warning: Model could not be loaded on startup. {e}")
    solver = None
    planner = None

class ApplicantData(BaseModel):
    AMT_CREDIT: float
    AMT_INCOME_TOTAL: float
    AMT_ANNUITY: float
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    NAME_EDUCATION_TYPE: str

@app.get("/")
def read_root():
    return {"message": "Credit Recovery Intelligence API is running."}

@app.post("/predict")
def predict_risk(applicant: ApplicantData):
    if solver is None or solver.risk_model.model is None:
        raise HTTPException(status_code=503, detail="Risk model not loaded.")
        
    df = pd.DataFrame([applicant.model_dump()])
    risk = solver.risk_model.predict_risk(df)[0]
    return {"risk_score": float(risk), "approved": risk <= solver.threshold}

@app.post("/generate_roadmap")
def generate_roadmap(applicant: ApplicantData):
    if solver is None or planner is None:
        raise HTTPException(status_code=503, detail="Risk model or planner not loaded.")
        
    df = pd.DataFrame([applicant.model_dump()])
    result = solver.generate_recourse(df)
    
    if result["status"] == "success":
        # Pass the one-shot solver result to the sequential planner
        sequential_plan = planner.generate_timeline(result["original_state"], result["new_state"])
        result["sequential_plan"] = sequential_plan
        
    return result
