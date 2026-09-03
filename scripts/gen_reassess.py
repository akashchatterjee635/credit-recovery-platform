import re

main_path = "backend/main.py"
with open(main_path, "r") as f:
    content = f.read()

# Need to update SQLite schema to track snapshot_index, plan_version, replan_reason
schema_updates = """
# Phase 4 schema updates
cursor.execute('''
    CREATE TABLE IF NOT EXISTS snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        journey_id INTEGER,
        snapshot_index INTEGER,
        state_data TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')
cursor.execute('''
    CREATE TABLE IF NOT EXISTS recovery_plans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        journey_id INTEGER,
        plan_version INTEGER,
        parent_plan_id INTEGER,
        replan_reason TEXT,
        roadmap_data TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')
"""
if "recovery_plans" not in content:
    content = content.replace("conn.commit()", schema_updates + "conn.commit()")

# Implement POST /journeys/{journey_id}/reassess
reassess_endpoint = """
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
"""
if "/reassess" not in content:
    content += "\n" + reassess_endpoint

with open(main_path, "w") as f:
    f.write(content)
print("Updated main.py with schema and /reassess")
