# Credit Recovery Intelligence Platform

An algorithmic recourse platform that acts as a post-decision recovery operating layer. Instead of just offering a standard 'declined' reason code, this platform recalculates sequential, high-integrity, and auditable recovery paths for temporarily ineligible credit applicants without exposing proprietary lender risk thresholds.

## Overview
Built based on advanced research in sequential algorithmic recourse and closed-loop control principles, the system consists of:
*   **Cost-Aware Solver**: A constraint-based mathematical optimization engine that strictly bounds feature adjustments based on structural financial logic (e.g., Debt-to-Income, Annuity-to-Credit limits).
*   **Closed-Loop Sequential Planner**: Discretizes recovery paths into actionable, month-by-month timelines honoring anti-gaming limitations.
*   **Risk Model Adapter**: A LightGBM classifier integrating advanced data pipelines (IQR outlier clipping, VIF multicollinearity checks).
*   **Borrower Dashboard**: A Streamlit interface simulating the user-facing roadmap.
*   **FastAPI Engine**: The backend architecture exposing predictions and recourse roadmaps.

## Architecture Stack
*   **Backend:** Python 3, FastAPI, SciPy, Scikit-Learn, LightGBM, Pandas
*   **Frontend:** Streamlit
*   **Database:** PostgreSQL (with SQLite fallback) via SQLAlchemy
*   **Dataset Setup:** Home Credit Default Risk

## Local Installation

1. **Clone the repository:**
   \\ash
   git clone https://github.com/akashchatterjee635/credit-recovery-platform.git
   cd credit-recovery-platform
   \
2. **Set up the virtual environment:**
   \\ash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   \
3. **Get the Dataset:**
   The training pipeline expects the Home Credit Default Risk dataset from Kaggle.
   Ensure your \kaggle.json\ is configured, then run:
   \\ash
   python fetch_kagglehub.py
   \
4. **Train the Model:**
   \\ash
   python backend/models/risk_model.py
   \
5. **Run the Application:**
   Open two terminals.
   
   *Terminal 1 (Backend API):*
   \\ash
   uvicorn backend.main:app --reload
   \   
   *Terminal 2 (Frontend Dashboard):*
   \\ash
   streamlit run frontend/app.py
   \