# Credit Recovery Intelligence Platform

An algorithmic recourse platform that acts as a post-decision recovery operating layer. Instead of just offering a standard 'declined' reason code, this platform recalculates sequential, high-integrity, and auditable recovery paths for temporarily ineligible credit applicants without exposing proprietary lender risk thresholds.

## Overview
Built based on advanced research in sequential algorithmic recourse and closed-loop control principles, the system consists of:
*   **Temporal & Deep Risk Modeling**: A PyTorch-based sequence modeling layer featuring Temporal Convolutional Networks (TCN) fused with FT-Transformers. This architecture tracks longitudinal borrower behavior (e.g., historical repayment traces) and applies temporal attention to explicitly differentiate historical delinquency from recent behavioral deterioration.
*   **Model-Agnostic Adapter Contract**: Supports plug-and-play risk evaluators via the `BaseRiskAdapter`, featuring a calibrated LightGBM baseline and a DeepRiskAdapter enforcing strict historical state immutability during optimization.
*   **Cost-Aware Solver**: A constraint-based mathematical optimization engine that strictly bounds feature adjustments based on structural financial logic (e.g., Debt-to-Income, Annuity-to-Credit limits).
*   **Closed-Loop Sequential Planner**: Discretizes recovery paths into actionable timelines using Model Predictive Control (MPC) event triggers, honoring anti-gaming limitations.
*   **Borrower Dashboard & FastAPI Engine**: Full-stack integration exposing the recourse engine to users.

## Architecture Stack
*   **Backend:** Python 3, FastAPI, SciPy, Scikit-Learn, PyTorch, LightGBM, Pandas
*   **Frontend:** Streamlit
*   **Database:** PostgreSQL (with SQLite fallback) via SQLAlchemy
*   **Dataset Setup:** Home Credit Default Risk

## Local Installation

The reproducible quick path is:

```bash
make setup
make test
make train-smoke
make run
```

`train-smoke` creates a deterministic demo artifact so `/predict` can run without
raw Kaggle data. It is architecture-validation data only and must not be used for
predictive-performance claims.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/akashchatterjee635/credit-recovery-platform.git
   cd credit-recovery-platform
   ```
2. **Set up the virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -e ".[static,deep,ui,dev]"
   ```
3. **Get the Dataset:**
   The training pipeline expects the Home Credit Default Risk dataset from Kaggle.
   Ensure your `kaggle.json` is configured, then run:
   ```bash
   python fetch_kagglehub.py
   ```
4. **Train the Models:**
   *For the LightGBM Baseline:*
   ```bash
   python -m scripts.train_static --config configs/static.yaml
   ```
   *For the Temporal Deep Learning Pipeline:*
   ```bash
   python -m scripts.build_temporal_data --config configs/temporal.yaml
   python -m scripts.train_temporal --config configs/deep.yaml
   python -m scripts.evaluate --run-id <run-id>
   ```
5. **Run the Application:**
   Open two terminals.
   
   *Terminal 1 (Backend API):*
   ```bash
   uvicorn backend.main:app --reload
   ```
   *Terminal 2 (Frontend Dashboard):*
   ```bash
   streamlit run frontend/app.py
   ```

## Data, migrations, and evidence

See `DATASET.md` for source, licensing, checksums, and temporal availability.
Production temporal tensors are built only from genuine pre-cutoff auxiliary-table
events; the synthetic generator lives under `experiments/synthetic_smoke_test/`.

Apply schema changes with `alembic upgrade head`. Benchmark runs are written to
`experiments/outputs/<run_id>/` with configuration, applicant-level results,
environment metadata, summary intervals, and a report. Governance limitations are
documented in `docs/MODEL_CARD.md` and `docs/RECOURSE_CARD.md`.
