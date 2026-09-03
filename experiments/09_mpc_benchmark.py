"""
experiments/09_mpc_benchmark.py

Phase 4 MPC Benchmark — compares three recourse regimes:
  A. One-shot    (a_0 = x* - x_0 at t=0; no further intervention)
  B. Sequential  (distribute plan target evenly over T months; no replanning)
  C. MPC         (event-triggered closed-loop replanning)

All regimes share the exact same CRN disturbance stream per applicant (omega_i).

Metrics (per review points 3, 4, 5, 9):
  terminal_validity     : I[f(x_T) <= tau_T]
  trajectory_survival   : conditional on recovery, fraction of months in valid state
  recovery_availability : N_ever_recovered / N
  valid_state_occupancy : (1/T) * sum_t I[f(x_t) <= tau_t]   <- unconditional
  cumulative_cost       : sum_t ||a_tilde_t|| / disposable_income
  action_exec_rate      : months with ||a_tilde_t|| > 0 / T
  replan_triggers       : N_trigger (event fired)
  replan_attempts       : N_attempt (solver called)
  replan_successes      : N_success (solver found plan)
  replan_failures       : N_failure (solver called, failed)
  instability_cumulative: sum_t I_t (normalised L1 between consecutive targets)
  solver_feasibility    : N_success / N_attempt

Point 7: all regimes use same initial tau_target for fairness.
Point 8: CRN seeds cover action execution, income shocks, debt shocks, policy draws.
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.models.risk_model import RiskModelAdapter
from backend.engine.constraint_registry import ConstraintRegistry
from backend.engine.feature_contract import FEATURE_CONTRACT_V3
from backend.engine.solver_router import SolverRouter
from backend.engine.simulator import EnvironmentSimulator, DisturbanceConfig
from backend.engine.mpc_controller import MPCController

# ── Configuration ────────────────────────────────────────────────────────────
N_APPLICANTS = 5        # Stage B: deterministic sanity (increase for ablation)
T_HORIZON    = 12
DELTA_SAFETY = 0.05
BASE_TAU     = 0.30

# Stage B: zero disturbance. Change to "mild" / "moderate" / "severe" for ablations.
ENVIRONMENT  = "zero"


# ── Environment factory ───────────────────────────────────────────────────────
def _get_config(env_name: str) -> DisturbanceConfig:
    if env_name == "zero":
        return DisturbanceConfig.zero_disturbance()
    if env_name == "mild":
        return DisturbanceConfig.low_disturbance()
    if env_name == "moderate":
        cfg = DisturbanceConfig.moderate_disturbance()
        cfg.policy_shift = True
        return cfg
    if env_name == "severe":
        return DisturbanceConfig(
            p_miss=0.20, beta_alpha=5.0, beta_beta=1.5,
            p_income_shock=0.10, p_debt_shock=0.10,
            income_shock_min=0.50, income_shock_max=0.80,
            shifted_threshold=0.25, policy_shift=True,
        )
    raise ValueError(f"Unknown environment: {env_name!r}")


# ── Per-regime runner ─────────────────────────────────────────────────────────
def run_regime(
    regime_type: str,
    applicant_orig: pd.DataFrame,
    adapter,
    registry,
    train_df: pd.DataFrame,
    config: DisturbanceConfig,
    crn_seed: int,
) -> dict:
    """
    Simulates one applicant through T_HORIZON months under a given regime.

    Point 7: all regimes start from identical x_0 and use the same tau_target
    for the initial plan generation, so differences are purely due to adaptation.
    Point 8: CRN — same rng seed used for all regimes of the same applicant.
    """
    sim = EnvironmentSimulator()
    rng = np.random.RandomState(crn_seed)   # CRN: identical disturbance stream

    # Bug-9 fix: wired train_df so DiCE fallback works
    router = SolverRouter(
        risk_model=adapter,
        threshold=BASE_TAU,
        registry=registry,
        feature_contract=FEATURE_CONTRACT_V3,
        training_data=train_df,
    )

    # Point 7: all regimes target the same safety margin at t=0
    tau_target_init = BASE_TAU - DELTA_SAFETY

    # ── Initial plan (shared across all regimes for fairness) ────────────────
    initial_res = router.generate_recourse(
        applicant_orig,
        target_threshold=tau_target_init,
    )
    static_target: dict | None = None
    if initial_res.get("status") in ("success", "eligible"):
        new_state = initial_res.get("new_state", {})
        if new_state:
            actionable_classes = (
                "CONDITIONALLY_ACTIONABLE", "ACTIONABLE_STATE", "ACTIONABLE_BEHAVIOUR"
            )
            actionable_feats = [
                f for f, d in FEATURE_CONTRACT_V3.items()
                if (d.actionable or d.feature_class in actionable_classes)
                and f in applicant_orig.columns
            ]
            static_target = {f: float(new_state[f]) for f in actionable_feats if f in new_state}

    # ── MPC controller (only used for 'mpc' regime) ──────────────────────────
    mpc = MPCController(
        risk_model=adapter,
        base_threshold=BASE_TAU,
        feature_contract=FEATURE_CONTRACT_V3,
        solver_router=router,
        delta_safety=DELTA_SAFETY,
        gamma_stability=0.5,
        delta_r=0.03,
        delta_x=0.10,
    )
    # Seed MPC with the same initial plan so t=0 starts identically
    if static_target and regime_type == "mpc":
        mpc.previous_plan_target = dict(static_target)

    # ── Counters ─────────────────────────────────────────────────────────────
    replan_triggers   = 0
    replan_attempts   = 0
    replan_successes  = 0
    months_action     = 0
    cumulative_cost   = 0.0
    cumulative_instab = 0.0
    ever_recovered    = False
    recovery_month    = None
    months_valid_after_recovery = 0
    months_valid_total = 0

    current_state = applicant_orig.copy()
    current_tau   = BASE_TAU

    for t in range(T_HORIZON):
        current_tau = sim.policy_env.step(t, config, rng=rng)

        # ── Compute current risk ─────────────────────────────────────────────
        current_risk = float(adapter.predict_risk(current_state)[0])

        # ── Valid-state occupancy ────────────────────────────────────────────
        if current_risk <= current_tau:
            months_valid_total += 1
            if not ever_recovered:
                ever_recovered = True
                recovery_month = t
            months_valid_after_recovery += 1

        # ── Choose action ────────────────────────────────────────────────────
        if regime_type == "mpc":
            step_res = mpc.get_action(current_state, t, T_HORIZON, current_tau)
            a_t = step_res["action_t"]
            if step_res["replan_trigger"]:
                replan_triggers += 1
            if step_res["replan_attempt"]:
                replan_attempts  += 1
                replan_successes += int(step_res["replan_success"])
            cumulative_instab += step_res["instability_norm"]

        elif regime_type == "sequential":
            a_t = {}
            if static_target:
                remaining = max(1, T_HORIZON - t)
                for f, targ in static_target.items():
                    cur = float(current_state.iloc[0].get(f, 0))
                    delta = (targ - cur) / remaining
                    if abs(delta) > 1e-9:
                        a_t[f] = delta

        else:  # one-shot: full intervention at t=0 only
            a_t = {}
            if t == 0 and static_target:
                for f, targ in static_target.items():
                    cur = float(current_state.iloc[0].get(f, 0))
                    delta = targ - cur
                    if abs(delta) > 1e-9:
                        a_t[f] = delta

        # ── Simulate step (CRN synchronized) ────────────────────────────────
        next_state, log_t = sim.step(current_state, a_t, t, config, rng=rng)

        realized = log_t.get("realized_action", {})
        realized_norm = sum(abs(v) for v in realized.values())
        if realized_norm > 1e-9:
            months_action += 1

        # Cost scaled by disposable income
        inc = float(current_state.iloc[0].get("AMT_INCOME_TOTAL", 1)) or 1.0
        cumulative_cost += realized_norm / inc

        current_state = next_state

    # ── Terminal evaluation ───────────────────────────────────────────────────
    terminal_risk     = float(adapter.predict_risk(current_state)[0])
    terminal_validity = int(terminal_risk <= current_tau)

    # Point 9: unconditional valid-state occupancy  O_i = months_valid / T
    valid_occupancy = months_valid_total / T_HORIZON

    # Point 9: conditional trajectory survival  S_i (conditional on recovery)
    if ever_recovered and recovery_month is not None:
        denom = T_HORIZON - recovery_month
        trajectory_survival = months_valid_after_recovery / denom if denom > 0 else 0.0
    else:
        trajectory_survival = 0.0

    solver_feasibility = (
        replan_successes / replan_attempts if replan_attempts > 0 else float("nan")
    )

    return {
        # Primary outcomes
        "terminal_validity":    terminal_validity,
        "terminal_risk":        terminal_risk,
        "valid_occupancy":      valid_occupancy,
        "trajectory_survival":  trajectory_survival,
        "ever_recovered":       int(ever_recovered),
        "recovery_month":       recovery_month,
        # Cost
        "cumulative_cost":      cumulative_cost,
        "action_exec_rate":     months_action / T_HORIZON,
        # Replan diagnostics (Point 4)
        "replan_triggers":      replan_triggers,
        "replan_attempts":      replan_attempts,
        "replan_successes":     replan_successes,
        "replan_failures":      replan_attempts - replan_successes,
        "solver_feasibility":   solver_feasibility,
        "instability_cumulative": cumulative_instab,
        # Context
        "initial_risk":         float(adapter.predict_risk(applicant_orig)[0]),
        "initial_plan_found":   int(static_target is not None),
    }


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    adapter = RiskModelAdapter()
    adapter.load()

    test_df  = pd.read_csv("data/test_reference.csv").dropna(subset=["TARGET"])
    train_df = pd.read_csv("data/train_reference.csv").dropna(subset=["TARGET"]).head(500)

    registry = ConstraintRegistry()
    config   = _get_config(ENVIRONMENT)

    # Use applicants already above threshold where Phase-3 solver finds feasible recourse
    risks      = adapter.predict_risk(test_df)
    above_mask = risks > BASE_TAU
    sample_df  = test_df[above_mask].head(N_APPLICANTS)

    regimes      = ["one-shot", "sequential", "mpc"]
    all_results  = {r: [] for r in regimes}

    print(f"Environment  : {ENVIRONMENT}  (disturbance-free = {ENVIRONMENT == 'zero'})")
    print(f"Applicants   : {len(sample_df)}")
    print(f"T_horizon    : {T_HORIZON} months")
    print(f"delta_safety : {DELTA_SAFETY}  (tau_target = {BASE_TAU - DELTA_SAFETY})")
    print(f"MPC delta_r  : 0.03   delta_x : 0.10")
    print()

    for i in range(len(sample_df)):
        applicant = sample_df.iloc[[i]]
        seed      = 42 + i
        init_risk = float(adapter.predict_risk(applicant)[0])
        print(f"-- Applicant {i+1}/{len(sample_df)}  initial_risk={init_risk:.3f}", flush=True)

        for regime in regimes:
            res = run_regime(regime, applicant, adapter, registry, train_df, config, seed)
            all_results[regime].append(res)
            print(
                f"   [{regime:10s}]  "
                f"valid={res['terminal_validity']}  "
                f"occupancy={res['valid_occupancy']:.0%}  "
                f"cost={res['cumulative_cost']:.4f}  "
                f"exec={res['action_exec_rate']:.0%}  "
                f"triggers={res['replan_triggers']}  "
                f"attempts={res['replan_attempts']}  "
                f"successes={res['replan_successes']}  "
                f"instab={res['instability_cumulative']:.3f}",
                flush=True,
            )

    # ── Summary table ─────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print(f"  PHASE 4 RESULTS  N={N_APPLICANTS}  T={T_HORIZON}  ENV={ENVIRONMENT}")
    print("=" * 72)
    hdr = f"  {'Metric':<34} {'One-Shot':>10} {'Sequential':>12} {'MPC':>8}"
    print(hdr)
    print("-" * 72)

    def col(regime, key):
        vals = [r[key] for r in all_results[regime]
                if not isinstance(r[key], float) or not np.isnan(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    def fmt(v, spec):
        if isinstance(v, float) and np.isnan(v):
            return "   N/A"
        try:
            return spec.format(v)
        except Exception:
            return str(v)

    rows = [
        ("Terminal Validity",         "terminal_validity",      "{:.1%}"),
        ("Valid-State Occupancy",      "valid_occupancy",        "{:.1%}"),
        ("Recovery Availability",      "ever_recovered",         "{:.1%}"),
        ("Conditional Traj Survival",  "trajectory_survival",    "{:.1%}"),
        ("Avg Cumulative Cost",        "cumulative_cost",        "{:.4f}"),
        ("Action Execution Rate",      "action_exec_rate",       "{:.1%}"),
        ("Initial Plan Found",         "initial_plan_found",     "{:.1%}"),
        ("-- MPC only --",             None,                     ""),
        ("  Replan Triggers",          "replan_triggers",        "{:.1f}"),
        ("  Replan Attempts",          "replan_attempts",        "{:.1f}"),
        ("  Replan Successes",         "replan_successes",       "{:.1f}"),
        ("  Replan Failures",          "replan_failures",        "{:.1f}"),
        ("  Solver Feasibility Rate",  "solver_feasibility",     "{:.1%}"),
        ("  Instability (cumulative)", "instability_cumulative", "{:.3f}"),
    ]

    for label, key, spec in rows:
        if key is None:
            print(f"  {label}")
            continue
        c_os  = col("one-shot",   key)
        c_seq = col("sequential", key)
        c_mpc = col("mpc",        key)
        print(f"  {label:<34} {fmt(c_os, spec):>10} {fmt(c_seq, spec):>12} {fmt(c_mpc, spec):>8}")

    print("=" * 72)
    print()
    print("Point 10 sanity check:")
    print("  Under ZERO disturbance + on-plan execution, MPC should show:")
    print("  - replan_triggers ~ 1 (initial plan only)")
    print("  - action_exec_rate > 0%  (non-empty actions)")
    print("  - valid_occupancy increases over time")
    print()
    print("NOTE: All disturbance parameters are labelled simulation assumptions,")
    print("      not empirically estimated real-world frequencies.")
