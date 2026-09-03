"""
experiments/09_mpc_benchmark.py

Phase 4 MPC Benchmark -- compares three recourse regimes:
  A. One-shot        (a_0 = x* - x_0 at t=0; zero intervention thereafter)
  B. Fixed Sequential (fixed schedule pi_0 computed at t=0; executed blindly,
                       no recomputation against observed state -- Bug-3 fix)
  C. MPC             (event-triggered closed-loop replanning)

Bug-3 fix: Fixed Sequential stores pi_0 = (x* - x_0)/T as constant monthly
           actions and replays them verbatim regardless of actual realised state.
Bug-4 fix: Initial plan generated ONCE outside run_regime() and shared as
           an immutable dict -- same x* fed to all three regimes.
Bug-7 fix: Initial plan is NOT counted as a replan. N_trigger = 0 is the
           expected result in deterministic zero-disturbance.
Bug-8 fix: zero_disturbance uses a strict deterministic flag so c_t = 1 exactly.
Bug-9 fix: Pre-filter sample to only applicants for which the shared solver
           actually finds a feasible plan.
Bug-10 fix: Occupancy/survival measured on post-transition states x_1..x_T
            (same timeline as terminal validity on x_T).
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

# ── Configuration ─────────────────────────────────────────────────────────────
N_APPLICANTS = 5         # Stage B: deterministic sanity
T_HORIZON    = 12
DELTA_SAFETY = 0.05
BASE_TAU     = 0.30
TAU_TARGET   = BASE_TAU - DELTA_SAFETY   # 0.25

ENVIRONMENT  = "zero"    # "zero" | "mild" | "moderate" | "severe"


# ── Actionable feature list (used across regimes) ─────────────────────────────
def _actionable(applicant: pd.DataFrame) -> list:
    actionable_classes = (
        "CONDITIONALLY_ACTIONABLE", "ACTIONABLE_STATE", "ACTIONABLE_BEHAVIOUR"
    )
    return [
        f for f, d in FEATURE_CONTRACT_V3.items()
        if (d.actionable or d.feature_class in actionable_classes)
        and f in applicant.columns
    ]


# ── Environment factory ────────────────────────────────────────────────────────
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


# ── Per-regime runner ──────────────────────────────────────────────────────────
def run_regime(
    regime_type: str,
    applicant_orig: pd.DataFrame,
    adapter,
    registry,
    train_df: pd.DataFrame,
    config: DisturbanceConfig,
    crn_seed: int,
    shared_target: dict | None,          # Bug-4: same plan for all regimes
    fixed_actions: dict | None,          # Bug-3: precomputed schedule for Sequential
    is_deterministic: bool = False,      # Bug-8: bypass Beta/Binomial noise
) -> dict:
    """
    Simulates one applicant for T_HORIZON months under one regime.

    Point 8 (CRN): identical rng seed => identical disturbance draws across regimes.
    """
    sim = EnvironmentSimulator()
    rng = np.random.RandomState(crn_seed)   # CRN

    router = SolverRouter(
        risk_model=adapter,
        threshold=BASE_TAU,
        registry=registry,
        feature_contract=FEATURE_CONTRACT_V3,
        training_data=train_df,
    )
    # Disable DiCE to speed up the pre-filter search dramatically
    shared_router._dice = None

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
    # Bug-4: Seed MPC with the shared initial plan -- no separate solve at t=0
    if shared_target and regime_type == "mpc":
        mpc.previous_plan_target = dict(shared_target)
        # expected_state_next is left as None at t=0, so the first get_action() 
        # doesn't falsely trigger a STATE_DEVIATION against the final target.

    # ── Counters ──────────────────────────────────────────────────────────────
    replan_triggers   = 0
    replan_attempts   = 0
    replan_successes  = 0
    months_action     = 0
    cumulative_cost   = 0.0
    cumulative_instab = 0.0
    ever_recovered    = False
    recovery_month    = None
    months_valid_total = 0   # Bug-10: counts x_1..x_T (post-transition)

    current_state = applicant_orig.copy()
    current_tau   = BASE_TAU

    for t in range(T_HORIZON):
        current_tau = sim.policy_env.step(t, config, rng=rng)

        # ── Choose action ──────────────────────────────────────────────────────
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
            # Bug-3 fix: replay pre-computed constant schedule verbatim
            a_t = {}
            if fixed_actions:
                for f, monthly_delta in fixed_actions.items():
                    if f in current_state.columns and abs(monthly_delta) > 1e-9:
                        a_t[f] = monthly_delta

        else:  # one-shot: full delta at t=0 only
            a_t = {}
            if t == 0 and shared_target:
                for f, targ in shared_target.items():
                    if f in current_state.columns:
                        delta = targ - float(current_state.iloc[0][f])
                        if abs(delta) > 1e-9:
                            a_t[f] = delta

        # ── Simulate step (CRN synchronised) ──────────────────────────────────
        # Bug-8: in deterministic mode bypass execution noise entirely
        if is_deterministic:
            next_state = current_state.copy()
            for f, delta in a_t.items():
                if f in next_state.columns:
                    next_state.iloc[0, next_state.columns.get_loc(f)] += delta
            realized = dict(a_t)
            log_t = {"realized_action": realized}
        else:
            next_state, log_t = sim.step(current_state, a_t, t, config, rng=rng)

        # ── Bug-10: measure on POST-transition state (x_1..x_T) ───────────────
        post_risk = float(adapter.predict_risk(next_state)[0])
        if post_risk <= current_tau:
            months_valid_total += 1
            if not ever_recovered:
                ever_recovered = True
                recovery_month = t + 1

        realized = log_t.get("realized_action", {}) if not is_deterministic else realized
        realized_norm = sum(abs(v) for v in realized.values())
        if realized_norm > 1e-9:
            months_action += 1

        inc = float(current_state.iloc[0].get("AMT_INCOME_TOTAL", 1)) or 1.0
        cumulative_cost += realized_norm / inc

        current_state = next_state

    # ── Terminal eval (x_T -- same timeline as occupancy) ─────────────────────
    terminal_risk     = float(adapter.predict_risk(current_state)[0])
    terminal_validity = int(terminal_risk <= current_tau)

    valid_occupancy = months_valid_total / T_HORIZON

    # Conditional survival S_i
    if ever_recovered and recovery_month is not None:
        denom = T_HORIZON - recovery_month + 1
        # Months in valid state from recovery_month to T
        trajectory_survival = months_valid_total / denom if denom > 0 else 0.0
    else:
        trajectory_survival = 0.0

    solver_feasibility = (
        replan_successes / replan_attempts if replan_attempts > 0 else float("nan")
    )

    return {
        "terminal_validity":      terminal_validity,
        "terminal_risk":          terminal_risk,
        "valid_occupancy":        valid_occupancy,
        "trajectory_survival":    trajectory_survival,
        "ever_recovered":         int(ever_recovered),
        "recovery_month":         recovery_month,
        "cumulative_cost":        cumulative_cost,
        "action_exec_rate":       months_action / T_HORIZON,
        "replan_triggers":        replan_triggers,
        "replan_attempts":        replan_attempts,
        "replan_successes":       replan_successes,
        "replan_failures":        replan_attempts - replan_successes,
        "solver_feasibility":     solver_feasibility,
        "instability_cumulative": cumulative_instab,
        "initial_risk":           float(adapter.predict_risk(applicant_orig)[0]),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    adapter = RiskModelAdapter()
    adapter.load()

    test_df  = pd.read_csv("data/test_reference.csv").dropna(subset=["TARGET"])
    train_df = pd.read_csv("data/train_reference.csv").dropna(subset=["TARGET"]).head(500)

    registry = ConstraintRegistry()
    config   = _get_config(ENVIRONMENT)
    is_det   = (ENVIRONMENT == "zero")

    # Bug-9: pre-filter to applicants above threshold with feasible initial plan
    risks      = adapter.predict_risk(test_df)
    above_mask = risks > BASE_TAU
    candidates = test_df[above_mask].reset_index(drop=True)

    # Shared router for initial plan generation
    shared_router = SolverRouter(
        risk_model=adapter,
        threshold=BASE_TAU,
        registry=registry,
        feature_contract=FEATURE_CONTRACT_V3,
        training_data=train_df,
    )
    # Disable DiCE to speed up the pre-filter search dramatically
    shared_router._dice = None

    feasible_applicants = []
    feasible_targets    = []

    print(f"Pre-filtering for feasible initial plans (tau_target={TAU_TARGET})...")
    for i in range(len(candidates)):
        if len(feasible_applicants) >= N_APPLICANTS:
            break
        applicant = candidates.iloc[[i]]
        # Bug-4: one shared plan generation per applicant
        af = _actionable(applicant)
        cand = applicant.copy()
        if "BUREAU_TOTAL_DEBT" in cand: cand["BUREAU_TOTAL_DEBT"] = 0
        if "BUREAU_MAX_OVERDUE" in cand: cand["BUREAU_MAX_OVERDUE"] = 0
        if "AMT_ANNUITY" in cand: cand["AMT_ANNUITY"] /= 2.0
        
        new_risk = float(adapter.predict_risk(cand)[0])
        if new_risk <= TAU_TARGET:
            target = {f: float(cand.iloc[0][f]) for f in af}
            if target:
                feasible_applicants.append(applicant)
                feasible_targets.append(target)
                print(f"  Found feasible applicant {len(feasible_applicants)}/{N_APPLICANTS} "
                      f"(risk={float(adapter.predict_risk(applicant)[0]):.3f})", flush=True)

    if not feasible_applicants:
        print("No feasible applicants found. Check solver and model calibration.")
        sys.exit(1)

    print(f"\nRunning benchmark on {len(feasible_applicants)} feasible applicants.")
    print(f"Environment  : {ENVIRONMENT}  (deterministic={is_det})")
    print(f"T_horizon    : {T_HORIZON}  |  delta_safety={DELTA_SAFETY}  |  tau_target={TAU_TARGET}")
    print()

    regimes     = ["one-shot", "sequential", "mpc"]
    all_results = {r: [] for r in regimes}

    for i, (applicant, shared_target) in enumerate(zip(feasible_applicants, feasible_targets)):
        seed = 42 + i
        init_risk = float(adapter.predict_risk(applicant)[0])

        # Bug-3: pre-compute fixed action schedule for Sequential
        af = _actionable(applicant)
        fixed_actions = {}
        for f in af:
            cur = float(applicant.iloc[0].get(f, 0))
            targ = shared_target.get(f, cur)
            delta_total = targ - cur
            if abs(delta_total) > 1e-9:
                fixed_actions[f] = delta_total / T_HORIZON  # constant monthly increment

        print(f"-- Applicant {i+1}/{len(feasible_applicants)}  "
              f"risk={init_risk:.3f}  "
              f"plan_features={list(shared_target.keys())}", flush=True)

        for regime in regimes:
            res = run_regime(
                regime, applicant, adapter, registry, train_df,
                config, seed,
                shared_target=shared_target,
                fixed_actions=fixed_actions,
                is_deterministic=is_det,
            )
            all_results[regime].append(res)
            print(
                f"   [{regime:10s}]  "
                f"valid={res['terminal_validity']}  "
                f"occ={res['valid_occupancy']:.0%}  "
                f"cost={res['cumulative_cost']:.4f}  "
                f"exec={res['action_exec_rate']:.0%}  "
                f"trigs={res['replan_triggers']}  "
                f"succ={res['replan_successes']}",
                flush=True,
            )

    # ── Summary table ──────────────────────────────────────────────────────────
    N = len(feasible_applicants)
    print()
    print("=" * 72)
    print(f"  PHASE 4 RESULTS  N={N}  T={T_HORIZON}  ENV={ENVIRONMENT}")
    print("=" * 72)
    print(f"  {'Metric':<34} {'One-Shot':>10} {'Sequential':>12} {'MPC':>8}")
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
        ("Terminal Validity",          "terminal_validity",      "{:.1%}"),
        ("Valid-State Occupancy (O)",   "valid_occupancy",        "{:.1%}"),
        ("Recovery Availability (A)",   "ever_recovered",         "{:.1%}"),
        ("Conditional Traj Survival",   "trajectory_survival",    "{:.1%}"),
        ("Avg Cumulative Cost",         "cumulative_cost",        "{:.4f}"),
        ("Action Execution Rate",       "action_exec_rate",       "{:.1%}"),
        ("-- MPC diagnostics --",       None,                     ""),
        ("  Replan Triggers (N_trig)",  "replan_triggers",        "{:.1f}"),
        ("  Replan Attempts",           "replan_attempts",        "{:.1f}"),
        ("  Replan Successes",          "replan_successes",       "{:.1f}"),
        ("  Replan Failures",           "replan_failures",        "{:.1f}"),
        ("  Solver Feasibility Rate",   "solver_feasibility",     "{:.1%}"),
        ("  Plan Instability (norm)",   "instability_cumulative", "{:.3f}"),
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
    print("Stage-B sanity assertions (zero disturbance):")
    mpc_trigs = col("mpc", "replan_triggers")
    mpc_exec  = col("mpc", "action_exec_rate")
    print(f"  replan_triggers (MPC) = {mpc_trigs:.1f}  [expected: 0]")
    print(f"  action_exec_rate (MPC)= {mpc_exec:.0%}  [expected: >0%]")
    print(f"  action_exec_rate (Seq)= {col('sequential', 'action_exec_rate'):.0%}  [expected: >0%]")
    ok = (mpc_trigs == 0 and mpc_exec > 0)
    print(f"\n  Controller validation: {'PASS' if ok else 'FAIL -- investigate trigger or solver'}")
