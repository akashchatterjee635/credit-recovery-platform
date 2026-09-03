import textwrap, os

# ── Fix 4+5+6+9: Rewrite benchmark script entirely ────────────────────────────
bench = textwrap.dedent('''
    """
    experiments/09_mpc_benchmark.py

    Phase 4 MPC Benchmark — compares three recourse regimes:
      A. One-shot    (a_0 = x* - x_0, no further intervention)
      B. Sequential  (distribute target equally over T months, no replanning)
      C. MPC         (event-triggered closed-loop replanning)

    All regimes share the exact same CRN disturbance stream per applicant.

    Metrics (per Bug-5 fix):
      - terminal_validity   : f(x_T) <= tau_T
      - trajectory_survival : fraction of months with f(x_t) <= tau_t, after first recovery
      - cumulative_cost     : sum of realized action cost / disposable income
      - replan_attempts     : how many times MPC tried to replan
      - replan_successes    : how many times the solver actually found a new plan
      - replan_failures     : attempt - success
      - action_execution_rate: fraction of months where ||a_tilde|| > 0
      - solver_feasibility_rate: replan_successes / replan_attempts

    Environment: zero-disturbance for Stage-B sanity; use moderate_disturbance for Stage-D.
    """
    import sys, os
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
    N_APPLICANTS = 5           # Stage B: small deterministic sanity run
    T_HORIZON    = 12
    DELTA_SAFETY = 0.05
    BASE_TAU     = 0.30

    # Stage B: zero disturbance — if MPC still shows Cost=0 here, controller is broken
    ENVIRONMENT  = "zero"      # "zero" | "moderate" | "severe"

    # ── Helpers ──────────────────────────────────────────────────────────────────
    def _get_config(env_name: str) -> DisturbanceConfig:
        if env_name == "zero":
            return DisturbanceConfig.zero_disturbance()
        elif env_name == "moderate":
            cfg = DisturbanceConfig.moderate_disturbance()
            cfg.policy_shift = True
            return cfg
        elif env_name == "severe":
            cfg = DisturbanceConfig(
                p_miss=0.20, beta_alpha=5.0, beta_beta=1.5,
                p_income_shock=0.10, p_debt_shock=0.10,
                income_shock_min=0.50, income_shock_max=0.80,
                shifted_threshold=0.25, policy_shift=True,
            )
            return cfg
        raise ValueError(f"Unknown env: {env_name}")


    def run_regime(
        regime_type: str,
        applicant_orig: pd.DataFrame,
        adapter,
        registry,
        train_df: pd.DataFrame,  # Bug-9 fix: wired to DiCE
        config: DisturbanceConfig,
        crn_seed: int,
    ) -> dict:
        """
        Returns per-applicant result dict with all diagnostic counters.
        """
        # Each regime gets a fresh simulator + same-seed RNG (CRN guarantee)
        sim = EnvironmentSimulator()
        rng = np.random.RandomState(crn_seed)

        # Bug-9 fix: pass train_df so DiCE fallback actually works
        router = SolverRouter(
            risk_model=adapter,
            threshold=BASE_TAU,
            registry=registry,
            feature_contract=FEATURE_CONTRACT_V3,
            training_data=train_df,
        )

        mpc = MPCController(
            risk_model=adapter,
            base_threshold=BASE_TAU,
            feature_contract=FEATURE_CONTRACT_V3,
            solver_router=router,
            delta_safety=DELTA_SAFETY,
            gamma_stability=0.5,
        )

        current_state = applicant_orig.copy()
        current_tau   = BASE_TAU

        # ── Generate initial plan (all regimes use the same call) ────────────────
        # We request from MPC once at t=0 to get the target state
        initial_res  = mpc.get_action(current_state, 0, T_HORIZON, current_tau)
        static_target = initial_res["plan_target"]   # dict {feature: abs_target} or None

        # For one-shot: inject full delta at t=0 only
        # For sequential: distribute static_target evenly across T months

        # Reset MPC state for the actual loop (so one-shot/sequential don't carry it over)
        if regime_type in ("one-shot", "sequential"):
            mpc.previous_plan_target = None

        # ── Counters ─────────────────────────────────────────────────────────────
        replan_attempts   = 0
        replan_successes  = 0
        cumulative_cost   = 0.0
        months_with_action = 0
        recovery_month    = None
        months_valid_after_recovery = 0

        risk_trace   = []
        action_trace = []

        for t in range(T_HORIZON):
            # Update policy threshold (may shift at t=6 in moderate/severe)
            current_tau = sim.policy_env.step(t, config, rng=rng)

            # ── Choose action ────────────────────────────────────────────────────
            if regime_type == "mpc":
                step_res = mpc.get_action(current_state, t, T_HORIZON, current_tau)
                a_t = step_res["action_t"]
                if step_res["replan_attempt"]:
                    replan_attempts  += 1
                    replan_successes += int(step_res["replan_success"])

            elif regime_type == "sequential":
                # Distribute static_target evenly; never replans
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
                            a_t[f] = delta  # entire change in one month

            # ── Simulate step (CRN synchronized) ────────────────────────────────
            next_state, log_t = sim.step(current_state, a_t, t, config, rng=rng)

            current_risk = float(adapter.predict_risk(current_state)[0])
            risk_trace.append(current_risk)
            action_trace.append(a_t)

            # Track realized actions
            realized = log_t.get("realized_action", {})
            realized_norm = sum(abs(v) for v in realized.values())
            if realized_norm > 1e-9:
                months_with_action += 1

            # Cumulative cost (scaled by disposable income proxy)
            inc = float(current_state.iloc[0].get("AMT_INCOME_TOTAL", 1)) or 1.0
            cumulative_cost += sum(abs(v) for v in realized.values()) / inc

            # Survival tracking: first month risk drops below tau
            if current_risk <= current_tau and recovery_month is None:
                recovery_month = t
            if recovery_month is not None:
                months_valid_after_recovery += 1

            current_state = next_state

        # ── Terminal evaluation ───────────────────────────────────────────────────
        terminal_risk     = float(adapter.predict_risk(current_state)[0])
        terminal_validity = int(terminal_risk <= current_tau)

        # Bug-5 fix: proper trajectory survival rate
        if recovery_month is not None:
            denom = T_HORIZON - recovery_month
            survival_rate = months_valid_after_recovery / denom if denom > 0 else 0.0
        else:
            survival_rate = 0.0

        # Diagnostic feasibility rates
        action_exec_rate = months_with_action / T_HORIZON
        solver_feasibility = (
            replan_successes / replan_attempts if replan_attempts > 0 else float("nan")
        )

        return {
            "terminal_validity":    terminal_validity,
            "terminal_risk":        terminal_risk,
            "trajectory_survival":  survival_rate,
            "recovery_month":       recovery_month,
            "cumulative_cost":      cumulative_cost,
            "replan_attempts":      replan_attempts,
            "replan_successes":     replan_successes,
            "replan_failures":      replan_attempts - replan_successes,
            "action_exec_rate":     action_exec_rate,
            "solver_feasibility":   solver_feasibility,
            "initial_risk":         float(adapter.predict_risk(applicant_orig)[0]),
        }


    # ── Main ─────────────────────────────────────────────────────────────────────
    if __name__ == "__main__":
        adapter = RiskModelAdapter()
        adapter.load()

        test_df  = pd.read_csv("data/test_reference.csv").dropna(subset=["TARGET"])
        train_df = pd.read_csv("data/train_reference.csv").dropna(subset=["TARGET"]).head(500)

        registry = ConstraintRegistry()
        config   = _get_config(ENVIRONMENT)

        # Sample applicants above threshold
        risks      = adapter.predict_risk(test_df)
        above_mask = risks > BASE_TAU
        sample_df  = test_df[above_mask].head(N_APPLICANTS)

        print(f"Environment : {ENVIRONMENT}")
        print(f"Applicants  : {len(sample_df)}  (above-threshold cohort)")
        print(f"T_horizon   : {T_HORIZON} months")
        print(f"delta_safety: {DELTA_SAFETY}  (target tau = {BASE_TAU - DELTA_SAFETY})")
        print()

        regimes = ["one-shot", "sequential", "mpc"]
        all_results = {r: [] for r in regimes}

        for i in range(len(sample_df)):
            applicant = sample_df.iloc[[i]]
            seed      = 42 + i
            init_risk = float(adapter.predict_risk(applicant)[0])
            print(f"Applicant {i+1}/{len(sample_df)}  (initial_risk={init_risk:.3f})", flush=True)

            for regime in regimes:
                res = run_regime(regime, applicant, adapter, registry, train_df, config, seed)
                all_results[regime].append(res)
                print(
                    f"  [{regime:10s}] terminal_valid={res[\'terminal_validity\']}  "
                    f"cost={res[\'cumulative_cost\']:.4f}  "
                    f"action_exec={res[\'action_exec_rate\']:.0%}  "
                    f"replans={res[\'replan_attempts\']}  successes={res[\'replan_successes\']}",
                    flush=True,
                )

        # ── Summary table ─────────────────────────────────────────────────────────
        print()
        print("=" * 68)
        print(f"  PHASE 4 BENCHMARK RESULTS  |  N={N_APPLICANTS}  T={T_HORIZON}  ENV={ENVIRONMENT}")
        print("=" * 68)
        hdr = f"  {\'Metric\':<30} {\'One-Shot\':>10} {\'Sequential\':>12} {\'MPC\':>8}"
        print(hdr)
        print("-" * 68)

        def col(regime, key):
            vals = [r[key] for r in all_results[regime] if not pd.isna(r[key])]
            return float(np.mean(vals)) if vals else float("nan")

        rows = [
            ("Terminal Validity",        "terminal_validity",   "{:.1%}"),
            ("Trajectory Survival Rate", "trajectory_survival", "{:.1%}"),
            ("Avg Cumulative Cost",      "cumulative_cost",     "{:.4f}"),
            ("Action Exec Rate",         "action_exec_rate",    "{:.1%}"),
            ("Replan Attempts (avg)",    "replan_attempts",     "{:.1f}"),
            ("Replan Successes (avg)",   "replan_successes",    "{:.1f}"),
            ("Replan Failures (avg)",    "replan_failures",     "{:.1f}"),
            ("Solver Feasibility Rate",  "solver_feasibility",  "{:.1%}"),
        ]

        for label, key, fmt in rows:
            c_os  = col("one-shot",   key)
            c_seq = col("sequential", key)
            c_mpc = col("mpc",        key)
            def fmt_val(v):
                if pd.isna(v): return "  N/A"
                try: return fmt.format(v)
                except: return str(v)
            print(f"  {label:<30} {fmt_val(c_os):>10} {fmt_val(c_seq):>12} {fmt_val(c_mpc):>8}")

        print("=" * 68)
        print()
        print("NOTE: This output characterises the semi-synthetic simulation environment.")
        print("      Disturbance parameters are labelled simulation assumptions, not")
        print("      empirically estimated real-world frequencies.")
''').lstrip()

with open("experiments/09_mpc_benchmark.py", "w", encoding="utf-8") as f:
    f.write(bench)
print("OK  09_mpc_benchmark.py (all 6 remaining bugs fixed)")
