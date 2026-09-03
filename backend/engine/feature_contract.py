from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FeatureDefinition:
    name: str
    feature_class: str   # see 10-class taxonomy below
    actionable: bool     # True only for ACTIONABLE_* and CONDITIONALLY_ACTIONABLE
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    allowed_categories: Optional[List[str]] = None
    cost_weight: float = 1.0
    description: str = ''

# 10-class taxonomy
# IMMUTABLE                – birth date, gender
# HISTORICAL_IMMUTABLE     – past default event, CCJ
# ACTIONABLE_STATE         – outstanding bureau debt (borrower can pay it down)
# ACTIONABLE_BEHAVIOUR     – repayment consistency (borrower can improve it)
# CONDITIONALLY_ACTIONABLE – credit utilisation (borrower can reduce it)
# DERIVED                  – DTI ratio (computed, not directly changeable)
# TIME_EVOLVING            – employment tenure (changes with time automatically)
# PLANNING_ONLY            – income target (a goal state, not a direct action)
# LENDER_HIDDEN            – fraud flags (not shown to borrower)
# LENDER_CONTROLLED        – interest rate, product terms

FEATURE_CONTRACT_V2 = {
    # ── Immutable ────────────────────────────────────────────────────────────
    'DAYS_BIRTH': FeatureDefinition(
        'DAYS_BIRTH', 'IMMUTABLE', False,
        description='Age in days (negative). Cannot be changed.'),

    # ── Time-evolving (improves automatically, not actionable directly) ───────
    'DAYS_EMPLOYED': FeatureDefinition(
        'DAYS_EMPLOYED', 'TIME_EVOLVING', False,
        description='Days at current employer. Grows with tenure; not directly actionable.'),

    # ── Immutable (credential, hard to change) ───────────────────────────────
    'NAME_EDUCATION_TYPE': FeatureDefinition(
        'NAME_EDUCATION_TYPE', 'IMMUTABLE', False,
        description='Highest education level. Treated as immutable for recourse purposes.'),

    # ── Lender-controlled (product terms set by lender, not borrower) ─────────
    'AMT_CREDIT': FeatureDefinition(
        'AMT_CREDIT', 'LENDER_CONTROLLED', False,
        min_val=10000, cost_weight=0.5,
        description='Total credit amount. Set by the lender, not a borrower action.'),

    # ── Planning-only (income is a TARGET STATE, not a direct action) ─────────
    'AMT_INCOME_TOTAL': FeatureDefinition(
        'AMT_INCOME_TOTAL', 'PLANNING_ONLY', False,
        min_val=0, cost_weight=2.0,
        description='Total income. A target state; borrower cannot directly execute '
                    'an income increase — it represents a planning goal.'),

    # ── Conditionally actionable (borrower can restructure repayment schedule) ─
    'AMT_ANNUITY': FeatureDefinition(
        'AMT_ANNUITY', 'CONDITIONALLY_ACTIONABLE', True,
        min_val=0, cost_weight=1.0,
        description='Monthly annuity payment. Borrower can negotiate repayment schedule '
                    'within structural bounds (3-10% of credit).'),
}

# Backwards-compatible alias used by legacy code
COMMON_FEATURE_CONTRACT = FEATURE_CONTRACT_V2
