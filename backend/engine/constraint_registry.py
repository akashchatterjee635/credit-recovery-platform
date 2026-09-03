from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Constraint:
    constraint_id: str
    description: str
    constraint_type: str
    source: str
    confidence: str
    hard_or_soft: str
    params: Dict[str, Any] = field(default_factory=dict)


_REGISTRY: List[Constraint] = [
    Constraint('DTI_MAX_001', 'Annuity <= 40pct income', 'structural', 'prototype', 'LOW', 'hard', {'max_dti': 0.40}),
    Constraint('ANNUITY_CREDIT_MIN_001', 'Annuity >= 3pct credit', 'structural', 'prototype', 'LOW', 'hard', {'min_ratio': 0.03}),
    Constraint('ANNUITY_CREDIT_MAX_001', 'Annuity <= 10pct credit', 'structural', 'prototype', 'LOW', 'hard', {'max_ratio': 0.10}),
    Constraint('MONTHLY_INCOME_CAP_001', 'Max income change/mo', 'plausibility', 'prototype', 'LOW', 'hard', {'max_monthly_change': 5000.0}),
    Constraint('MONTHLY_CREDIT_CAP_001', 'Max credit change/mo', 'plausibility', 'prototype', 'LOW', 'hard', {'max_monthly_change': 50000.0}),
    Constraint('MONTHLY_ANNUITY_CAP_001', 'Max annuity change/mo', 'plausibility', 'prototype', 'LOW', 'hard', {'max_monthly_change': 2000.0}),
    Constraint('MONTHLY_DEBT_PAYDOWN_CAP_001', 'Max debt paydown/mo', 'plausibility', 'prototype', 'LOW', 'hard', {'max_monthly_change': 25000.0}),
    Constraint('MONTHLY_OVERDUE_RESOLUTION_CAP_001', 'Max overdue resolved/mo', 'plausibility', 'prototype', 'LOW', 'hard', {'max_monthly_change': 50000.0}),
    Constraint('MONTHLY_ACTIVE_CREDIT_CAP_001', 'Max credit lines closed/mo', 'plausibility', 'prototype', 'LOW', 'hard', {'max_monthly_change': 2.0}),
    Constraint('RECOURSE_THRESHOLD_001', 'Risk threshold for recourse', 'policy', 'validation_policy', 'MEDIUM', 'hard', {'threshold': 0.30}),
]


class ConstraintRegistry:
    def __init__(self, constraints=None):
        self._constraints = constraints or _REGISTRY

    def all_constraints(self):
        return self._constraints

    def hard_constraints(self):
        return [c for c in self._constraints if c.hard_or_soft == 'hard']

    def structural_constraints(self):
        return [c for c in self._constraints if c.constraint_type == 'structural']

    def get(self, constraint_id):
        for c in self._constraints:
            if c.constraint_id == constraint_id:
                return c
        return None

    def recourse_threshold(self):
        c = self.get('RECOURSE_THRESHOLD_001')
        return c.params['threshold'] if c else 0.30

    def monthly_cap(self, feature):
        mapping = {
            'AMT_INCOME_TOTAL': 'MONTHLY_INCOME_CAP_001',
            'AMT_CREDIT': 'MONTHLY_CREDIT_CAP_001',
            'AMT_ANNUITY': 'MONTHLY_ANNUITY_CAP_001',
            'BUREAU_TOTAL_DEBT': 'MONTHLY_DEBT_PAYDOWN_CAP_001',
            'BUREAU_MAX_OVERDUE': 'MONTHLY_OVERDUE_RESOLUTION_CAP_001',
            'BUREAU_ACTIVE_COUNT': 'MONTHLY_ACTIVE_CREDIT_CAP_001',
        }
        cid = mapping.get(feature)
        c = self.get(cid) if cid else None
        return c.params['max_monthly_change'] if c else None

    def summary(self):
        rows = [f'{c.constraint_id:<40} [{c.confidence}] [{c.hard_or_soft}] {c.description}' for c in self._constraints]
        return '\n'.join(rows)


DEFAULT_REGISTRY = ConstraintRegistry()
