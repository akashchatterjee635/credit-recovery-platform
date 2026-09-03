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
    Constraint('DTI_MAX_001',
               'Annuity <= 40% of gross income (DTI cap)',
               'structural', 'prototype_assumption', 'LOW', 'hard',
               {'max_dti': 0.40}),
    Constraint('ANNUITY_CREDIT_MIN_001',
               'Annuity >= 3% of credit amount',
               'structural', 'prototype_assumption', 'LOW', 'hard',
               {'min_ratio': 0.03}),
    Constraint('ANNUITY_CREDIT_MAX_001',
               'Annuity <= 10% of credit amount',
               'structural', 'prototype_assumption', 'LOW', 'hard',
               {'max_ratio': 0.10}),
    Constraint('MONTHLY_INCOME_CAP_001',
               'Max verifiable income change per month',
               'plausibility', 'prototype_assumption', 'LOW', 'hard',
               {'max_monthly_change': 5000.0}),
    Constraint('MONTHLY_CREDIT_CAP_001',
               'Max credit change per month',
               'plausibility', 'prototype_assumption', 'LOW', 'hard',
               {'max_monthly_change': 50000.0}),
    Constraint('MONTHLY_ANNUITY_CAP_001',
               'Max annuity change per month',
               'plausibility', 'prototype_assumption', 'LOW', 'hard',
               {'max_monthly_change': 2000.0}),
]


class ConstraintRegistry:
    def __init__(self, constraints=None):
        self._constraints = constraints or _REGISTRY

    def all_constraints(self):
        return self._constraints

    def hard_constraints(self):
        return [c for c in self._constraints if c.hard_or_soft == 'hard']

    def get(self, constraint_id):
        for c in self._constraints:
            if c.constraint_id == constraint_id:
                return c
        return None

    def monthly_cap(self, feature):
        mapping = {
            'AMT_INCOME_TOTAL': 'MONTHLY_INCOME_CAP_001',
            'AMT_CREDIT':       'MONTHLY_CREDIT_CAP_001',
            'AMT_ANNUITY':      'MONTHLY_ANNUITY_CAP_001',
        }
        cid = mapping.get(feature)
        c = self.get(cid) if cid else None
        return c.params['max_monthly_change'] if c else None

    def summary(self):
        rows = [f'{c.constraint_id:<30} [{c.confidence:<6}] [{c.hard_or_soft}] {c.description}'
                for c in self._constraints]
        return 'Constraint Registry\n' + '='*80 + '\n' + '\n'.join(rows)


DEFAULT_REGISTRY = ConstraintRegistry()
