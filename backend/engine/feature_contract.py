from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FeatureDefinition:
    name: str
    feature_class: str
    actionable: bool
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    allowed_categories: Optional[List[str]] = None
    cost_weight: float = 1.0
    domain: str = 'continuous'
    description: str = ''
    corresponding_action: str = ''


FEATURE_CONTRACT_V3 = {
    'DAYS_BIRTH': FeatureDefinition(
        'DAYS_BIRTH', 'IMMUTABLE', False, domain='integer',
        description='Age in days (negative). Cannot be changed.'),
    'NAME_EDUCATION_TYPE': FeatureDefinition(
        'NAME_EDUCATION_TYPE', 'IMMUTABLE', False,
        description='Highest education level.'),
    'DAYS_EMPLOYED': FeatureDefinition(
        'DAYS_EMPLOYED', 'TIME_EVOLVING', False, domain='integer',
        description='Employment tenure. Changes naturally with time.'),
    'INST_LATE_RATIO': FeatureDefinition(
        'INST_LATE_RATIO', 'TIME_EVOLVING', False,
        description='Ratio of late installment payments.',
        corresponding_action='Future on-time repayment'),
    'INST_AVG_DAYS_LATE': FeatureDefinition(
        'INST_AVG_DAYS_LATE', 'TIME_EVOLVING', False,
        description='Average days late on installments.',
        corresponding_action='Future payment punctuality'),
    'AMT_CREDIT': FeatureDefinition(
        'AMT_CREDIT', 'LENDER_CONTROLLED', False, min_val=10000,
        cost_weight=0.5, description='Total credit amount. Set by lender.'),
    'AMT_INCOME_TOTAL': FeatureDefinition(
        'AMT_INCOME_TOTAL', 'PLANNING_ONLY', False, min_val=0,
        cost_weight=2.0, description='Total income. A planning target.'),
    'AMT_ANNUITY': FeatureDefinition(
        'AMT_ANNUITY', 'CONDITIONALLY_ACTIONABLE', True, min_val=0,
        cost_weight=1.0, description='Monthly annuity payment.',
        corresponding_action='Negotiate repayment schedule'),
    'BUREAU_ACTIVE_COUNT': FeatureDefinition(
        'BUREAU_ACTIVE_COUNT', 'CONDITIONALLY_ACTIONABLE', True, min_val=0, domain='integer',
        cost_weight=0.8, description='Number of active bureau credit lines.',
        corresponding_action='Close unnecessary credit lines'),
    'BUREAU_TOTAL_DEBT': FeatureDefinition(
        'BUREAU_TOTAL_DEBT', 'ACTIONABLE_STATE', True, min_val=0,
        cost_weight=1.5, description='Total outstanding bureau debt.',
        corresponding_action='Debt pay-down'),
    'BUREAU_MAX_OVERDUE': FeatureDefinition(
        'BUREAU_MAX_OVERDUE', 'ACTIONABLE_STATE', True, min_val=0,
        cost_weight=1.2, description='Maximum overdue amount in bureau.',
        corresponding_action='Resolve overdue obligation'),
    'PREV_REFUSED_RATIO': FeatureDefinition(
        'PREV_REFUSED_RATIO', 'HISTORICAL_IMMUTABLE', False,
        description='Ratio of refused previous applications.'),
    'DERIVED_DTI': FeatureDefinition(
        'DERIVED_DTI', 'DERIVED', False,
        description='Debt-to-income ratio. Computed.'),
    'DERIVED_ANNUITY_CREDIT_RATIO': FeatureDefinition(
        'DERIVED_ANNUITY_CREDIT_RATIO', 'DERIVED', False,
        description='Annuity/Credit ratio. Computed.'),
    'DERIVED_CREDIT_INCOME_RATIO': FeatureDefinition(
        'DERIVED_CREDIT_INCOME_RATIO', 'DERIVED', False,
        description='Credit/Income ratio. Computed.'),
    'BUREAU_AVG_DAYS_CREDIT': FeatureDefinition(
        'BUREAU_AVG_DAYS_CREDIT', 'TIME_EVOLVING', False,
        description='Average age of bureau credit lines.'),
    'BUREAU_AVG_DAYS_OVERDUE': FeatureDefinition(
        'BUREAU_AVG_DAYS_OVERDUE', 'TIME_EVOLVING', False,
        description='Average days overdue across bureau records.'),
    'PREV_APP_COUNT': FeatureDefinition(
        'PREV_APP_COUNT', 'HISTORICAL_IMMUTABLE', False, domain='integer',
        description='Number of previous applications.'),
    'PREV_AMT_CREDIT_MEAN': FeatureDefinition(
        'PREV_AMT_CREDIT_MEAN', 'HISTORICAL_IMMUTABLE', False,
        description='Mean credit of previous applications.'),
    'INST_AVG_PAYMENT_DIFF': FeatureDefinition(
        'INST_AVG_PAYMENT_DIFF', 'TIME_EVOLVING', False,
        description='Avg diff between payment and instalment amount.'),
}

FEATURE_CONTRACT_V2 = FEATURE_CONTRACT_V3
COMMON_FEATURE_CONTRACT = FEATURE_CONTRACT_V3
