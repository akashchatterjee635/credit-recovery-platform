from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class FeatureDefinition:
    name: str
    feature_class: str # 'immutable', 'recourse', 'planning_only', 'lender_hidden'
    actionable: bool
    min_val: float = None
    max_val: float = None
    allowed_categories: List[str] = None
    cost_weight: float = 1.0

# Define the common feature contract
COMMON_FEATURE_CONTRACT = {
    'DAYS_BIRTH': FeatureDefinition('DAYS_BIRTH', 'immutable', False),
    'DAYS_EMPLOYED': FeatureDefinition('DAYS_EMPLOYED', 'immutable', False),
    'NAME_EDUCATION_TYPE': FeatureDefinition('NAME_EDUCATION_TYPE', 'immutable', False),
    
    # Recourse variables (these can be modified by the solver)
    'AMT_CREDIT': FeatureDefinition('AMT_CREDIT', 'recourse', True, min_val=0, cost_weight=0.5),
    'AMT_INCOME_TOTAL': FeatureDefinition('AMT_INCOME_TOTAL', 'recourse', True, min_val=0, cost_weight=2.0),
    'AMT_ANNUITY': FeatureDefinition('AMT_ANNUITY', 'recourse', True, min_val=0, cost_weight=1.0),
}
