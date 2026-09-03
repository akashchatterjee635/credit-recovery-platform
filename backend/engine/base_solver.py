from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import pandas as pd


@dataclass
class RecourseResult:
    status: str              # success | failed | eligible | infeasible_within_horizon
    solver: str
    message: str
    original_risk: Optional[float] = None
    new_risk: Optional[float] = None
    cost: Optional[float] = None
    original_state: Optional[Dict[str, Any]] = None
    new_state: Optional[Dict[str, Any]] = None
    violations: List[str] = field(default_factory=list)
    gate_results: Dict[str, bool] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class BaseSolver(ABC):
    '''Abstract interface every solver must implement.'''
    solver_name: str = 'BaseSolver'

    @abstractmethod
    def generate_recourse(self, applicant: pd.DataFrame) -> RecourseResult:
        ...
