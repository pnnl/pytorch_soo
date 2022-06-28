"""
A Data Class for specifying Line Search behaviors
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LineSearchSpec:
    max_searches: int = 10
    extrapolation_factor: Optional[float] = 0.5
    sufficient_decrease: float = 0.9
    curvature_constant: Optional[float] = None
