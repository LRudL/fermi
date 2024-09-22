from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Estimate:
    lower: float
    value: float
    upper: float
    unit: str
    name: str | None = None
    reasoning_trace: Any | None = None

@dataclass
class Estimator:
    fn: Callable[[str], Estimate]
    name: str