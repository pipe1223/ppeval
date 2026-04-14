from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RubricDimension:
    name: str
    description: str
    min_score: int = 1
    max_score: int = 5


@dataclass
class HumanReviewItem:
    sample_id: str
    task_type: str
    prompt: Optional[str] = None
    prediction: Optional[str] = None
    reference: Optional[str] = None
    retrieved_context: Optional[List[str]] = None
    notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PairwiseReviewItem:
    sample_id: str
    task_type: str
    prompt: Optional[str]
    baseline_output: str
    candidate_output: str
    reference: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
