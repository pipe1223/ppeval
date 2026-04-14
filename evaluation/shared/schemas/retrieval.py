from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RagSample:
    sample_id: str
    question: str
    retrieved_ids: List[str] = field(default_factory=list)
    gold_ids: List[str] = field(default_factory=list)
    answer: Optional[str] = None
    gold_answer: Optional[str] = None
    cited_ids: List[str] = field(default_factory=list)
    is_supported: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GraphRagSample(RagSample):
    retrieved_node_ids: List[str] = field(default_factory=list)
    gold_node_ids: List[str] = field(default_factory=list)
    retrieved_edge_ids: List[str] = field(default_factory=list)
    gold_edge_ids: List[str] = field(default_factory=list)
    retrieved_paths: List[List[str]] = field(default_factory=list)
    gold_paths: List[List[str]] = field(default_factory=list)
    requires_multihop: bool = False
