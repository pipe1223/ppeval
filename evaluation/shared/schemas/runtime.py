from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class SystemRecord:
    request_id: str
    latency_ms: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    success: bool = True
    cache_hit: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
