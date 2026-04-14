from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence


def _normalize_winner(value: str) -> str:
    value = (value or "").strip().lower()
    aliases = {
        "candidate": "candidate",
        "new": "candidate",
        "model_b": "candidate",
        "baseline": "baseline",
        "old": "baseline",
        "model_a": "baseline",
        "tie": "tie",
        "draw": "tie",
    }
    return aliases.get(value, value)


def evaluate_pairwise_samples(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        return {"info": {"n_samples": 0}, "results": {}}

    counts = Counter()
    for sample in samples:
        winner = _normalize_winner(str(sample.get("winner", "tie")))
        counts[winner] += 1

    n = float(len(samples))
    results = {
        "CandidateWinRate": counts["candidate"] / n,
        "BaselineWinRate": counts["baseline"] / n,
        "TieRate": counts["tie"] / n,
        "NetWinRate": (counts["candidate"] - counts["baseline"]) / n,
    }
    return {"info": {"n_samples": len(samples), "mode": "pairwise"}, "results": results}
