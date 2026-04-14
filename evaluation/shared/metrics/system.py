from __future__ import annotations

from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    idx = (len(xs) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(xs) - 1)
    frac = idx - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def summarize_system_metrics(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    if not rows:
        return {"n_records": 0, "results": {}}

    latencies = [float(r.get("latency_ms", 0.0) or 0.0) for r in rows if r.get("latency_ms") is not None]
    costs = [float(r.get("cost_usd", 0.0) or 0.0) for r in rows if r.get("cost_usd") is not None]
    input_tokens = [float(r.get("input_tokens", 0.0) or 0.0) for r in rows if r.get("input_tokens") is not None]
    output_tokens = [float(r.get("output_tokens", 0.0) or 0.0) for r in rows if r.get("output_tokens") is not None]
    successes = [bool(r.get("success", True)) for r in rows]
    cache_hits = [bool(r.get("cache_hit", False)) for r in rows if "cache_hit" in r]

    results = {
        "LatencyP50Ms": _percentile(latencies, 0.50),
        "LatencyP95Ms": _percentile(latencies, 0.95),
        "LatencyP99Ms": _percentile(latencies, 0.99),
        "LatencyMeanMs": mean(latencies) if latencies else 0.0,
        "AverageCostUsd": mean(costs) if costs else 0.0,
        "TotalCostUsd": sum(costs),
        "AverageInputTokens": mean(input_tokens) if input_tokens else 0.0,
        "AverageOutputTokens": mean(output_tokens) if output_tokens else 0.0,
        "SuccessRate": sum(1 for x in successes if x) / float(len(successes)),
    }
    if cache_hits:
        results["CacheHitRate"] = sum(1 for x in cache_hits if x) / float(len(cache_hits))
    return {"n_records": len(rows), "results": results}
