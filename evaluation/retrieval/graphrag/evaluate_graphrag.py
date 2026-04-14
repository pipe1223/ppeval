from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

if __package__ in {None, ""}:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from ranking.metrics import evaluate_ranking_case


PathType = Sequence[str]


def _normalize_path(path: Iterable[str]) -> Tuple[str, ...]:
    return tuple(str(node) for node in path)


def set_recall(retrieved_ids: Iterable[str], gold_ids: Iterable[str]) -> float:
    gold = set(str(x) for x in gold_ids)
    if not gold:
        return 0.0
    retrieved = set(str(x) for x in retrieved_ids)
    hits = len(retrieved & gold)
    return hits / float(len(gold))


def path_hit_rate(retrieved_paths: Sequence[PathType], gold_paths: Sequence[PathType]) -> float:
    if not gold_paths:
        return 0.0
    retrieved = {_normalize_path(path) for path in retrieved_paths}
    gold = {_normalize_path(path) for path in gold_paths}
    return 1.0 if retrieved & gold else 0.0


def evaluate_graphrag_samples(samples: Sequence[Dict[str, Any]], ks: Sequence[int] = (1, 3, 5, 10)) -> Dict[str, Any]:
    if not samples:
        return {"info": {"n_samples": 0}, "results": {}}

    totals: Dict[str, float] = defaultdict(float)
    n_node_labels = 0
    n_edge_labels = 0
    n_path_labels = 0

    for sample in samples:
        retrieved_ids = sample.get("retrieved_ids", []) or []
        gold_ids = sample.get("gold_ids", []) or []
        ranking_scores = evaluate_ranking_case(retrieved_ids, gold_ids, ks=ks)
        for key, value in ranking_scores.items():
            totals[key] += float(value)

        if "gold_node_ids" in sample:
            n_node_labels += 1
            totals["NodeRecall"] += set_recall(sample.get("retrieved_node_ids", []), sample.get("gold_node_ids", []))

        if "gold_edge_ids" in sample:
            n_edge_labels += 1
            totals["EdgeRecall"] += set_recall(sample.get("retrieved_edge_ids", []), sample.get("gold_edge_ids", []))

        if "gold_paths" in sample:
            n_path_labels += 1
            totals["PathHitRate"] += path_hit_rate(sample.get("retrieved_paths", []), sample.get("gold_paths", []))

    n = float(len(samples))
    results: Dict[str, float] = {}
    for key, value in totals.items():
        if key == "NodeRecall" and n_node_labels > 0:
            results[key] = value / float(n_node_labels)
        elif key == "EdgeRecall" and n_edge_labels > 0:
            results[key] = value / float(n_edge_labels)
        elif key == "PathHitRate" and n_path_labels > 0:
            results[key] = value / float(n_path_labels)
        elif key not in {"NodeRecall", "EdgeRecall", "PathHitRate"}:
            results[key] = value / n

    return {
        "info": {
            "n_samples": len(samples),
            "ks": list(ks),
            "mode": "graphrag",
        },
        "results": results,
    }


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a simple GraphRAG results JSON file.")
    parser.add_argument("json_path", help="Path to a JSON file containing a list of GraphRAG samples.")
    args = parser.parse_args()

    data = load_json(args.json_path)
    if not isinstance(data, list):
        raise TypeError("Expected the top-level JSON value to be a list of samples.")

    report = evaluate_graphrag_samples(data)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
