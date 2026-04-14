from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from evaluation.shared.metrics.ranking import evaluate_ranking_case
from evaluation.shared.metrics.text import exact_match, token_f1


def _extract_samples(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        data = obj.get("data")
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
    raise TypeError("Expected a top-level list of samples or a dict with a 'data' list.")


def _load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _set_recall(retrieved_ids: Iterable[str], gold_ids: Iterable[str]) -> float:
    gold = set(str(x) for x in gold_ids)
    if not gold:
        return 0.0
    retrieved = set(str(x) for x in retrieved_ids)
    return len(retrieved & gold) / float(len(gold))


def _normalize_path(path: Iterable[str]) -> Tuple[str, ...]:
    return tuple(str(x) for x in path)


def _path_hit_rate(retrieved_paths: Sequence[Sequence[str]], gold_paths: Sequence[Sequence[str]]) -> float:
    if not gold_paths:
        return 0.0
    retrieved = {_normalize_path(p) for p in retrieved_paths}
    gold = {_normalize_path(p) for p in gold_paths}
    return 1.0 if retrieved & gold else 0.0


def evaluate_graphrag_samples(samples: Sequence[Dict[str, Any]], ks: Sequence[int] = (1, 3, 5, 10)) -> Dict[str, Any]:
    if not samples:
        return {"info": {"mode": "graphrag", "n_samples": 0, "ks": list(ks)}, "results": {}}

    totals: Dict[str, float] = defaultdict(float)
    n_answer_pairs = 0
    n_node_labels = 0
    n_edge_labels = 0
    n_path_labels = 0

    for sample in samples:
        retrieved_ids = sample.get("retrieved_ids", []) or []
        gold_ids = sample.get("gold_ids", []) or []
        if gold_ids or retrieved_ids:
            for key, value in evaluate_ranking_case(retrieved_ids, gold_ids, ks=ks).items():
                totals[key] += float(value)

        if "gold_node_ids" in sample:
            n_node_labels += 1
            totals["NodeRecall"] += _set_recall(sample.get("retrieved_node_ids", []), sample.get("gold_node_ids", []))
        if "gold_edge_ids" in sample:
            n_edge_labels += 1
            totals["EdgeRecall"] += _set_recall(sample.get("retrieved_edge_ids", []), sample.get("gold_edge_ids", []))
        if "gold_paths" in sample:
            n_path_labels += 1
            totals["PathHitRate"] += _path_hit_rate(sample.get("retrieved_paths", []), sample.get("gold_paths", []))

        if ("answer" in sample) or ("gold_answer" in sample):
            n_answer_pairs += 1
            answer = str(sample.get("answer", "") or "")
            gold_answer = str(sample.get("gold_answer", "") or "")
            totals["ExactMatch"] += exact_match(answer, gold_answer)
            totals["TokenF1"] += token_f1(answer, gold_answer)

    n = float(len(samples))
    results: Dict[str, float] = {}
    for key, value in totals.items():
        if key in {"ExactMatch", "TokenF1"}:
            denom = float(n_answer_pairs) if n_answer_pairs > 0 else n
            results[key] = value / denom if denom > 0 else 0.0
        elif key == "NodeRecall":
            results[key] = value / float(n_node_labels) if n_node_labels > 0 else 0.0
        elif key == "EdgeRecall":
            results[key] = value / float(n_edge_labels) if n_edge_labels > 0 else 0.0
        elif key == "PathHitRate":
            results[key] = value / float(n_path_labels) if n_path_labels > 0 else 0.0
        else:
            results[key] = value / n

    return {"info": {"mode": "graphrag", "n_samples": len(samples), "ks": list(ks)}, "results": results}


def evaluate_graphrag_file(path: str | Path, ks: Sequence[int] = (1, 3, 5, 10)) -> Dict[str, Any]:
    return evaluate_graphrag_samples(_extract_samples(_load_json(path)), ks=ks)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a GraphRAG JSON file.")
    parser.add_argument("json_path", help="Path to a JSON file containing GraphRAG samples.")
    args = parser.parse_args(argv)
    print(json.dumps(evaluate_graphrag_file(args.json_path), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
