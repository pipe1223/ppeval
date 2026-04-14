from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from evaluation.shared.metrics.ranking import evaluate_ranking_case
from evaluation.shared.metrics.text import answer_correctness_score


def _as_set(values: Iterable[Any]) -> Set[str]:
    return {str(v) for v in values}


def set_recall(retrieved_ids: Iterable[Any], gold_ids: Iterable[Any]) -> float:
    gold = _as_set(gold_ids)
    if not gold:
        return 0.0
    retrieved = _as_set(retrieved_ids)
    return len(retrieved & gold) / float(len(gold))


def _normalize_path(path: Iterable[Any]) -> Tuple[str, ...]:
    return tuple(str(node) for node in path)


def path_hit_rate(retrieved_paths: Sequence[Sequence[Any]], gold_paths: Sequence[Sequence[Any]]) -> float:
    gold = {_normalize_path(p) for p in gold_paths}
    if not gold:
        return 0.0
    retrieved = {_normalize_path(p) for p in retrieved_paths}
    return 1.0 if retrieved & gold else 0.0


def subgraph_coverage(retrieved_node_ids: Iterable[Any], gold_node_ids: Iterable[Any]) -> float:
    return set_recall(retrieved_node_ids, gold_node_ids)


def multi_hop_answer_success(sample: Dict[str, Any]) -> float:
    gold_paths = sample.get("gold_paths", []) or []
    requires_multihop = bool(sample.get("requires_multihop", False)) or any(len(p) >= 3 for p in gold_paths)
    if not requires_multihop:
        return 0.0
    correctness = answer_correctness_score(str(sample.get("answer", "") or ""), str(sample.get("gold_answer", "") or ""))
    return correctness["AnswerCorrectness"]


def evaluate_graphrag_sample(sample: Dict[str, Any], ks: Sequence[int] = (1, 3, 5, 10)) -> Dict[str, float]:
    retrieved_ids = sample.get("retrieved_ids", []) or []
    gold_ids = sample.get("gold_ids", []) or []
    scores = evaluate_ranking_case(retrieved_ids, gold_ids, ks=ks)
    if "gold_node_ids" in sample:
        scores["NodeRecall"] = set_recall(sample.get("retrieved_node_ids", []) or [], sample.get("gold_node_ids", []) or [])
        scores["SubgraphCoverage"] = subgraph_coverage(sample.get("retrieved_node_ids", []) or [], sample.get("gold_node_ids", []) or [])
    if "gold_edge_ids" in sample:
        scores["EdgeRecall"] = set_recall(sample.get("retrieved_edge_ids", []) or [], sample.get("gold_edge_ids", []) or [])
    if "gold_paths" in sample:
        scores["PathHitRate"] = path_hit_rate(sample.get("retrieved_paths", []) or [], sample.get("gold_paths", []) or [])
    if "answer" in sample or "gold_answer" in sample:
        scores.update(answer_correctness_score(str(sample.get("answer", "") or ""), str(sample.get("gold_answer", "") or "")))
        scores["MultiHopAnswerSuccess"] = multi_hop_answer_success(sample)
    return scores
