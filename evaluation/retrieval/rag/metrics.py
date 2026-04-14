from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Set

from evaluation.shared.metrics.ranking import evaluate_ranking_case
from evaluation.shared.metrics.text import answer_correctness_score


def _as_set(values: Iterable[Any]) -> Set[str]:
    return {str(v) for v in values}


def context_precision(retrieved_ids: Sequence[Any], gold_ids: Iterable[Any]) -> float:
    retrieved = [str(x) for x in retrieved_ids]
    gold = _as_set(gold_ids)
    if not retrieved:
        return 0.0
    hits = sum(1 for x in retrieved if x in gold)
    return hits / float(len(retrieved))


def context_recall(retrieved_ids: Sequence[Any], gold_ids: Iterable[Any]) -> float:
    retrieved = _as_set(retrieved_ids)
    gold = _as_set(gold_ids)
    if not gold:
        return 0.0
    return len(retrieved & gold) / float(len(gold))


def citation_precision(cited_ids: Sequence[Any], gold_ids: Iterable[Any]) -> float:
    cited = [str(x) for x in cited_ids]
    gold = _as_set(gold_ids)
    if not cited:
        return 0.0
    hits = sum(1 for x in cited if x in gold)
    return hits / float(len(cited))


def citation_recall(cited_ids: Sequence[Any], gold_ids: Iterable[Any]) -> float:
    cited = _as_set(cited_ids)
    gold = _as_set(gold_ids)
    if not gold:
        return 0.0
    return len(cited & gold) / float(len(gold))


def support_rate(is_supported: Any) -> float:
    return 1.0 if bool(is_supported) else 0.0


def faithfulness_proxy(sample: Dict[str, Any]) -> float:
    """Deterministic proxy, not a semantic LLM faithfulness judge.

    Preferred fields:
    - `claim_support`: list[bool]
    Fallback fields:
    - `is_supported`: bool
    - `cited_ids` and `gold_ids`: overlap-based proxy
    """
    if isinstance(sample.get("claim_support"), list) and sample["claim_support"]:
        claims = [bool(x) for x in sample["claim_support"]]
        return sum(1 for x in claims if x) / float(len(claims))
    if "is_supported" in sample:
        return support_rate(sample.get("is_supported"))
    cited = sample.get("cited_ids", []) or []
    gold = sample.get("gold_ids", []) or []
    if cited and gold:
        return citation_precision(cited, gold)
    return 0.0


def evaluate_rag_sample(sample: Dict[str, Any], ks: Sequence[int] = (1, 3, 5, 10)) -> Dict[str, float]:
    retrieved_ids = sample.get("retrieved_ids", []) or []
    gold_ids = sample.get("gold_ids", []) or []
    scores = evaluate_ranking_case(retrieved_ids, gold_ids, ks=ks)
    scores["ContextPrecision"] = context_precision(retrieved_ids, gold_ids)
    scores["ContextRecall"] = context_recall(retrieved_ids, gold_ids)
    scores["FaithfulnessProxy"] = faithfulness_proxy(sample)
    if "answer" in sample or "gold_answer" in sample:
        scores.update(answer_correctness_score(str(sample.get("answer", "") or ""), str(sample.get("gold_answer", "") or "")))
    if "cited_ids" in sample:
        scores["CitationPrecision"] = citation_precision(sample.get("cited_ids", []) or [], gold_ids)
        scores["CitationRecall"] = citation_recall(sample.get("cited_ids", []) or [], gold_ids)
    if "is_supported" in sample:
        scores["SupportRate"] = support_rate(sample.get("is_supported"))
    return scores
