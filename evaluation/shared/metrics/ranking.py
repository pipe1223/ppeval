from __future__ import annotations

from math import log2
from typing import Dict, Iterable, Sequence, Set, List

import numpy as np


def _normalize_ids(values: Iterable[str]) -> List[str]:
    return [str(v) for v in values]


def _gold_set(gold_ids: Iterable[str]) -> Set[str]:
    return set(_normalize_ids(gold_ids))


def precision_at_k(retrieved_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = _normalize_ids(retrieved_ids[:k])
    if not top_k:
        return 0.0
    gold = _gold_set(gold_ids)
    hits = sum(1 for item in top_k if item in gold)
    return hits / float(len(top_k))


def recall_at_k(retrieved_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float:
    gold = _gold_set(gold_ids)
    if not gold or k <= 0:
        return 0.0
    top_k = _normalize_ids(retrieved_ids[:k])
    hits = sum(1 for item in top_k if item in gold)
    return hits / float(len(gold))


def hit_rate_at_k(retrieved_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float:
    gold = _gold_set(gold_ids)
    if not gold or k <= 0:
        return 0.0
    top_k = _normalize_ids(retrieved_ids[:k])
    return 1.0 if any(item in gold for item in top_k) else 0.0


def reciprocal_rank(retrieved_ids: Sequence[str], gold_ids: Iterable[str]) -> float:
    gold = _gold_set(gold_ids)
    if not gold:
        return 0.0
    for idx, item in enumerate(_normalize_ids(retrieved_ids), start=1):
        if item in gold:
            return 1.0 / float(idx)
    return 0.0


def average_precision(retrieved_ids: Sequence[str], gold_ids: Iterable[str], k: int | None = None) -> float:
    gold = _gold_set(gold_ids)
    if not gold:
        return 0.0
    ranked = _normalize_ids(retrieved_ids if k is None else retrieved_ids[:k])
    if not ranked:
        return 0.0
    score = 0.0
    hit_count = 0
    for idx, item in enumerate(ranked, start=1):
        if item in gold:
            hit_count += 1
            score += hit_count / float(idx)
    return score / float(len(gold))


def dcg_at_k(retrieved_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float:
    gold = _gold_set(gold_ids)
    if not gold or k <= 0:
        return 0.0
    score = 0.0
    for idx, item in enumerate(_normalize_ids(retrieved_ids[:k]), start=1):
        relevance = 1.0 if item in gold else 0.0
        if relevance > 0:
            score += relevance / log2(idx + 1)
    return score


def ndcg_at_k(retrieved_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float:
    gold = _gold_set(gold_ids)
    if not gold or k <= 0:
        return 0.0
    dcg = dcg_at_k(retrieved_ids, gold, k)
    ideal_hits = min(len(gold), k)
    ideal = sum(1.0 / log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / ideal if ideal > 0 else 0.0


def evaluate_ranking_case(retrieved_ids: Sequence[str], gold_ids: Iterable[str], ks: Sequence[int] = (1, 3, 5, 10)) -> Dict[str, float]:
    results: Dict[str, float] = {
        "MRR": reciprocal_rank(retrieved_ids, gold_ids),
        "AP": average_precision(retrieved_ids, gold_ids),
    }
    for k in ks:
        results[f"Precision@{k}"] = precision_at_k(retrieved_ids, gold_ids, k)
        results[f"Recall@{k}"] = recall_at_k(retrieved_ids, gold_ids, k)
        results[f"HitRate@{k}"] = hit_rate_at_k(retrieved_ids, gold_ids, k)
        results[f"nDCG@{k}"] = ndcg_at_k(retrieved_ids, gold_ids, k)
    return results


def recall_at_k_from_similarity(similarity_matrix: np.ndarray, k: int) -> float:
    recalls = []
    num_queries = similarity_matrix.shape[0]
    for i in range(num_queries):
        sorted_indices = np.argsort(-similarity_matrix[i])
        top_k = sorted_indices[:k]
        recalls.append(1.0 if i in top_k else 0.0)
    return float(np.mean(recalls)) if recalls else 0.0


def average_precision_from_similarity(similarity_matrix: np.ndarray, query_idx: int) -> float:
    sorted_indices = np.argsort(-similarity_matrix[query_idx])
    ap = 0.0
    hits = 0
    for i, idx in enumerate(sorted_indices):
        if idx == query_idx:
            hits += 1
            ap += hits / float(i + 1)
    return ap


def mean_average_precision_from_similarity(similarity_matrix: np.ndarray) -> float:
    aps = [average_precision_from_similarity(similarity_matrix, i) for i in range(similarity_matrix.shape[0])]
    return float(np.mean(aps)) if aps else 0.0
