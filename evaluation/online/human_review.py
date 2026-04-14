from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence

from evaluation.shared.schemas.review import HumanReviewItem, PairwiseReviewItem
from evaluation.shared.utils.io import load_jsonl, write_jsonl


def build_human_review_queue(samples: Sequence[Dict[str, Any]], path: str, task_type: str = "generic") -> None:
    rows: List[Dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        item = HumanReviewItem(
            sample_id=str(sample.get("sample_id", idx)),
            task_type=task_type,
            prompt=sample.get("question") or sample.get("prompt"),
            prediction=sample.get("answer") or sample.get("prediction"),
            reference=sample.get("gold_answer") or sample.get("reference"),
            retrieved_context=sample.get("retrieved_context"),
            metadata={k: v for k, v in sample.items() if k not in {"question", "prompt", "answer", "prediction", "gold_answer", "reference", "retrieved_context"}},
        )
        rows.append(item.to_dict())
    write_jsonl(path, rows)


def build_pairwise_review_queue(samples: Sequence[Dict[str, Any]], path: str, task_type: str = "generic") -> None:
    rows: List[Dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        item = PairwiseReviewItem(
            sample_id=str(sample.get("sample_id", idx)),
            task_type=task_type,
            prompt=sample.get("question") or sample.get("prompt"),
            baseline_output=str(sample.get("baseline_output", "")),
            candidate_output=str(sample.get("candidate_output", "")),
            reference=sample.get("reference") or sample.get("gold_answer"),
            metadata={k: v for k, v in sample.items() if k not in {"question", "prompt", "baseline_output", "candidate_output", "reference", "gold_answer"}},
        )
        rows.append(item.to_dict())
    write_jsonl(path, rows)


def aggregate_human_review_labels(path: str) -> Dict[str, Any]:
    rows = load_jsonl(path)
    labels = Counter(str(r.get("label", "unlabeled")) for r in rows)
    scored = [float(r.get("score")) for r in rows if r.get("score") is not None]
    mean_score = sum(scored) / float(len(scored)) if scored else 0.0
    return {
        "n_reviews": len(rows),
        "label_distribution": dict(labels),
        "mean_score": mean_score,
    }
