from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, Mapping, Sequence


def safe_divide(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2.0 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def classification_metrics(y_true: Sequence[Any], y_pred: Sequence[Any]) -> Dict[str, Any]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    labels = sorted({str(x) for x in y_true} | {str(x) for x in y_pred})
    per_label: Dict[str, Dict[str, float]] = {}
    total_correct = 0
    for yt, yp in zip(y_true, y_pred):
        total_correct += int(str(yt) == str(yp))
    for label in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if str(yt) == label and str(yp) == label)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if str(yt) != label and str(yp) == label)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if str(yt) == label and str(yp) != label)
        per_label[label] = precision_recall_f1(tp, fp, fn)
    macro_precision = sum(v["precision"] for v in per_label.values()) / len(per_label) if per_label else 0.0
    macro_recall = sum(v["recall"] for v in per_label.values()) / len(per_label) if per_label else 0.0
    macro_f1 = sum(v["f1"] for v in per_label.values()) / len(per_label) if per_label else 0.0
    support = Counter(str(x) for x in y_true)
    return {
        "accuracy": safe_divide(total_correct, len(y_true)),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "support": dict(support),
        "per_label": per_label,
    }


def top_k_accuracy(gold_id: Any, ranked_ids: Sequence[Any], k: int) -> float:
    if k <= 0:
        return 0.0
    return 1.0 if str(gold_id) in {str(x) for x in ranked_ids[:k]} else 0.0
