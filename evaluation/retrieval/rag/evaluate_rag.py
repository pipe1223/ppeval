from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

if __package__ in {None, ""}:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from ranking.metrics import evaluate_ranking_case


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if (prediction or "").strip() == (reference or "").strip() else 0.0


def _tokenize(text: str) -> List[str]:
    return [tok for tok in (text or "").lower().strip().split() if tok]


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts: Dict[str, int] = defaultdict(int)
    ref_counts: Dict[str, int] = defaultdict(int)
    for token in pred_tokens:
        pred_counts[token] += 1
    for token in ref_tokens:
        ref_counts[token] += 1

    common = 0
    for token in pred_counts:
        common += min(pred_counts[token], ref_counts[token])

    if common == 0:
        return 0.0

    precision = common / float(len(pred_tokens))
    recall = common / float(len(ref_tokens))
    return 2.0 * precision * recall / (precision + recall)


def evaluate_rag_samples(samples: Sequence[Dict[str, Any]], ks: Sequence[int] = (1, 3, 5, 10)) -> Dict[str, Any]:
    if not samples:
        return {"info": {"n_samples": 0}, "results": {}}

    totals: Dict[str, float] = defaultdict(float)
    n_supported = 0
    n_support_labels = 0

    for sample in samples:
        retrieved_ids = sample.get("retrieved_ids", []) or []
        gold_ids = sample.get("gold_ids", []) or []
        ranking_scores = evaluate_ranking_case(retrieved_ids, gold_ids, ks=ks)
        for key, value in ranking_scores.items():
            totals[key] += float(value)

        answer = str(sample.get("answer", "") or "")
        gold_answer = str(sample.get("gold_answer", "") or "")
        totals["ExactMatch"] += exact_match(answer, gold_answer)
        totals["TokenF1"] += token_f1(answer, gold_answer)

        if "is_supported" in sample:
            n_support_labels += 1
            if bool(sample.get("is_supported")):
                n_supported += 1

    n = float(len(samples))
    results = {key: value / n for key, value in totals.items()}
    if n_support_labels > 0:
        results["SupportRate"] = n_supported / float(n_support_labels)

    return {
        "info": {
            "n_samples": len(samples),
            "ks": list(ks),
            "mode": "rag",
        },
        "results": results,
    }


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a simple RAG results JSON file.")
    parser.add_argument("json_path", help="Path to a JSON file containing a list of RAG samples.")
    args = parser.parse_args()

    data = load_json(args.json_path)
    if not isinstance(data, list):
        raise TypeError("Expected the top-level JSON value to be a list of samples.")

    report = evaluate_rag_samples(data)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
