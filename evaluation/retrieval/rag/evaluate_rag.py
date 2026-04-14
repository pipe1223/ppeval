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
from typing import Any, Dict, Iterable, List, Sequence

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


def evaluate_rag_samples(samples: Sequence[Dict[str, Any]], ks: Sequence[int] = (1, 3, 5, 10)) -> Dict[str, Any]:
    if not samples:
        return {"info": {"mode": "rag", "n_samples": 0, "ks": list(ks)}, "results": {}}

    totals: Dict[str, float] = defaultdict(float)
    n_answer_pairs = 0
    n_support_labels = 0
    support_hits = 0.0

    for sample in samples:
        retrieved_ids = sample.get("retrieved_ids", []) or []
        gold_ids = sample.get("gold_ids", []) or []
        if gold_ids or retrieved_ids:
            for key, value in evaluate_ranking_case(retrieved_ids, gold_ids, ks=ks).items():
                totals[key] += float(value)

        if ("answer" in sample) or ("gold_answer" in sample):
            n_answer_pairs += 1
            answer = str(sample.get("answer", "") or "")
            gold_answer = str(sample.get("gold_answer", "") or "")
            totals["ExactMatch"] += exact_match(answer, gold_answer)
            totals["TokenF1"] += token_f1(answer, gold_answer)

        if "is_supported" in sample:
            n_support_labels += 1
            support_hits += 1.0 if bool(sample.get("is_supported")) else 0.0

    n = float(len(samples))
    results: Dict[str, float] = {}
    for key, value in totals.items():
        if key in {"ExactMatch", "TokenF1"}:
            denom = float(n_answer_pairs) if n_answer_pairs > 0 else n
            results[key] = value / denom if denom > 0 else 0.0
        else:
            results[key] = value / n
    if n_support_labels > 0:
        results["SupportRate"] = support_hits / float(n_support_labels)

    return {"info": {"mode": "rag", "n_samples": len(samples), "ks": list(ks)}, "results": results}


def evaluate_rag_file(path: str | Path, ks: Sequence[int] = (1, 3, 5, 10)) -> Dict[str, Any]:
    return evaluate_rag_samples(_extract_samples(_load_json(path)), ks=ks)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a RAG JSON file.")
    parser.add_argument("json_path", help="Path to a JSON file containing RAG samples.")
    args = parser.parse_args(argv)
    print(json.dumps(evaluate_rag_file(args.json_path), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
