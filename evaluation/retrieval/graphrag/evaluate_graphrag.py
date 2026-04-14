from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    _HERE = Path(__file__).resolve()
    for _candidate in _HERE.parents:
        if (_candidate / "evaluation" / "__init__.py").exists():
            _ROOT = str(_candidate)
            if _ROOT not in sys.path:
                sys.path.insert(0, _ROOT)
            break



from collections import defaultdict
from typing import Any, Dict, Sequence

from evaluation.retrieval.graphrag.metrics import evaluate_graphrag_sample
from evaluation.shared.utils.io import load_json


def evaluate_graphrag_samples(samples: Sequence[Dict[str, Any]], ks: Sequence[int] = (1, 3, 5, 10)) -> Dict[str, Any]:
    if not samples:
        return {"info": {"n_samples": 0, "mode": "graphrag"}, "results": {}}
    totals: Dict[str, float] = defaultdict(float)
    for sample in samples:
        scores = evaluate_graphrag_sample(sample, ks=ks)
        for key, value in scores.items():
            totals[key] += float(value)
    n = float(len(samples))
    results = {key: value / n for key, value in totals.items()}
    return {"info": {"n_samples": len(samples), "mode": "graphrag", "ks": list(ks)}, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a JSON list of GraphRAG samples.")
    parser.add_argument("json_path", help="Path to the GraphRAG JSON file.")
    args = parser.parse_args()
    data = load_json(args.json_path)
    if not isinstance(data, list):
        raise TypeError("Expected top-level JSON value to be a list of samples.")
    print(json.dumps(evaluate_graphrag_samples(data), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
