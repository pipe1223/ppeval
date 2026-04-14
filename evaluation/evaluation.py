from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from evaluation.visual.legacy.evaluation import evaluate as evaluate_visual
from evaluation.retrieval.rag.evaluate_rag import evaluate_rag_file
from evaluation.retrieval.graphrag.evaluate_graphrag import evaluate_graphrag_file


def _load_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_records(obj: Any) -> list[dict]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        data = obj.get("data")
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
    return []


def _looks_like_visual_combined(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    data = obj.get("data")
    if not isinstance(data, list) or not data:
        return False
    first = data[0]
    return isinstance(first, dict) and ("answer" in first) and ("gt" in first)


def _looks_like_visual_separate(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    data = obj.get("data")
    entry = None
    if isinstance(data, list) and data:
        entry = data[0]
    elif isinstance(data, dict) and data:
        entry = next(iter(data.values()))
    return isinstance(entry, dict) and ("task_config" in entry) and ("results" in entry)


def _looks_like_graphrag(obj: Any) -> bool:
    if isinstance(obj, dict) and str(obj.get("task_family", "")).lower() == "graphrag":
        return True
    for rec in _extract_records(obj):
        if any(k in rec for k in ("gold_node_ids", "retrieved_node_ids", "gold_edge_ids", "retrieved_edge_ids", "gold_paths", "retrieved_paths")):
            return True
    return False


def _looks_like_rag(obj: Any) -> bool:
    if isinstance(obj, dict) and str(obj.get("task_family", "")).lower() == "rag":
        return True
    records = _extract_records(obj)
    if not records:
        return False
    first = records[0]
    return any(k in first for k in ("retrieved_ids", "gold_ids", "answer", "gold_answer"))


def evaluate(
    pred_path: str | Path,
    *,
    gt_paths: Optional[Sequence[str]] = None,
    gt_dir: Optional[str] = None,
    result_key: str = "merged_results",
    iou: float = 0.5,
    strict: bool = False,
    mode: str = "auto",
    ks: Sequence[int] = (1, 3, 5, 10),
) -> Dict[str, Any]:
    pred_path = str(pred_path)
    obj = _load_json(pred_path)
    want_gt = bool(gt_paths) or bool(gt_dir)

    if mode not in {"auto", "visual-combined", "visual-separate", "rag", "graphrag"}:
        raise ValueError("mode must be one of: auto, visual-combined, visual-separate, rag, graphrag")

    if mode == "visual-combined":
        return evaluate_visual(pred_path, iou=iou, mode="combined")
    if mode == "visual-separate":
        return evaluate_visual(pred_path, gt_paths=gt_paths, gt_dir=gt_dir, result_key=result_key, iou=iou, strict=strict, mode="separate")
    if mode == "rag":
        return evaluate_rag_file(pred_path, ks=ks)
    if mode == "graphrag":
        return evaluate_graphrag_file(pred_path, ks=ks)

    if want_gt:
        return evaluate_visual(pred_path, gt_paths=gt_paths, gt_dir=gt_dir, result_key=result_key, iou=iou, strict=strict, mode="separate")
    if _looks_like_graphrag(obj):
        return evaluate_graphrag_file(pred_path, ks=ks)
    if _looks_like_rag(obj):
        return evaluate_rag_file(pred_path, ks=ks)
    if _looks_like_visual_combined(obj):
        return evaluate_visual(pred_path, iou=iou, mode="combined")
    if _looks_like_visual_separate(obj):
        return {
            "info": {"pred_path": pred_path, "mode": "auto"},
            "results": {"error": "This looks like a visual result-only file. Provide gt_paths or gt_dir, or use mode='visual-separate'."},
        }
    return {
        "info": {"pred_path": pred_path, "mode": "auto"},
        "results": {"error": "Could not infer evaluation family from the provided JSON."},
    }


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified router for visual, RAG, and GraphRAG evaluation.")
    p.add_argument("pred", help="Path to the evaluation JSON file.")
    p.add_argument("--gt", action="append", default=[], help="GT JSON path(s) for visual separate evaluation.")
    p.add_argument("--gt-dir", default=None, help="Folder containing GT JSON files for visual separate evaluation.")
    p.add_argument("--result-key", default="merged_results", help="Which result key to evaluate in visual separate mode.")
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold for HBB detection.")
    p.add_argument("--strict", action="store_true", help="Fail if any GT is missing in visual separate mode.")
    p.add_argument("--mode", default="auto", choices=["auto", "visual-combined", "visual-separate", "rag", "graphrag"], help="Force the evaluation family.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    out = evaluate(
        args.pred,
        gt_paths=args.gt if args.gt else None,
        gt_dir=args.gt_dir,
        result_key=args.result_key,
        iou=float(args.iou),
        strict=bool(args.strict),
        mode=str(args.mode),
    )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
