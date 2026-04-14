from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

if __package__ in {None, ""}:  # pragma: no cover - script execution bootstrap
    _ROOT = str(Path(__file__).resolve().parents[1])
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from evaluation.online.pairwise import evaluate_pairwise_samples
from evaluation.retrieval.graphrag.evaluate_graphrag import evaluate_graphrag_samples
from evaluation.retrieval.rag.evaluate_rag import evaluate_rag_samples
from evaluation.shared.metrics.system import summarize_system_metrics
from evaluation.shared.utils.io import load_json
from evaluation.visual.legacy.evaluation_combined import evaluate_combined_file
from evaluation.visual.legacy.evaluation_separated import evaluate_files


def _looks_like_combined(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    data = obj.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        return isinstance(first, dict) and ("answer" in first) and ("gt" in first)
    return False


def _looks_like_separate(obj: Any) -> bool:
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
    return isinstance(obj, list) and bool(obj) and isinstance(obj[0], dict) and any(k in obj[0] for k in ("retrieved_node_ids", "gold_node_ids", "retrieved_edge_ids", "gold_edge_ids", "gold_paths"))


def _looks_like_rag(obj: Any) -> bool:
    return isinstance(obj, list) and bool(obj) and isinstance(obj[0], dict) and any(k in obj[0] for k in ("retrieved_ids", "gold_ids", "cited_ids", "gold_answer"))


def _looks_like_pairwise(obj: Any) -> bool:
    return isinstance(obj, list) and bool(obj) and isinstance(obj[0], dict) and any(k in obj[0] for k in ("winner", "baseline_output", "candidate_output"))


def _looks_like_system_records(obj: Any) -> bool:
    return isinstance(obj, list) and bool(obj) and isinstance(obj[0], dict) and any(k in obj[0] for k in ("latency_ms", "cost_usd", "input_tokens", "output_tokens", "success"))


def evaluate(pred_path: str | Path, *, gt_paths: Optional[Sequence[str]] = None, gt_dir: Optional[str] = None, result_key: str = "merged_results", iou: float = 0.5, strict: bool = False, mode: str = "auto") -> Dict[str, Any]:
    pred_path = str(pred_path)
    want_gt = bool(gt_paths) or bool(gt_dir)
    valid_modes = {"auto", "combined", "separate", "rag", "graphrag", "pairwise", "system"}
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of: {sorted(valid_modes)}")

    obj = load_json(pred_path)

    if mode == "combined":
        return _run_combined(pred_path, iou=iou)
    if mode == "separate":
        if not want_gt:
            raise ValueError("mode='separate' requires gt_paths or gt_dir")
        return _run_separate(pred_path, gt_paths=gt_paths, gt_dir=gt_dir, result_key=result_key, iou=iou, strict=strict)
    if mode == "rag":
        return _run_rag(obj)
    if mode == "graphrag":
        return _run_graphrag(obj)
    if mode == "pairwise":
        return _run_pairwise(obj)
    if mode == "system":
        return _run_system(obj)

    if want_gt:
        if _looks_like_combined(obj) and not _looks_like_separate(obj):
            out = _run_combined(pred_path, iou=iou)
            out.setdefault("info", {})
            out["info"]["mode"] = "combined"
            out["info"]["note"] = "GT paths were provided but the file looks like combined (answer+gt); ran combined evaluation. Use mode='separate' to force separate evaluation."
            return out
        return _run_separate(pred_path, gt_paths=gt_paths, gt_dir=gt_dir, result_key=result_key, iou=iou, strict=strict)

    if _looks_like_combined(obj):
        return _run_combined(pred_path, iou=iou)
    if _looks_like_separate(obj):
        return {"info": {"pred_path": pred_path, "mode": "auto"}, "results": {"error": "This JSON looks like a result-only visual file; provide gt_paths or gt_dir."}}
    if _looks_like_graphrag(obj):
        return _run_graphrag(obj)
    if _looks_like_rag(obj):
        return _run_rag(obj)
    if _looks_like_pairwise(obj):
        return _run_pairwise(obj)
    if _looks_like_system_records(obj):
        return _run_system(obj)

    return {"info": {"pred_path": pred_path, "mode": "auto"}, "results": {"error": "Could not detect evaluation schema. Use --mode to force routing."}}


def _run_combined(pred_path: str, *, iou: float) -> Dict[str, Any]:
    out = evaluate_combined_file(pred_path, iou=float(iou))
    out.setdefault("info", {})
    out["info"]["mode"] = "combined"
    return out


def _run_separate(pred_path: str, *, gt_paths: Optional[Sequence[str]], gt_dir: Optional[str], result_key: str, iou: float, strict: bool) -> Dict[str, Any]:
    out = evaluate_files(pred_path, gt_paths=list(gt_paths) if gt_paths else None, gt_dir=gt_dir, result_key=result_key, iou=float(iou), strict=bool(strict))
    out.setdefault("info", {})
    out["info"]["mode"] = "separate"
    return out


def _run_rag(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, list):
        raise TypeError("RAG evaluation expects a list of samples.")
    return evaluate_rag_samples(obj)


def _run_graphrag(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, list):
        raise TypeError("GraphRAG evaluation expects a list of samples.")
    return evaluate_graphrag_samples(obj)


def _run_pairwise(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, list):
        raise TypeError("Pairwise evaluation expects a list of samples.")
    return evaluate_pairwise_samples(obj)


def _run_system(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, list):
        raise TypeError("System evaluation expects a list of records.")
    return summarize_system_metrics(obj)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified PP Evaluation router for visual, RAG, GraphRAG, pairwise, and system metrics.")
    p.add_argument("pred", help="Path to the input JSON file.")
    p.add_argument("--gt", action="append", default=[], help="Path to a GT JSON (repeatable, separate visual mode only).")
    p.add_argument("--gt-dir", default=None, help="Folder containing GT JSON files (separate visual mode only).")
    p.add_argument("--result-key", default="merged_results", help="Result key for separate visual mode.")
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold for visual HBB evaluation.")
    p.add_argument("--strict", action="store_true", help="Fail if any GT is missing in separate visual mode.")
    p.add_argument("--mode", default="auto", choices=["auto", "combined", "separate", "rag", "graphrag", "pairwise", "system"], help="Force routing mode.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    out = evaluate(args.pred, gt_paths=args.gt if args.gt else None, gt_dir=args.gt_dir, result_key=args.result_key, iou=float(args.iou), strict=bool(args.strict), mode=str(args.mode))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
