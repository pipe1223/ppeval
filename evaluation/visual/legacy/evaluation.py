from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


def _load_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _looks_like_combined(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    data = obj.get("data")
    if not isinstance(data, list) or not data:
        return False
    first = data[0]
    return isinstance(first, dict) and ("answer" in first) and ("gt" in first)


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


def evaluate(pred_path: str | Path, *, gt_paths: Optional[Sequence[str]] = None, gt_dir: Optional[str] = None, result_key: str = "merged_results", iou: float = 0.5, strict: bool = False, mode: str = "auto") -> Dict[str, Any]:
    pred_path = str(pred_path)
    want_gt = bool(gt_paths) or bool(gt_dir)
    if mode not in {"auto", "combined", "separate"}:
        raise ValueError("mode must be one of: auto, combined, separate")
    obj = _load_json(pred_path)
    is_combined = _looks_like_combined(obj)
    is_separate = _looks_like_separate(obj)
    if mode == "combined":
        return _run_combined(pred_path, iou=iou)
    if mode == "separate":
        if not want_gt:
            raise ValueError("mode='separate' requires gt_paths or gt_dir")
        return _run_separate(pred_path, gt_paths=gt_paths, gt_dir=gt_dir, result_key=result_key, iou=iou, strict=strict)
    if want_gt:
        if is_combined and not is_separate:
            out = _run_combined(pred_path, iou=iou)
            out.setdefault("info", {})
            out["info"]["mode"] = "combined"
            out["info"]["note"] = "GT paths were provided but the file looks like combined (answer+gt); ran combined evaluation. Use mode='separate' to force separate evaluation."
            return out
        return _run_separate(pred_path, gt_paths=gt_paths, gt_dir=gt_dir, result_key=result_key, iou=iou, strict=strict)
    if is_combined:
        return _run_combined(pred_path, iou=iou)
    hint = "This JSON looks like a result-only file; provide gt_paths or gt_dir."
    if is_separate:
        hint += " (Detected separate schema: task_config/results)"
    elif not (is_combined or is_separate):
        hint += " (Could not detect schema; check JSON structure.)"
    return {"info": {"pred_path": pred_path, "mode": "auto"}, "results": {"error": hint}}


def _run_combined(pred_path: str, *, iou: float) -> Dict[str, Any]:
    try:
        from .evaluation_combined import evaluate_combined_file  # type: ignore
    except Exception:
        from evaluation_combined import evaluate_combined_file  # type: ignore
    out = evaluate_combined_file(pred_path, iou=float(iou))
    out.setdefault("info", {})
    out["info"]["mode"] = "combined"
    return out


def _run_separate(pred_path: str, *, gt_paths: Optional[Sequence[str]], gt_dir: Optional[str], result_key: str, iou: float, strict: bool) -> Dict[str, Any]:
    try:
        from .evaluation_separated import evaluate_files  # type: ignore
    except Exception:
        from evaluation_separated import evaluate_files  # type: ignore
    out = evaluate_files(pred_path, gt_paths=list(gt_paths) if gt_paths else None, gt_dir=gt_dir, result_key=result_key, iou=float(iou), strict=bool(strict))
    out.setdefault("info", {})
    out["info"]["mode"] = "separate"
    return out


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified bridge for separate and combined evaluation JSON formats.")
    p.add_argument("pred", help="Path to result-only JSON OR combined (answer+gt) JSON.")
    p.add_argument("--gt", action="append", default=[], help="Path to a GT JSON (repeatable).")
    p.add_argument("--gt-dir", default=None, help="Folder containing per-task GT JSON files.")
    p.add_argument("--result-key", default="merged_results", help="Which result key to evaluate (separate mode).")
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold for REG_DET_HBB.")
    p.add_argument("--strict", action="store_true", help="Fail if any GT is missing (separate mode).")
    p.add_argument("--mode", default="auto", choices=["auto", "combined", "separate"], help="Force routing mode.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    out = evaluate(args.pred, gt_paths=args.gt if args.gt else None, gt_dir=args.gt_dir, result_key=args.result_key, iou=float(args.iou), strict=bool(args.strict), mode=str(args.mode))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
