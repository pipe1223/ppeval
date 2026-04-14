
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, List


def _load_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _looks_like_combined(obj: Any) -> bool:
    """
    Combined schema typically has:
      - top-level info.task
      - data is a list
      - each item has "answer" and "gt" (and often "question"/"crop"/"image")
    """
    if not isinstance(obj, dict):
        return False
    data = obj.get("data")
    if not isinstance(data, list) or not data:
        return False
    first = data[0]
    if not isinstance(first, dict):
        return False
    return ("answer" in first) and ("gt" in first)


def _looks_like_separate(obj: Any) -> bool:
    """
    Separate schema typically has:
      - data is a list OR dict keyed by image_path
      - each entry has task_config + results
    """
    if not isinstance(obj, dict):
        return False
    data = obj.get("data")
    entry = None
    if isinstance(data, list) and data:
        entry = data[0]
    elif isinstance(data, dict) and data:
        # peek first value
        entry = next(iter(data.values()))
    if not isinstance(entry, dict):
        return False
    return ("task_config" in entry) and ("results" in entry)


def evaluate(
    pred_path: str | Path,
    *,
    gt_paths: Optional[Sequence[str]] = None,
    gt_dir: Optional[str] = None,
    result_key: str = "merged_results",
    iou: float = 0.5,
    strict: bool = False,
    mode: str = "auto",  # "auto" | "combined" | "separate"
) -> Dict[str, Any]:
    """
    Unified evaluation entrypoint.

    Parameters
    ----------
    pred_path:
      Path to either a result-only JSON (separate mode) or a combined result+GT JSON (combined mode).
    gt_paths / gt_dir:
      Optional GT(s) (used in separate mode).
    mode:
      - "auto"     : route based on presence of gt_paths/gt_dir and file schema
      - "combined" : force combined evaluation (ignores gt_paths/gt_dir)
      - "separate" : force separate evaluation (requires gt_paths or gt_dir)

    Returns
    -------
    Dict[str,Any] : bridge-style output
    """
    pred_path = str(pred_path)
    want_gt = bool(gt_paths) or bool(gt_dir)

    if mode not in {"auto", "combined", "separate"}:
        raise ValueError("mode must be one of: auto, combined, separate")

    # Quick schema sniff
    obj = _load_json(pred_path)
    is_combined = _looks_like_combined(obj)
    is_separate = _looks_like_separate(obj)

    # Mode forcing
    if mode == "combined":
        return _run_combined(pred_path, iou=iou)
    if mode == "separate":
        if not want_gt:
            raise ValueError("mode='separate' requires gt_paths or gt_dir")
        return _run_separate(pred_path, gt_paths=gt_paths, gt_dir=gt_dir, result_key=result_key, iou=iou, strict=strict)

    # Auto routing
    if want_gt:
        # If user provided GT, we assume they want the separate pipeline.
        # If the file is actually combined, we still prefer combined unless user forces separate.
        if is_combined and not is_separate:
            # user provided GT but file is clearly combined; do combined eval by default
            out = _run_combined(pred_path, iou=iou)
            out.setdefault("info", {})
            out["info"]["mode"] = "combined"
            out["info"]["note"] = "GT paths were provided but the file looks like combined (answer+gt); ran combined evaluation. Use mode='separate' to force separate evaluation."
            return out
        return _run_separate(pred_path, gt_paths=gt_paths, gt_dir=gt_dir, result_key=result_key, iou=iou, strict=strict)

    # No GT provided: only possible if combined
    if is_combined:
        return _run_combined(pred_path, iou=iou)

    # Otherwise we can't evaluate
    hint = "This JSON looks like a result-only file; provide gt_paths or gt_dir."
    if is_separate:
        hint += " (Detected separate schema: task_config/results)"
    elif not (is_combined or is_separate):
        hint += " (Could not detect schema; check JSON structure.)"

    return {"info": {"pred_path": pred_path, "mode": "auto"}, "results": {"error": hint}}


def _run_combined(pred_path: str, *, iou: float) -> Dict[str, Any]:
    # Prefer local module names; user may have copied files with different names.
    try:
        from evaluation_combined import evaluate_combined_file  # type: ignore
    except Exception as e:
        raise ImportError("Could not import evaluation_combined.evaluate_combined_file") from e

    out = evaluate_combined_file(pred_path, iou=float(iou))
    out.setdefault("info", {})
    out["info"]["mode"] = "combined"
    return out


def _run_separate(
    pred_path: str,
    *,
    gt_paths: Optional[Sequence[str]],
    gt_dir: Optional[str],
    result_key: str,
    iou: float,
    strict: bool,
) -> Dict[str, Any]:
    # Use latest multitask bridge (v3) if present; otherwise fall back to v2/v1.
    try:
        from evaluation_separated import evaluate_files  # type: ignore
    except Exception as e:
        raise ImportError("Could not import multitask bridge evaluate_files") from e

    out = evaluate_files(
        pred_path,
        gt_paths=list(gt_paths) if gt_paths else None,
        gt_dir=gt_dir,
        result_key=result_key,
        iou=float(iou),
        strict=bool(strict),
    )
    out.setdefault("info", {})
    out["info"]["mode"] = "separate"
    return out


# -----------------------
# CLI
# -----------------------

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
