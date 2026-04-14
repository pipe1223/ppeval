"""
evaluation_combined.py (patched)

Fix: segmentation.evaluate_segmentation may return 4 values OR 9 values:
  (m_dice, m_iou, m_vc8, m_vc16,
   m_class_dice, m_class_iou, m_class_vc8, m_class_vc16,
   gt_counter_per_class)

This patch makes the bridge accept either return signature and keeps the output
compatible (always includes the 4 global metrics; includes per-class metrics
when available).

Also tries importing segmentation from `eval_core.segmentation` first (your
renamed layout), then falls back to `eval_core.segmentation`.
"""

from __future__ import annotations

import argparse
import json
import inspect
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from result_gt_reader import EvalFile, load_eval_file

# Depending on your repo layout, const may live under eval_core.const
# or eval_core.const. Keep tianhuieval as the default.
from eval_core import const


def _extract_class(question: Optional[str]) -> Optional[str]:
    if not isinstance(question, str):
        return None
    q = question.strip()
    patterns = [
        r"Detect all (.*?) in the image",
        r"Segment out (.*?) in the image",
        r"Change detection of (.*?) in the image",
    ]
    for p in patterns:
        m = re.search(p, q, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def _call_segmentation(y_true: List[Any], y_pred: List[Any], crops: List[Any], class_list: Optional[List[str]]):
    # Try your local layout first
    try:
        from eval_core.segmentation import evaluate_segmentation  # type: ignore
    except Exception:
        from tianhuieval.segmentation import evaluate_segmentation  # type: ignore

    # Some versions accept (y_true, y_pred, crops, class_list), some accept only 3 args.
    try:
        sig = inspect.signature(evaluate_segmentation)
        if len(sig.parameters) >= 4:
            return evaluate_segmentation(y_true, y_pred, crops, class_list or [])
        return evaluate_segmentation(y_true, y_pred, crops)
    except Exception:
        # be permissive
        try:
            return evaluate_segmentation(y_true, y_pred, crops, class_list or [])
        except TypeError:
            return evaluate_segmentation(y_true, y_pred, crops)


def evaluate_from_ytrue_ypred(
    *,
    task: str,
    y_true: List[Any],
    y_pred: List[Any],
    crops: Optional[List[Any]] = None,
    questions: Optional[List[Optional[str]]] = None,
    iou: float = 0.5,
    dataset: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    task = str(task)

    if task in const.SEG_TAGS:
        if crops is None:
            raise ValueError(f"crops is required for task={task} (PIX_SEG/PIX_CHG)")

        class_list = None
        if questions:
            class_list = [(_extract_class(q) or "PIPE") for q in questions]

        out = _call_segmentation(y_true, y_pred, crops, class_list)

        if not isinstance(out, tuple):
            raise RuntimeError(f"evaluate_segmentation returned non-tuple: {type(out).__name__}")

        # Always support at least first 4
        if len(out) < 4:
            raise RuntimeError(f"evaluate_segmentation returned too few values: {len(out)}")

        m_dice, m_iou, m_vc8, m_vc16 = out[0], out[1], out[2], out[3]

        results_payload: Dict[str, Any] = {
            const.JSON_SAVE_MDICE: m_dice,
            const.JSON_SAVE_MIOU: m_iou,
            "mVC8": m_vc8,
            "mVC16": m_vc16,
        }

        # If per-class outputs are available (the common 9-value signature), include them.
        if len(out) >= 9:
            m_class_dice, m_class_iou, m_class_vc8, m_class_vc16, gt_counter_per_class = out[4], out[5], out[6], out[7], out[8]
            results_payload.update({
                "mClassDice": m_class_dice,
                "mClassIoU": m_class_iou,
                "mClassVC8": m_class_vc8,
                "mClassVC16": m_class_vc16,
                "GT_per_class": gt_counter_per_class,
            })

    elif task in const.CAP_TAGS:
        try:
            from eval_core.captioning import evaluate_captioning  # type: ignore
        except Exception:
            from tianhuieval.captioning import evaluate_captioning  # type: ignore
        bleu, meteor, rouge_score, rouge_l, cider = evaluate_captioning(y_true, y_pred)
        results_payload = {
            const.JSON_SAVE_BELU: bleu,
            const.JSON_SAVE_METERO: meteor,
            const.JSON_SAVE_ROUGE: rouge_score,
            const.JSON_SAVE_ROUGEL: rouge_l,
            const.JSON_SAVE_CIDER: cider,
        }

    elif task in const.VQA_TAGS or "VQA" in task:
        try:
            from eval_core.vqa import evaluate_vqa  # type: ignore
        except Exception:
            from tianhuieval.vqa import evaluate_vqa  # type: ignore
        acc, prec, rec, f1 = evaluate_vqa(y_true, y_pred)
        results_payload = {
            const.JSON_SAVE_ACC: acc * 100,
            const.JSON_SAVE_PREC: prec * 100,
            const.JSON_SAVE_RECALL: rec * 100,
            const.JSON_SAVE_F1: f1 * 100,
        }

    elif task in const.CLS_TAGS:
        class_names: List[str] = []
        for gt in y_true:
            s = str(gt).strip().lower()
            if s and s not in class_names:
                class_names.append(s)

        try:
            from eval_core.classification import evaluate_classification  # type: ignore
        except Exception:
            from tianhuieval.classification import evaluate_classification  # type: ignore

        out = evaluate_classification(
            [str(x).strip().lower() for x in y_true],
            [str(x).strip().lower() for x in y_pred],
            class_names,
        )

        results_payload = {"raw": out}

    elif task == "REG_DET_HBB":
        # Uses your in-memory detection.py
        try:
            from eval_core.detection_hbb import evaluate_detection  # type: ignore
        except Exception:
            from tianhuieval.detection import evaluate_detection  # type: ignore

        out = evaluate_detection(y_true, y_pred, iou=iou)

        if isinstance(out, tuple) and len(out) == 6:
            mAP, ap_dict, gt_count, macro_p, macro_r, macro_f1 = out
            results_payload = {
                "mAP": mAP,
                "AP_per_class": ap_dict,
                "GT_per_class": gt_count,
                const.JSON_SAVE_PREC: macro_p,
                const.JSON_SAVE_RECALL: macro_r,
                const.JSON_SAVE_F1: macro_f1,
                "IoU": iou,
            }
        else:
            results_payload = {"raw": out}

    elif task in ("REG_DET_OBB", "REG_VG"):
        try:
            from eval_core.detection_obb import evaluate_detection2  # type: ignore
        except Exception:
            from tianhuieval.detection_obb import evaluate_detection2  # type: ignore

        data_json: List[Dict[str, Any]] = []
        for i in range(len(y_true)):
            data_json.append(
                {
                    "gt": y_true[i],
                    "answer": y_pred[i],
                    "question": (questions[i] if questions and i < len(questions) else None),
                }
            )
        det = evaluate_detection2(data_json, pre_box=False, is_box=False, is_obb=True)
        results_payload = det

    else:
        results_payload = {"error": f"Unsupported task: {task}"}

    info_payload = {
        "task": task,
        "dataset": dataset,
        "model": model,
        "n_eval": len(y_true),
    }
    return {"info": info_payload, "results": results_payload}


def evaluate_evalfile(ef: EvalFile, *, iou: float = 0.5) -> Dict[str, Any]:
    task = ef.task
    y_true = ef.y_true()
    y_pred = ef.y_pred()
    crops = ef.crops(default=(0, 0, 0, 0)) if task in const.SEG_TAGS else None

    if task == "REG_DET_HBB":
        y_true_hbb, y_pred_hbb = ef.to_hbb_detection_lists()
        return evaluate_from_ytrue_ypred(
            task=task,
            y_true=y_true_hbb,
            y_pred=y_pred_hbb,
            crops=None,
            questions=[s.question for s in ef.data],
            iou=iou,
            dataset=ef.info.dataset,
            model=ef.info.model,
        )

    if task in ("REG_DET_OBB", "REG_VG"):
        return evaluate_from_ytrue_ypred(
            task=task,
            y_true=y_true,
            y_pred=y_pred,
            crops=None,
            questions=[s.question for s in ef.data],
            iou=iou,
            dataset=ef.info.dataset,
            model=ef.info.model,
        )

    return evaluate_from_ytrue_ypred(
        task=task,
        y_true=y_true,
        y_pred=y_pred,
        crops=crops,
        questions=[s.question for s in ef.data],
        iou=iou,
        dataset=ef.info.dataset,
        model=ef.info.model,
    )


def evaluate_combined_file(path: str | Path, *, iou: float = 0.5) -> Dict[str, Any]:
    ef = load_eval_file(path)
    return evaluate_evalfile(ef, iou=iou)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a combined (pred+gt) JSON (no disk I/O).")
    p.add_argument("json_path", help="Path to combined JSON file (contains answer+gt).")
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold for REG_DET_HBB (default: 0.5).")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    out = evaluate_combined_file(args.json_path, iou=float(args.iou))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
