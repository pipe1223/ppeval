from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .result_reader import load_results
from .gt_reader import load_gt
from .eval_core import const


def _extract_class(question: Optional[str]) -> Optional[str]:
    if not isinstance(question, str):
        return None
    q = question.strip()
    patterns = [r"Detect all (.*?) in the image", r"Segment out (.*?) in the image", r"Is there a (.*?) in the image"]
    for p in patterns:
        m = re.search(p, q, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def _label_to_multilabel_str(label: Any) -> str:
    if label is None:
        return ""
    if isinstance(label, list):
        parts = []
        for x in label:
            if x is None:
                continue
            parts.append(x.strip() if isinstance(x, str) else str(x))
        return ", ".join([p for p in parts if p])
    if isinstance(label, str):
        return label.strip()
    return str(label)


def _first_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        return _first_text(x[0]) if x else ""
    return x.strip() if isinstance(x, str) else str(x)


def _gt_entry_to_question(gt_entry) -> Optional[str]:
    try:
        if gt_entry.gt and gt_entry.gt[0].question:
            return gt_entry.gt[0].question
    except Exception:
        pass
    try:
        q = gt_entry.extra.get("question")
    except Exception:
        q = None
    return q if isinstance(q, str) else None


def _gt_entry_to_crop(gt_entry) -> Optional[Tuple[float, float, float, float]]:
    try:
        for it in gt_entry.gt:
            if it.crop is not None:
                return it.crop
    except Exception:
        pass
    try:
        c = gt_entry.extra.get("crop")
    except Exception:
        c = None
    if isinstance(c, (list, tuple)) and len(c) == 4 and all(isinstance(v, (int, float)) for v in c):
        return (float(c[0]), float(c[1]), float(c[2]), float(c[3]))
    return None


def _build_pairs_for_task(entries, gt_file, *, result_key: str, strict: bool):
    y_true_raw: List[Any] = []
    y_pred_raw: List[Any] = []
    questions: List[Optional[str]] = []
    crops: List[Tuple[float, float, float, float]] = []
    missing = 0
    for e in entries:
        key_candidates = [getattr(e, "key", None), e.task_config.image_path]
        gt_entry = None
        for k in key_candidates:
            if not k:
                continue
            try:
                gt_entry = gt_file.get_entry(str(k))
                break
            except Exception:
                continue
        if gt_entry is None:
            missing += 1
            if strict:
                raise KeyError(f"Missing GT for image '{e.task_config.image_path}'")
            continue
        rs = e.get_result(result_key)
        y_pred_raw.append(rs.prediction)
        y_true_raw.append(gt_entry.gt)
        questions.append(_gt_entry_to_question(gt_entry))
        crop = _gt_entry_to_crop(gt_entry)
        crops.append(crop if crop is not None else (0.0, 0.0, 0.0, 0.0))
    return y_true_raw, y_pred_raw, questions, crops, missing


def _to_cls_lists(y_true_raw, y_pred_raw):
    y_true, y_pred = [], []
    for gt_items, pred_items in zip(y_true_raw, y_pred_raw):
        y_true.append((_label_to_multilabel_str(getattr(gt_items[0], "label", gt_items[0])) if gt_items else "").lower())
        pred_labels: List[str] = []
        for p in pred_items:
            if hasattr(p, "text"):
                s = str(getattr(p, "text")).strip()
                if s:
                    pred_labels.append(s)
            elif hasattr(p, "label"):
                s = str(getattr(p, "label")).strip()
                if s:
                    pred_labels.append(s)
            else:
                d = getattr(p, "data", None)
                if isinstance(d, str) and d.strip():
                    pred_labels.append(d.strip())
        y_pred.append(", ".join(pred_labels).lower())
    return y_true, y_pred


def _to_text_lists(y_true_raw, y_pred_raw):
    y_true, y_pred = [], []
    for gt_items, pred_items in zip(y_true_raw, y_pred_raw):
        y_true.append(_first_text(getattr(gt_items[0], "label", "")) if gt_items else "")
        pred_text = ""
        for p in pred_items:
            if hasattr(p, "text"):
                pred_text = str(getattr(p, "text")).strip()
                break
        if not pred_text:
            for p in pred_items:
                d = getattr(p, "data", None)
                if isinstance(d, str) and d.strip():
                    pred_text = d.strip()
                    break
        y_pred.append(pred_text)
    return y_true, y_pred


def _to_seg_polygons(y_true_raw, y_pred_raw):
    def gt_items_to_polys(gt_items):
        polys = []
        for it in gt_items or []:
            coord = getattr(it, "coord", None)
            if coord is not None:
                polys.append([[float(x) for x in coord]])
                continue
            extra = getattr(it, "extra", {}) or {}
            cr = extra.get("coord_raw")
            if isinstance(cr, list):
                if cr and all(isinstance(x, (int, float)) for x in cr) and len(cr) % 2 == 0:
                    polys.append([[float(x) for x in cr]])
                elif cr and isinstance(cr[0], list):
                    ring = cr[0]
                    if isinstance(ring, list) and ring and all(isinstance(x, (int, float)) for x in ring) and len(ring) % 2 == 0:
                        polys.append([[float(x) for x in ring]])
        return polys
    def pred_items_to_polys(pred_items):
        polys = []
        for p in pred_items or []:
            if hasattr(p, "coord"):
                coord = getattr(p, "coord")
                if coord is not None and len(coord) % 2 == 0:
                    polys.append([[float(x) for x in coord]])
                    continue
            d = getattr(p, "data", None)
            if isinstance(d, list):
                if d and all(isinstance(x, (int, float)) for x in d) and len(d) % 2 == 0:
                    polys.append([[float(x) for x in d]])
                elif d and isinstance(d[0], list):
                    ring = d[0]
                    if isinstance(ring, list) and ring and all(isinstance(x, (int, float)) for x in ring) and len(ring) % 2 == 0:
                        polys.append([[float(x) for x in ring]])
        return polys
    return [gt_items_to_polys(x) for x in y_true_raw], [pred_items_to_polys(x) for x in y_pred_raw]


def _to_hbb_lists(y_true_raw, y_pred_raw, questions):
    def to_bbox4(coord):
        if coord is None:
            return None
        if isinstance(coord, (list, tuple)) and len(coord) >= 4 and all(isinstance(x, (int, float)) for x in coord[:4]):
            return [float(coord[0]), float(coord[1]), float(coord[2]), float(coord[3])]
        return None
    y_true, y_pred = [], []
    for gt_items, pred_items, q in zip(y_true_raw, y_pred_raw, questions):
        default_cls = _extract_class(q) or "PIPE"
        gt_list = []
        for it in gt_items or []:
            bbox = to_bbox4(getattr(it, "coord", None))
            if bbox is None:
                extra = getattr(it, "extra", {}) or {}
                bbox = to_bbox4(extra.get("coord_raw"))
            if bbox is None:
                continue
            lab = getattr(it, "label", None)
            lab_s = _first_text(lab) if lab is not None else default_cls
            if not lab_s:
                lab_s = default_cls
            item = {"label": lab_s, "bbox": bbox}
            extra = getattr(it, "extra", {}) or {}
            if "difficult" in extra:
                try:
                    item["difficult"] = bool(extra["difficult"])
                except Exception:
                    pass
            gt_list.append(item)
        pred_list = []
        for p in pred_items or []:
            if hasattr(p, "coord"):
                bbox = to_bbox4(getattr(p, "coord", None))
                if bbox is None:
                    continue
                lab = getattr(p, "label", None) or default_cls
                conf = getattr(p, "confidence", None)
                try:
                    conf_f = float(conf) if conf is not None else 1.0
                except Exception:
                    conf_f = 1.0
                pred_list.append({"label": str(lab), "bbox": bbox, "confidence": conf_f})
        y_true.append(gt_list)
        y_pred.append(pred_list)
    return y_true, y_pred


def _to_obb_data(y_true_raw, y_pred_raw, questions):
    def to_quad8(coord):
        if coord is None:
            return None
        if isinstance(coord, (list, tuple)) and len(coord) == 8 and all(isinstance(x, (int, float)) for x in coord):
            return [float(x) for x in coord]
        return None
    data = []
    for gt_items, pred_items, q in zip(y_true_raw, y_pred_raw, questions):
        gt_quads, pred_quads = [], []
        for it in gt_items or []:
            quad = to_quad8(getattr(it, "coord", None))
            if quad is None:
                extra = getattr(it, "extra", {}) or {}
                quad = to_quad8(extra.get("coord_raw"))
            if quad is not None:
                gt_quads.append(quad)
        for p in pred_items or []:
            if hasattr(p, "coord"):
                quad = to_quad8(getattr(p, "coord", None))
                if quad is not None:
                    pred_quads.append(quad)
            else:
                d = getattr(p, "data", None)
                if isinstance(d, list) and d and isinstance(d[0], (int, float)) and len(d) == 8:
                    pred_quads.append([float(x) for x in d])
        if not q:
            synth = None
            if gt_items:
                lab = getattr(gt_items[0], "label", None)
                lab_s = _first_text(lab)
                if lab_s:
                    synth = f"Detect all {lab_s} in the image"
            q = synth or "Detect all object in the image"
        data.append({"question": q, "gt": gt_quads, "answer": pred_quads})
    return data


def _evaluate_one_task(*, task_type: str, entries, gt_file, result_key: str, iou: float, strict: bool):
    y_true_raw, y_pred_raw, questions, crops, n_missing = _build_pairs_for_task(entries, gt_file, result_key=result_key, strict=strict)

    if task_type in const.CLS_TAGS:
        try:
            from .eval_core.classification import evaluate_classification  # type: ignore
        except Exception:
            from tianhuieval.classification import evaluate_classification  # type: ignore
        y_true, y_pred = _to_cls_lists(y_true_raw, y_pred_raw)
        classes = []
        for s in y_true:
            for lab in [x.strip() for x in s.split(",")]:
                if lab and lab not in classes:
                    classes.append(lab)
        results = {"raw": evaluate_classification(y_true, y_pred, classes)}
    elif task_type in const.CAP_TAGS:
        try:
            from .eval_core.captioning import evaluate_captioning  # type: ignore
        except Exception:
            from tianhuieval.captioning import evaluate_captioning  # type: ignore
        y_true, y_pred = _to_text_lists(y_true_raw, y_pred_raw)
        bleu, meteor, rouge_score, rouge_l, cider = evaluate_captioning(y_true, y_pred)
        results = {const.JSON_SAVE_BELU: bleu, const.JSON_SAVE_METERO: meteor, const.JSON_SAVE_ROUGE: rouge_score, const.JSON_SAVE_ROUGEL: rouge_l, const.JSON_SAVE_CIDER: cider}
    elif task_type in const.VQA_TAGS or "VQA" in task_type:
        try:
            from .eval_core.vqa import evaluate_vqa  # type: ignore
        except Exception:
            from tianhuieval.vqa import evaluate_vqa  # type: ignore
        y_true, y_pred = _to_text_lists(y_true_raw, y_pred_raw)
        acc, prec, rec, f1 = evaluate_vqa(y_true, y_pred)
        results = {const.JSON_SAVE_ACC: acc * 100, const.JSON_SAVE_PREC: prec * 100, const.JSON_SAVE_RECALL: rec * 100, const.JSON_SAVE_F1: f1 * 100}
    elif task_type in const.SEG_TAGS:
        try:
            from .eval_core.segmentation import evaluate_segmentation  # type: ignore
        except Exception:
            from tianhuieval.segmentation import evaluate_segmentation  # type: ignore
        y_true, y_pred = _to_seg_polygons(y_true_raw, y_pred_raw)
        class_list = [_extract_class(q) or "PIPE" for q in questions]
        m_dice, m_iou, m_vc8, m_vc16 = evaluate_segmentation(y_true, y_pred, crops, class_list)
        results = {const.JSON_SAVE_MDICE: m_dice, const.JSON_SAVE_MIOU: m_iou, "mVC8": m_vc8, "mVC16": m_vc16}
    elif task_type == "REG_DET_HBB":
        try:
            from .eval_core.detection_hbb import evaluate_detection  # type: ignore
        except Exception:
            from tianhuieval.detection import evaluate_detection  # type: ignore
        y_true_det, y_pred_det = _to_hbb_lists(y_true_raw, y_pred_raw, questions)
        out = evaluate_detection(y_true_det, y_pred_det, iou=iou)
        if isinstance(out, tuple) and len(out) == 6:
            mAP, ap_dict, gt_count, macro_p, macro_r, macro_f1 = out
            results = {"mAP": mAP, "AP_per_class": ap_dict, "GT_per_class": gt_count, const.JSON_SAVE_PREC: macro_p, const.JSON_SAVE_RECALL: macro_r, const.JSON_SAVE_F1: macro_f1, "IoU": iou}
        else:
            results = {"raw": out}
    elif task_type in ("REG_DET_OBB", "REG_VG"):
        try:
            from .eval_core.detection_obb import evaluate_detection2  # type: ignore
        except Exception:
            from tianhuieval.detection_obb import evaluate_detection2  # type: ignore
        results = evaluate_detection2(_to_obb_data(y_true_raw, y_pred_raw, questions), pre_box=False, is_box=False, is_obb=True)
    else:
        results = {"error": f"Unsupported task_type: {task_type}"}

    return {"info": {"task": task_type, "n_eval": len(y_true_raw), "n_missing_gt": n_missing, "result_key": result_key}, "results": results}


def load_gt_map(*, gt_paths: Optional[Sequence[str]] = None, gt_dir: Optional[str] = None) -> Dict[str, Any]:
    paths: List[str] = []
    if gt_paths:
        paths.extend([str(p) for p in gt_paths])
    if gt_dir:
        gt_dir_p = Path(gt_dir)
        if gt_dir_p.is_dir():
            paths.extend([str(p) for p in sorted(gt_dir_p.glob("*.json"))])
    gt_map: Dict[str, Any] = {}
    for p in paths:
        gf = load_gt(p)
        try:
            split = gf.split_by_task()  # type: ignore[attr-defined]
        except Exception:
            split = None
        if isinstance(split, dict) and split and not (len(split) == 1 and "UNKNOWN" in split):
            for t, sub in split.items():
                if t and t != "UNKNOWN":
                    gt_map[str(t)] = sub
            continue
        task = getattr(gf.info, "task", None)
        if not task:
            m = re.search(r"\[([^\]]+)\]", Path(p).name)
            task = m.group(1) if m else None
        if task:
            gt_map[str(task)] = gf
    return gt_map


def evaluate_files(pred_path: str | Path, *, gt_paths: Optional[Sequence[str]] = None, gt_dir: Optional[str] = None, result_key: str = "merged_results", iou: float = 0.5, strict: bool = False) -> Dict[str, Any]:
    rf = load_results(pred_path)
    gt_map = load_gt_map(gt_paths=gt_paths, gt_dir=gt_dir)
    groups: Dict[str, List[Any]] = {}
    for e in rf.data:
        groups.setdefault(e.task_config.task_type, []).append(e)
    out_by_task: Dict[str, Any] = {}
    for tt, entries in groups.items():
        gf = gt_map.get(tt)
        if gf is None:
            out_by_task[tt] = {"info": {"task": tt, "error": "missing_gt", "n_entries": len(entries)}, "results": {"error": f"No GT found for task={tt}. Provide GT via --gt/--gt-dir and ensure GT has info.task or per-item task_type in gt[]."}}
            continue
        out_by_task[tt] = _evaluate_one_task(task_type=tt, entries=entries, gt_file=gf, result_key=result_key, iou=iou, strict=strict)
    return {"info": {"pred_path": str(pred_path), "model": rf.info.model, "dataset": rf.info.dataset, "result_key": result_key, "task_types": sorted(groups.keys())}, "by_task": out_by_task}


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate results.json vs GT json(s) for all task_type groups.")
    p.add_argument("--pred", required=True, help="Path to result-only JSON.")
    p.add_argument("--gt", action="append", default=[], help="Path to a GT JSON (repeat for multiple tasks).")
    p.add_argument("--gt-dir", default=None, help="Folder containing GT JSON files.")
    p.add_argument("--result-key", default="merged_results", help="Which result key to evaluate (e.g. merged_results, 0.2).")
    p.add_argument("--iou", type=float, default=0.5, help="IoU for REG_DET_HBB (default: 0.5).")
    p.add_argument("--strict", action="store_true", help="Fail if any GT is missing for a sample.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    out = evaluate_files(args.pred, gt_paths=args.gt if args.gt else None, gt_dir=args.gt_dir, result_key=args.result_key, iou=float(args.iou), strict=bool(args.strict))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
