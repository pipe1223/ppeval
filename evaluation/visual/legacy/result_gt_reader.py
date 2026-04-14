from __future__ import annotations

"""
result_gt_reader.py

Reader for "combined" evaluation JSONs that contain BOTH predictions ("answer")
and ground truth ("gt") in the same file.

Expected top-level schema:
{
  "info": {"task": "...", "model": "...", "dataset": "..."},
  "data": [
    {"image": [... or str], "crop": [...], "question": "...", "answer": ..., "gt": ...}
  ]
}

Notes:
- `image` may be a string or a list:
    - ["path/to/img"] -> image_path
    - ["path/to/t1", "path/to/t2"] -> image_path, image2_path (e.g., change detection)
- `crop` may be [] or [x1,y1,x2,y2]; empty becomes None.
- `data` may also be a dict keyed by id:
    "data": {"sample_id": {...}, "sample_id2": {...}}
  which will be normalized into a list with "id" injected.

This module is intentionally dependency-free (stdlib only).
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


Number = float


# ----------------------------
# Dataclasses
# ----------------------------

@dataclass(frozen=True)
class EvalInfo:
    task: str
    model: Optional[str]
    dataset: Optional[str]
    json_path: str


@dataclass(frozen=True)
class EvalSample:
    """
    One evaluation sample containing:
    - question/prompt
    - prediction in `answer`
    - ground truth in `gt`
    """
    id: Optional[str]
    image_path: str
    image2_path: Optional[str]
    crop: Optional[Tuple[Number, Number, Number, Number]]
    question: Optional[str]
    answer: Any
    gt: Any
    meta: Mapping[str, Any]

    def extracted_class(self) -> Optional[str]:
        """
        Best-effort extraction of a class/target name from common prompt patterns.

        Examples:
          "Detect all swimming pool in the image.\nUse horizontal bounding boxes."
          -> "swimming pool"

          "Segment out road in the image."
          -> "road"
        """
        if not self.question:
            return None

        q = self.question.strip()

        # Detect all <class> in the image
        m = re.search(r"detect\s+all\s+(.*?)\s+in\s+the\s+image", q, flags=re.IGNORECASE)
        if m:
            return _clean_label(m.group(1))

        # Segment out <class> in the image
        m = re.search(r"segment\s+out\s+(.*?)\s+in\s+the\s+image", q, flags=re.IGNORECASE)
        if m:
            return _clean_label(m.group(1))

        # Describe the <box> ... or Classify the region ... (not class-extractable)
        return None


@dataclass(frozen=True)
class EvalFile:
    info: EvalInfo
    data: Tuple[EvalSample, ...]

    def __len__(self) -> int:
        return len(self.data)

    @property
    def task(self) -> str:
        return self.info.task

    def y_true(self) -> List[Any]:
        return [s.gt for s in self.data]

    def y_pred(self) -> List[Any]:
        return [s.answer for s in self.data]

    def crops(self, default: Tuple[Number, Number, Number, Number] = (0, 0, 0, 0)) -> List[Tuple[Number, Number, Number, Number]]:
        """
        Return a crop list aligned with data[].

        If crop is missing/empty for some samples, returns `default` for those.
        (Segmentation evaluators commonly expect a crop for every sample.)
        """
        out: List[Tuple[Number, Number, Number, Number]] = []
        for s in self.data:
            out.append(s.crop if s.crop is not None else default)
        return out

    def class_list_from_questions(self, default: str = "PIPE") -> List[str]:
        """
        Return a per-sample class list, extracted from question when possible.

        Useful for segmentation per-class reporting (optional) and for detection when labels are not explicit.
        """
        cls: List[str] = []
        for s in self.data:
            c = s.extracted_class()
            cls.append(c if c else default)
        return cls

    # ----------------------------
    # Converters for evaluation
    # ----------------------------

    def to_hbb_detection_lists(
        self,
        *,
        iou_label_from_question: bool = True,
        default_label: str = "PIPE",
        normalize_label: bool = True,
    ) -> Tuple[List[List[Dict[str, Any]]], List[List[Dict[str, Any]]]]:
        """
        Convert combined JSON samples (gt/answer as [l,t,r,b] lists) into y_true/y_pred
        lists suitable for the *in-memory* HBB detector evaluator.

        Output formats (per image):
          y_true[i] = [{"label": str, "bbox": [l,t,r,b], "difficult": bool?}, ...]
          y_pred[i] = [{"label": str, "bbox": [l,t,r,b], "confidence": float}, ...]
        """
        y_true: List[List[Dict[str, Any]]] = []
        y_pred: List[List[Dict[str, Any]]] = []

        for s in self.data:
            label = s.extracted_class() if iou_label_from_question else None
            if not label:
                label = default_label
            if normalize_label:
                label = _clean_label(label)

            gt_boxes = _coerce_list(s.gt)
            pred_boxes = _coerce_list(s.answer)

            # GT
            gt_list: List[Dict[str, Any]] = []
            for b in gt_boxes:
                bbox, difficult = _parse_gt_hbb_box(b)
                if bbox is None:
                    continue
                item = {"label": label, "bbox": bbox}
                if difficult is not None:
                    item["difficult"] = difficult
                gt_list.append(item)

            # Pred
            pred_list: List[Dict[str, Any]] = []
            for b in pred_boxes:
                bbox, conf = _parse_pred_hbb_box(b)
                if bbox is None:
                    continue
                pred_list.append({"label": label, "bbox": bbox, "confidence": conf})

            y_true.append(gt_list)
            y_pred.append(pred_list)

        return y_true, y_pred


# ----------------------------
# Loading / parsing
# ----------------------------

def load_eval_file(path: Union[str, Path]) -> EvalFile:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise TypeError(f"Top-level JSON must be an object, got {type(raw).__name__}")

    info_obj = raw.get("info")
    if not isinstance(info_obj, dict):
        raise TypeError("Top-level field 'info' must be an object")

    info = EvalInfo(
        task=str(info_obj.get("task") or ""),
        model=_none_if_empty(info_obj.get("model")),
        dataset=_none_if_empty(info_obj.get("dataset")),
        json_path=str(path),
    )
    if not info.task:
        raise ValueError("info.task is required and must be a non-empty string")

    data_obj = raw.get("data")
    if isinstance(data_obj, dict):
        # Normalize dict-of-samples into list-of-samples.
        normalized: List[Dict[str, Any]] = []
        for k, v in data_obj.items():
            if not isinstance(v, dict):
                raise TypeError(f"data['{k}'] must be an object")
            vv = dict(v)
            vv.setdefault("id", str(k))
            normalized.append(vv)
        data_list = normalized
    elif isinstance(data_obj, list):
        data_list = data_obj
    else:
        raise TypeError("Top-level field 'data' must be a list or an object keyed by id")

    samples: List[EvalSample] = []
    for idx, entry in enumerate(data_list):
        if not isinstance(entry, dict):
            raise TypeError(f"data[{idx}] must be an object")

        sid = _none_if_empty(entry.get("id"))

        image_path, image2_path = _parse_image_field(entry.get("image"))

        crop = _parse_crop(entry.get("crop"))

        question = _none_if_empty(entry.get("question"))

        # Prediction and GT
        if "answer" not in entry:
            raise KeyError(f"data[{idx}] missing required field 'answer'")
        if "gt" not in entry:
            raise KeyError(f"data[{idx}] missing required field 'gt'")

        answer = entry.get("answer")
        gt = entry.get("gt")

        meta = entry.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {"meta": meta}

        samples.append(
            EvalSample(
                id=sid,
                image_path=image_path,
                image2_path=image2_path,
                crop=crop,
                question=question,
                answer=answer,
                gt=gt,
                meta=meta,
            )
        )

    return EvalFile(info=info, data=tuple(samples))


def load_many(paths: Sequence[Union[str, Path]]) -> List[EvalFile]:
    return [load_eval_file(p) for p in paths]


# ----------------------------
# Helpers
# ----------------------------

def _none_if_empty(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _parse_image_field(v: Any) -> Tuple[str, Optional[str]]:
    """
    image can be:
      - "path"
      - ["path"]
      - ["path1", "path2"]
    """
    if isinstance(v, str):
        return v, None
    if isinstance(v, list) and all(isinstance(x, str) for x in v):
        if len(v) == 0:
            raise ValueError("image field cannot be an empty list")
        if len(v) == 1:
            return v[0], None
        if len(v) >= 2:
            return v[0], v[1]
    raise TypeError("data[].image must be a string or a list of strings")


def _parse_crop(v: Any) -> Optional[Tuple[Number, Number, Number, Number]]:
    if v is None:
        return None
    if isinstance(v, list) and len(v) == 0:
        return None
    if isinstance(v, (list, tuple)) and len(v) == 4:
        try:
            return (float(v[0]), float(v[1]), float(v[2]), float(v[3]))
        except Exception as e:
            raise TypeError(f"crop must contain 4 numbers, got {v!r}") from e
    # Some files may store crop as [] or omitted; treat other types as None
    return None


def _coerce_list(v: Any) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    # Some tasks store a single item as a scalar; treat as a single-element list.
    return [v]


def _clean_label(s: str) -> str:
    s = (s or "").strip()
    # remove trailing punctuation commonly produced by generative models
    s = re.sub(r"[ \t\r\n]+", " ", s)
    s = s.strip().strip(".").strip()
    return s


def _parse_gt_hbb_box(v: Any) -> Tuple[Optional[List[Number]], Optional[bool]]:
    """
    Supported GT box formats:
      - ["l","t","r","b"]
      - [l,t,r,b]
      - {"bbox":[l,t,r,b], "difficult": true}
      - {"left":..,"top":..,"right":..,"bottom":.., "difficult": ...}
    Returns: (bbox or None, difficult or None)
    """
    difficult: Optional[bool] = None

    if isinstance(v, dict):
        if "difficult" in v:
            difficult = bool(v["difficult"])
        if "bbox" in v and isinstance(v["bbox"], (list, tuple)) and len(v["bbox"]) == 4:
            b = v["bbox"]
            return [float(b[0]), float(b[1]), float(b[2]), float(b[3])], difficult
        if all(k in v for k in ("left", "top", "right", "bottom")):
            return [float(v["left"]), float(v["top"]), float(v["right"]), float(v["bottom"])], difficult

    if isinstance(v, (list, tuple)) and len(v) == 4:
        try:
            return [float(v[0]), float(v[1]), float(v[2]), float(v[3])], difficult
        except Exception:
            return None, difficult

    return None, difficult


def _parse_pred_hbb_box(v: Any) -> Tuple[Optional[List[Number]], Number]:
    """
    Supported pred box formats:
      - ["l","t","r","b"]                 -> conf defaults to 1.0
      - [l,t,r,b]                         -> conf defaults to 1.0
      - ["score","l","t","r","b"]         -> score parsed
      - ["label","score","l","t","r","b"] -> score parsed (label ignored here)
      - {"bbox":[l,t,r,b], "confidence":0.9} (or "score")
    Returns: (bbox or None, confidence)
    """
    conf: Number = 1.0

    if isinstance(v, dict):
        if "confidence" in v:
            try:
                conf = float(v["confidence"])
            except Exception:
                conf = 1.0
        elif "score" in v:
            try:
                conf = float(v["score"])
            except Exception:
                conf = 1.0

        if "bbox" in v and isinstance(v["bbox"], (list, tuple)) and len(v["bbox"]) == 4:
            b = v["bbox"]
            return [float(b[0]), float(b[1]), float(b[2]), float(b[3])], conf

        if all(k in v for k in ("left", "top", "right", "bottom")):
            return [float(v["left"]), float(v["top"]), float(v["right"]), float(v["bottom"])], conf

    if isinstance(v, (list, tuple)):
        if len(v) == 4:
            try:
                return [float(v[0]), float(v[1]), float(v[2]), float(v[3])], conf
            except Exception:
                return None, conf
        if len(v) == 5:
            # [score, l, t, r, b]
            try:
                conf = float(v[0])
            except Exception:
                conf = 1.0
            try:
                return [float(v[1]), float(v[2]), float(v[3]), float(v[4])], conf
            except Exception:
                return None, conf
        if len(v) >= 6:
            # [label, score, l, t, r, b] (or longer)
            try:
                conf = float(v[1])
            except Exception:
                conf = 1.0
            try:
                return [float(v[2]), float(v[3]), float(v[4]), float(v[5])], conf
            except Exception:
                return None, conf

    return None, conf


# ----------------------------
# CLI
# ----------------------------

def _cli() -> int:
    ap = argparse.ArgumentParser(description="Read combined result+gt JSON files.")
    ap.add_argument("json_path", type=str, help="Path to combined JSON")
    ap.add_argument("--summary", action="store_true", help="Print a short summary")
    ap.add_argument("--print-first", action="store_true", help="Print the first sample (debug)")
    args = ap.parse_args()

    ef = load_eval_file(args.json_path)

    if args.summary:
        n = len(ef)
        print(json.dumps({
            "task": ef.task,
            "model": ef.info.model,
            "dataset": ef.info.dataset,
            "json_path": ef.info.json_path,
            "num_samples": n,
            "has_crop": sum(1 for s in ef.data if s.crop is not None),
            "has_image2": sum(1 for s in ef.data if s.image2_path is not None),
        }, indent=2))

    if args.print_first and len(ef) > 0:
        s = ef.data[0]
        print(json.dumps({
            "id": s.id,
            "image_path": s.image_path,
            "image2_path": s.image2_path,
            "crop": s.crop,
            "question": s.question,
            "answer_type": type(s.answer).__name__,
            "gt_type": type(s.gt).__name__,
            "extracted_class": s.extracted_class(),
        }, indent=2))

    if not args.summary and not args.print_first:
        # default behavior
        print(f"Loaded {len(ef)} samples for task={ef.task}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
