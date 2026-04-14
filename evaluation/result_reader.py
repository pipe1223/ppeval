from __future__ import annotations

"""
result_reader.py

Reader for model *result-only* JSON files.

Supports both:
  A) OLD format:
     {"info": {...}, "data": [ {"task_config": {...}, "results": {...}}, ... ]}

  B) NEW format (recommended):
     {"info": {...}, "data": { "<image_path>": {"task_config": {...}, "results": {...}}, ... } }

Key features:
- Tolerant parsing (info.json_path may be null, dataset may be null)
- `data` may be list OR dict (keyed by image path)
- Optional `by_image` index for fast lookup
- Predictions parsed into:
    - DetectionPrediction (dict with coord+label)
    - TextPrediction (string)
    - RawPrediction (fallback)

This module is dependency-free (stdlib only).
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
class Info:
    model: str
    dataset: Optional[str]
    json_path: Optional[str]
    task: Optional[str]
    extra: Mapping[str, Any]


@dataclass(frozen=True)
class TaskConfig:
    image_path: str
    image2_path: Optional[str]
    task_type: str
    extra: Mapping[str, Any]


@dataclass(frozen=True)
class DetectionPrediction:
    coord: Tuple[Number, ...]
    label: str
    confidence: Optional[Number]
    extra: Mapping[str, Any]

    def points(self) -> List[Tuple[Number, Number]]:
        if len(self.coord) % 2 != 0:
            raise ValueError(f"coord length must be even, got {len(self.coord)}")
        return [(self.coord[i], self.coord[i + 1]) for i in range(0, len(self.coord), 2)]


@dataclass(frozen=True)
class TextPrediction:
    text: str


@dataclass(frozen=True)
class RawPrediction:
    data: Any


Prediction = Union[DetectionPrediction, TextPrediction, RawPrediction]


@dataclass(frozen=True)
class ResultSet:
    prediction: Tuple[Prediction, ...]
    meta: Mapping[str, Any]


@dataclass(frozen=True)
class TaskEntry:
    """
    One entry corresponds to one image (or image pair).
    """
    key: str  # usually image_path; from dict key when present, else derived from task_config.image_path
    task_config: TaskConfig
    results: Mapping[str, ResultSet]
    extra: Mapping[str, Any]

    def available_result_keys(self) -> List[str]:
        return sorted(self.results.keys(), key=lambda k: (k != "merged_results", k))

    def get_result(self, key: str = "merged_results") -> ResultSet:
        if key in self.results:
            return self.results[key]
        # fallback: merged_results > first available
        if "merged_results" in self.results:
            return self.results["merged_results"]
        if self.results:
            return self.results[next(iter(self.results))]
        raise KeyError("No results available for this entry")

    def iter_predictions(self, key: str = "merged_results") -> Iterable[Prediction]:
        return iter(self.get_result(key).prediction)


@dataclass(frozen=True)
class ResultsFile:
    info: Info
    data: Tuple[TaskEntry, ...]
    by_image: Mapping[str, TaskEntry]

    def __len__(self) -> int:
        return len(self.data)

    def iter_entries(self) -> Iterable[TaskEntry]:
        return iter(self.data)

    def all_result_keys(self) -> List[str]:
        keys = set()
        for e in self.data:
            keys.update(e.results.keys())
        return sorted(keys, key=lambda k: (k != "merged_results", k))

    def task_types(self) -> List[str]:
        return sorted({e.task_config.task_type for e in self.data})

    def primary_task_type(self) -> Optional[str]:
        types = self.task_types()
        return types[0] if len(types) == 1 else None

    def get_entry(self, image_path: str) -> TaskEntry:
        """
        Lookup by exact key; if not found, fallback to basename match.
        """
        if image_path in self.by_image:
            return self.by_image[image_path]
        base = Path(image_path).name
        for k, v in self.by_image.items():
            if Path(k).name == base:
                return v
        raise KeyError(f"No entry for image_path={image_path!r}")


# ----------------------------
# Parsing helpers
# ----------------------------

def _as_optional_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    # be permissive: allow numbers / identifiers as strings
    return str(v)


def _as_str(v: Any, field: str) -> str:
    if isinstance(v, str):
        return v
    if v is None:
        raise TypeError(f"Field '{field}' expected str, got null")
    return str(v)


def _as_coord_tuple(v: Any) -> Tuple[Number, ...]:
    if not isinstance(v, list):
        raise TypeError(f"Field 'coord' expected list, got {type(v).__name__}")
    coords: List[Number] = []
    for i, item in enumerate(v):
        if not isinstance(item, (int, float)):
            raise TypeError(f"coord[{i}] expected number, got {type(item).__name__}")
        coords.append(float(item))
    if len(coords) % 2 != 0:
        raise ValueError(f"coord length must be even, got {len(coords)}")
    return tuple(coords)


def _parse_detection_prediction(obj: Mapping[str, Any]) -> DetectionPrediction:
    coord = _as_coord_tuple(obj.get("coord"))
    label = _as_str(obj.get("label"), "label")
    conf = obj.get("confidence")
    confidence: Optional[Number]
    if conf is None:
        confidence = None
    elif isinstance(conf, (int, float)):
        confidence = float(conf)
    else:
        confidence = None
    extra = {k: v for k, v in obj.items() if k not in {"coord", "label", "confidence"}}
    return DetectionPrediction(coord=coord, label=label, confidence=confidence, extra=extra)


def _parse_prediction_item(item: Any) -> Prediction:
    if isinstance(item, str):
        return TextPrediction(text=item)
    if isinstance(item, dict):
        if "coord" in item and "label" in item:
            return _parse_detection_prediction(item)
        return RawPrediction(data=item)
    return RawPrediction(data=item)


def _parse_prediction_list(v: Any) -> Tuple[Prediction, ...]:
    # allow scalar strings
    if isinstance(v, str):
        return (TextPrediction(text=v),)
    if not isinstance(v, list):
        raise TypeError(f"Field 'prediction' expected list (or str), got {type(v).__name__}")
    return tuple(_parse_prediction_item(x) for x in v)


def _parse_result_set(obj: Mapping[str, Any]) -> ResultSet:
    if "prediction" not in obj:
        raise KeyError("Result object missing required field 'prediction'")
    preds = _parse_prediction_list(obj.get("prediction"))
    meta = {k: v for k, v in obj.items() if k != "prediction"}
    return ResultSet(prediction=preds, meta=meta)


def _parse_task_config(obj: Mapping[str, Any]) -> TaskConfig:
    image2 = obj.get("image2_path", obj.get("image2"))
    extra = {k: v for k, v in obj.items() if k not in {"image_path", "image2_path", "image2", "task_type"}}
    return TaskConfig(
        image_path=_as_str(obj.get("image_path"), "image_path"),
        image2_path=_as_optional_str(image2),
        task_type=_as_str(obj.get("task_type"), "task_type"),
        extra=extra,
    )


def _parse_info(obj: Mapping[str, Any]) -> Info:
    known = {"model", "dataset", "json_path", "task"}
    extra = {k: v for k, v in obj.items() if k not in known}
    return Info(
        model=_as_str(obj.get("model"), "model"),
        dataset=_as_optional_str(obj.get("dataset")),
        json_path=_as_optional_str(obj.get("json_path")),
        task=_as_optional_str(obj.get("task")),
        extra=extra,
    )


def parse_results_dict(results_obj: Mapping[str, Any]) -> Dict[str, ResultSet]:
    parsed: Dict[str, ResultSet] = {}
    for key, value in results_obj.items():
        if not isinstance(value, dict):
            raise TypeError(f"results['{key}'] expected object, got {type(value).__name__}")
        parsed[str(key)] = _parse_result_set(value)
    return parsed


# ----------------------------
# Public API
# ----------------------------

def load_results(path: str | Path) -> ResultsFile:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise TypeError(f"Top-level JSON must be an object, got {type(raw).__name__}")

    info_obj = raw.get("info")
    if not isinstance(info_obj, dict):
        raise TypeError("Top-level field 'info' must be an object")
    info = _parse_info(info_obj)

    data_obj = raw.get("data")
    if isinstance(data_obj, dict):
        items = list(data_obj.items())
    elif isinstance(data_obj, list):
        items = [(None, x) for x in data_obj]
    else:
        raise TypeError("Top-level field 'data' must be a list or an object keyed by image path")

    entries: List[TaskEntry] = []
    by_image: Dict[str, TaskEntry] = {}

    for idx, (maybe_key, entry) in enumerate(items):
        if not isinstance(entry, dict):
            raise TypeError(f"data[{idx}] must be an object")

        tc_obj = entry.get("task_config")
        if not isinstance(tc_obj, dict):
            raise TypeError(f"data[{idx}].task_config must be an object")
        task_config = _parse_task_config(tc_obj)

        res_obj = entry.get("results")
        if not isinstance(res_obj, dict):
            raise TypeError(f"data[{idx}].results must be an object")
        results = parse_results_dict(res_obj)

        # compute key: dict key > task_config.image_path
        key = str(maybe_key) if isinstance(maybe_key, str) and maybe_key else task_config.image_path

        extra = {k: v for k, v in entry.items() if k not in {"task_config", "results"}}
        te = TaskEntry(key=key, task_config=task_config, results=results, extra=extra)

        entries.append(te)
        by_image[key] = te

    # Deterministic order: sort by key
    entries_sorted = tuple(sorted(entries, key=lambda e: e.key))
    by_image_sorted = {e.key: e for e in entries_sorted}

    return ResultsFile(info=info, data=entries_sorted, by_image=by_image_sorted)


def load_many(paths: Sequence[str | Path]) -> List[ResultsFile]:
    return [load_results(p) for p in paths]


def summarize(results_file: ResultsFile, key: str = "merged_results") -> Dict[str, Any]:
    n_entries = len(results_file)
    available_keys = results_file.all_result_keys()
    counts: List[int] = []

    for e in results_file.data:
        k = key if key in e.results else ("merged_results" if "merged_results" in e.results else (available_keys[0] if available_keys else key))
        counts.append(len(e.results[k].prediction) if k in e.results else 0)

    return {
        "model": results_file.info.model,
        "dataset": results_file.info.dataset,
        "json_path": results_file.info.json_path,
        "task": results_file.info.task,
        "task_types": results_file.task_types(),
        "num_entries": n_entries,
        "all_result_keys": available_keys,
        "total_predictions_for_key": sum(counts),
    }


def flatten_predictions(results_file: ResultsFile, key: str = "merged_results") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry in results_file.data:
        if key not in entry.results:
            continue
        rs = entry.results[key]
        for pred in rs.prediction:
            base = {
                "key": entry.key,
                "image_path": entry.task_config.image_path,
                "image2_path": entry.task_config.image2_path,
                "task_type": entry.task_config.task_type,
                "result_key": key,
                "meta": dict(rs.meta),
            }
            if isinstance(pred, DetectionPrediction):
                rows.append(
                    {
                        **base,
                        "pred_kind": "detection",
                        "label": pred.label,
                        "confidence": pred.confidence,
                        "coord": list(pred.coord),
                        "extra": dict(pred.extra),
                    }
                )
            elif isinstance(pred, TextPrediction):
                rows.append({**base, "pred_kind": "text", "text": pred.text})
            else:
                rows.append({**base, "pred_kind": "raw", "raw": pred.data})
    return rows


# ----------------------------
# CLI
# ----------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Read and inspect results JSON.")
    p.add_argument("json_file", type=str, help="Path to a JSON results file.")
    p.add_argument("--result-key", type=str, default="merged_results", help="Results key to inspect.")
    p.add_argument("--print-keys", action="store_true", help="Print all available result keys found in the file.")
    p.add_argument("--summary", action="store_true", help="Print a summary.")
    p.add_argument("--list", action="store_true", help="List entries with prediction counts for the chosen key.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    rf = load_results(args.json_file)

    if args.print_keys:
        print("Result keys:", rf.all_result_keys())

    if args.summary:
        print(json.dumps(summarize(rf, key=args.result_key), indent=2))

    if args.list:
        for i, entry in enumerate(rf.data):
            rs = entry.get_result(args.result_key)
            print(f"[{i}] {entry.key} | task_type={entry.task_config.task_type} | n_pred({args.result_key})={len(rs.prediction)}")

    if not (args.print_keys or args.summary or args.list):
        print(f"Loaded {len(rf)} entries from {args.json_file}. Keys: {rf.all_result_keys()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
