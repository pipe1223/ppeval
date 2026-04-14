from __future__ import annotations

"""
gt_reader_v2.py

Ground-truth reader for JSON files keyed by image path, with optional multi-task support.

Supported schemas:

1) Dict-keyed (recommended):
{
  "info": {...},
  "data": {
    "image/Image1.tif": {
      "image_path": "image/Image1.tif",
      "image2_path": null,          # or "image2": null
      "gt": [ {...}, {...} ]        # list of GT items
    },
    ...
  }
}

2) List-based:
{
  "info": {...},
  "data": [
    { "image_path": "...", "image2_path": null, "gt": [...] },
    ...
  ]
}
(Will be normalized into a dict keyed by image_path.)

Multi-task GT in one file:
- You may add "task_type" inside each object in gt[]:
    { "label": [...], "question": "...", "task_type": "IMG_CLS" }
- Then you can call `GTFile.for_task("IMG_CLS")` or `GTFile.split_by_task()`.

Notes:
- Some template files may contain bare identifiers (e.g. dataset: GROUND_TRUTH).
  This reader includes a *relaxed JSON loader* that auto-quotes such tokens.
- GT items are parsed into a lightweight GTItem dataclass; unknown fields go into `extra`.
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
class GTItem:
    """
    One GT record inside an entry's gt[] list.

    Common fields:
    - label: class label(s) / answer / caption (varies by task)
    - coord: geometry (HBB/OBB/polygon/etc.) if present and numeric list
    - question: optional prompt string
    - task_type: optional per-item task tag (enables mixed-task GT files)
    - crop: optional [x1,y1,x2,y2] crop
    - extra: everything else
    """
    label: Any
    coord: Optional[Tuple[Number, ...]]
    question: Optional[str]
    task_type: Optional[str]
    crop: Optional[Tuple[Number, Number, Number, Number]]
    extra: Mapping[str, Any]


@dataclass(frozen=True)
class GTEntry:
    key: str  # dict key (usually image_path)
    image_path: str
    image2_path: Optional[str]
    gt: Tuple[GTItem, ...]
    extra: Mapping[str, Any]

    def __len__(self) -> int:
        return len(self.gt)

    def iter_gt(self) -> Iterable[GTItem]:
        return iter(self.gt)

    def primary_label(self) -> Any:
        return self.gt[0].label if self.gt else None

    def task_types(self) -> List[str]:
        ts = sorted({it.task_type for it in self.gt if it.task_type})
        return ts


@dataclass(frozen=True)
class GTFile:
    info: Info
    by_image: Mapping[str, GTEntry]

    def __len__(self) -> int:
        return len(self.by_image)

    def keys(self) -> List[str]:
        return sorted(self.by_image.keys())

    def get_entry(self, image_path: str) -> GTEntry:
        # Exact match
        if image_path in self.by_image:
            return self.by_image[image_path]

        # Fallback: basename match
        base = Path(image_path).name
        for k, v in self.by_image.items():
            if Path(k).name == base:
                return v
        raise KeyError(f"No GT entry for image_path={image_path!r}")

    def iter_entries(self) -> Iterable[GTEntry]:
        for k in self.keys():
            yield self.by_image[k]

    def task_types(self) -> List[str]:
        """
        Return sorted unique task_type values found in GTItem.task_type.
        If none are present, fall back to info.task (if provided).
        """
        ts = set()
        for e in self.by_image.values():
            for it in e.gt:
                if it.task_type:
                    ts.add(it.task_type)
        if ts:
            return sorted(ts)
        if self.info.task:
            return [self.info.task]
        return []

    def for_task(self, task_type: str) -> "GTFile":
        """
        Return a filtered GTFile view containing only GTItems for `task_type`.
        If no per-item task_type exists, this will return self only if info.task matches.
        """
        task_type = str(task_type)
        has_item_task = any(
            (it.task_type is not None)
            for e in self.by_image.values()
            for it in e.gt
        )

        if not has_item_task:
            # single-task file; accept only if task matches (or task unknown)
            if self.info.task and self.info.task != task_type:
                return GTFile(info=self.info, by_image={})
            return self

        filtered: Dict[str, GTEntry] = {}
        for k, e in self.by_image.items():
            keep_items = tuple(it for it in e.gt if it.task_type == task_type)
            if keep_items:
                filtered[k] = GTEntry(
                    key=e.key,
                    image_path=e.image_path,
                    image2_path=e.image2_path,
                    gt=keep_items,
                    extra=e.extra,
                )

        info = Info(
            model=self.info.model,
            dataset=self.info.dataset,
            json_path=self.info.json_path,
            task=task_type,
            extra=self.info.extra,
        )
        return GTFile(info=info, by_image=filtered)

    def split_by_task(self) -> Dict[str, "GTFile"]:
        """
        Split a mixed-task GT file into {task_type -> GTFile}.
        For single-task files, returns {info.task or "UNKNOWN": self}.
        """
        ts = self.task_types()
        if not ts:
            return {"UNKNOWN": self}
        return {t: self.for_task(t) for t in ts}


# ----------------------------
# Relaxed JSON loader
# ----------------------------

_BAREWORD_RE = re.compile(r':\s*([A-Za-z_][A-Za-z0-9_]*)\s*([,\n\r}])')

def load_json_relaxed(path: str | Path) -> Any:
    """
    Load JSON with a small relaxation:
      - if it sees `: IDENT,` where IDENT is a bareword token (not true/false/null),
        it will auto-quote it: `: "IDENT",`

    This mainly supports template-like files such as:
      "dataset": GROUND_TRUTH,
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        def repl(m: re.Match[str]) -> str:
            ident = m.group(1)
            tail = m.group(2)
            if ident in {"true", "false", "null"}:
                return f": {ident}{tail}"
            return f': "{ident}"{tail}'
        fixed = _BAREWORD_RE.sub(repl, text)
        return json.loads(fixed)


# ----------------------------
# Parsing helpers
# ----------------------------

def _as_optional_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    return str(v)


def _as_str(v: Any, field: str) -> str:
    if isinstance(v, str):
        return v
    if v is None:
        raise TypeError(f"Field '{field}' expected str, got null")
    return str(v)


def _as_number(v: Any, field: str) -> Number:
    if isinstance(v, (int, float)):
        return float(v)
    raise TypeError(f"Field '{field}' expected number, got {type(v).__name__}")


def _as_coord_tuple(v: Any) -> Optional[Tuple[Number, ...]]:
    if v is None:
        return None
    if not isinstance(v, list):
        return None
    coords: List[Number] = []
    for i, item in enumerate(v):
        if not isinstance(item, (int, float)):
            return None
        coords.append(float(item))
    if len(coords) % 2 != 0:
        return None
    return tuple(coords)


def _as_crop(v: Any) -> Optional[Tuple[Number, Number, Number, Number]]:
    if v is None:
        return None
    if isinstance(v, list) and len(v) == 0:
        return None
    if isinstance(v, (list, tuple)) and len(v) == 4 and all(isinstance(x, (int, float)) for x in v):
        return (float(v[0]), float(v[1]), float(v[2]), float(v[3]))
    return None


def _parse_info(obj: Mapping[str, Any]) -> Info:
    model = _as_str(obj.get("model", "GROUND_TRUTH"), "model")
    dataset = _as_optional_str(obj.get("dataset"))
    json_path = _as_optional_str(obj.get("json_path"))
    task = _as_optional_str(obj.get("task")) or _as_optional_str(obj.get("task_type"))
    extra = {k: v for k, v in obj.items() if k not in {"model", "dataset", "json_path", "task", "task_type"}}
    return Info(model=model, dataset=dataset, json_path=json_path, task=task, extra=extra)


def _parse_gt_item(obj: Mapping[str, Any]) -> GTItem:
    label = obj.get("label")
    coord = _as_coord_tuple(obj.get("coord"))
    question = _as_optional_str(obj.get("question"))
    task_type = _as_optional_str(obj.get("task_type")) or _as_optional_str(obj.get("task"))
    crop = _as_crop(obj.get("crop"))
    extra = {k: v for k, v in obj.items() if k not in {"label", "coord", "question", "task_type", "task", "crop"}}
    # Preserve raw coord if it's present but not numeric
    if "coord" in obj and coord is None:
        extra["coord_raw"] = obj.get("coord")
    return GTItem(label=label, coord=coord, question=question, task_type=task_type, crop=crop, extra=extra)


def _parse_entry(key: str, obj: Mapping[str, Any]) -> GTEntry:
    image_path = _as_str(obj.get("image_path") or obj.get("image") or key, "image_path")
    image2_path = _as_optional_str(obj.get("image2_path")) or _as_optional_str(obj.get("image2"))
    gt_list = obj.get("gt")
    if not isinstance(gt_list, list):
        raise TypeError(f"Entry '{key}'.gt expected list, got {type(gt_list).__name__}")

    items = tuple(_parse_gt_item(x) for x in gt_list if isinstance(x, dict))
    extra = {k: v for k, v in obj.items() if k not in {"image_path", "image", "image2_path", "image2", "gt"}}
    return GTEntry(key=key, image_path=image_path, image2_path=image2_path, gt=items, extra=extra)


def load_gt(path: str | Path) -> GTFile:
    path = Path(path)
    raw = load_json_relaxed(path)

    if not isinstance(raw, dict):
        raise TypeError(f"Top-level JSON must be an object, got {type(raw).__name__}")

    info_obj = raw.get("info", {})
    if not isinstance(info_obj, dict):
        raise TypeError("Top-level field 'info' must be an object")
    info = _parse_info(info_obj)

    data_obj = raw.get("data")
    if isinstance(data_obj, dict):
        entries_obj = data_obj
    elif isinstance(data_obj, list):
        # normalize list into dict keyed by image_path
        entries_obj = {}
        for idx, e in enumerate(data_obj):
            if not isinstance(e, dict):
                continue
            k = _as_optional_str(e.get("image_path")) or _as_optional_str(e.get("image")) or f"idx:{idx}"
            entries_obj[k] = e
    else:
        raise TypeError("Top-level field 'data' must be a dict or list")

    by_image: Dict[str, GTEntry] = {}
    for k, v in entries_obj.items():
        if not isinstance(v, dict):
            continue
        key = str(k)
        by_image[key] = _parse_entry(key, v)

    return GTFile(info=info, by_image=by_image)


def summarize(gf: GTFile) -> Dict[str, Any]:
    n = len(gf)
    n_items = sum(len(e.gt) for e in gf.by_image.values())
    task_types = gf.task_types()
    return {
        "model": gf.info.model,
        "dataset": gf.info.dataset,
        "json_path": gf.info.json_path,
        "info_task": gf.info.task,
        "task_types_in_items": task_types,
        "num_entries": n,
        "num_gt_items": n_items,
    }


# ----------------------------
# CLI
# ----------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Read and inspect GT JSON keyed by image path.")
    p.add_argument("json_file", type=str, help="Path to a GT JSON file.")
    p.add_argument("--summary", action="store_true", help="Print a summary.")
    p.add_argument("--print-keys", action="store_true", help="Print first 20 keys.")
    p.add_argument("--print-first", action="store_true", help="Print first entry details.")
    p.add_argument("--split", action="store_true", help="Print split counts by task_type (if mixed-task file).")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    gf = load_gt(args.json_file)

    if args.summary:
        print(json.dumps(summarize(gf), indent=2))

    if args.split:
        split = gf.split_by_task()
        print(json.dumps({k: len(v) for k, v in split.items()}, indent=2))

    if args.print_keys:
        keys = gf.keys()
        print("Keys (first 20):", keys[:20])

    if args.print_first:
        first_key = gf.keys()[0] if gf.keys() else None
        if first_key:
            e = gf.by_image[first_key]
            print(json.dumps({
                "key": e.key,
                "image_path": e.image_path,
                "image2_path": e.image2_path,
                "n_gt": len(e.gt),
                "first_gt": {
                    "label": e.gt[0].label if e.gt else None,
                    "coord": list(e.gt[0].coord) if (e.gt and e.gt[0].coord) else None,
                    "question": e.gt[0].question if e.gt else None,
                    "task_type": e.gt[0].task_type if e.gt else None,
                    "crop": list(e.gt[0].crop) if (e.gt and e.gt[0].crop) else None,
                    "extra": dict(e.gt[0].extra) if e.gt else None,
                } if e.gt else None
            }, indent=2))

    if not (args.summary or args.print_keys or args.print_first or args.split):
        print(f"Loaded {len(gf)} GT entries from {args.json_file}. Task types: {gf.task_types()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
