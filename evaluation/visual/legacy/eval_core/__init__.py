"""Legacy visual metric core with lazy imports."""

from __future__ import annotations

import importlib

from . import const

_EXPORT_MAP = {
    "evaluate_captioning": ("captioning", "evaluate_captioning"),
    "evaluate_classification": ("classification", "evaluate_classification"),
    "evaluate_detection_from_dir": ("detection", "evaluate_detection"),
    "evaluate_detection": ("detection_hbb", "evaluate_detection"),
    "evaluate_detection2": ("detection_obb", "evaluate_detection2"),
    "eva_disaster": ("disaster", "eva_disaster"),
    "evaluate_segmentation": ("segmentation", "evaluate_segmentation"),
    "evaluate_vqa": ("vqa", "evaluate_vqa"),
}

__all__ = ["const", *_EXPORT_MAP.keys()]


def __getattr__(name: str):
    if name == "const":
        return const
    if name in _EXPORT_MAP:
        module_name, attr_name = _EXPORT_MAP[name]
        module = importlib.import_module(f"{__name__}.{module_name}")
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
