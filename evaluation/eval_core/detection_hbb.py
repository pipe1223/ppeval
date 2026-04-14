"""Compatibility wrapper for the moved legacy visual evaluator."""
from __future__ import annotations

import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_ROOT = str(_THIS.parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluation.visual.legacy.eval_core.detection_hbb import *  # noqa: F401,F403
