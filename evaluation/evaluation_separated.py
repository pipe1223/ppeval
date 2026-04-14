"""Compatibility wrapper for the moved legacy visual evaluator."""
from __future__ import annotations

import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_ROOT = str(_THIS.parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluation.visual.legacy.evaluation_separated import *  # noqa: F401,F403

if __name__ == "__main__":
    from evaluation.visual.legacy.evaluation_separated import main
    raise SystemExit(main())
