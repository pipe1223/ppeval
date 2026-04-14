"""Compatibility wrapper for the moved legacy visual evaluator."""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.visual.legacy.gt_reader import *  # noqa: F401,F403

if __name__ == "__main__":
    from evaluation.visual.legacy.gt_reader import main
    raise SystemExit(main())
