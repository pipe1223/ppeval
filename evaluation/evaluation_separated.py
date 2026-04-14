"""Compatibility wrapper for the moved legacy visual evaluator."""
from evaluation.visual.legacy.evaluation_separated import *  # noqa: F401,F403

if __name__ == "__main__":
    from evaluation.visual.legacy.evaluation_separated import main
    raise SystemExit(main())
