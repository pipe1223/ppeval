"""Compatibility wrapper for the moved legacy visual evaluator."""
from evaluation.visual.legacy.evaluation_combined import *  # noqa: F401,F403

if __name__ == "__main__":
    from evaluation.visual.legacy.evaluation_combined import main
    raise SystemExit(main())
