"""Compatibility wrapper for the moved legacy visual evaluator."""
from evaluation.visual.legacy.result_gt_reader import *  # noqa: F401,F403

if __name__ == "__main__":
    from evaluation.visual.legacy.result_gt_reader import _cli
    raise SystemExit(_cli())
