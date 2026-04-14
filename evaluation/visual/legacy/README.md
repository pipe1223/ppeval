# Visual legacy

This directory is a placeholder for the current visual evaluation implementation during migration.

## Why this exists

The repository's active visual evaluator still mostly lives in the top-level `evaluation/` directory.
This placeholder makes the intended migration target visible without moving working code too early.

## Future migration candidates

Likely files to move here first:
- `evaluation_combined.py`
- `evaluation_separated.py`
- `result_gt_reader.py`
- `result_reader.py`
- `gt_reader.py`
- `eval_core/`

That move should happen only after import paths and docs are updated so the current workflow does not break.
