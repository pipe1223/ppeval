# Shared evaluation utilities

This directory is for code that can be reused by both visual and retrieval evaluation.

Typical examples:
- ranking metrics
- aggregation helpers
- JSON schemas
- report formatting
- file readers that are not tied to one task family

## Design rule

A utility belongs in `shared/` only if it is useful across more than one evaluation target.

Examples:
- `precision_at_k` can be used by retrieval and some grounding tasks
- simple report aggregation can be shared across many tasks

Examples that should **not** live here:
- detection IoU code that is specific to visual evaluation
- segmentation mask metrics
- graph-only path metrics

## Migration note

The existing visual evaluator still contains some logic that could later be moved here. For now, this folder is intentionally lightweight so the current working code does not break during reorganization.
