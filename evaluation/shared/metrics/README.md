# Shared metrics

This directory is reserved for metrics that can be reused across more than one evaluation family.

Examples of good candidates:
- ranking metrics used by retrieval and grounding
- generic precision / recall / F1 helpers
- aggregation helpers for macro / micro reporting

Examples that should stay elsewhere:
- box IoU specific to visual detection
- mask metrics for segmentation
- graph-only structure metrics
