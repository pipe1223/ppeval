# Visual evaluation

This directory is the target home for visual and vision-language evaluation.

Planned scope:
- classification
- captioning
- VQA
- HBB / OBB detection
- segmentation
- change detection
- grounding

## Current state

The repository's working visual evaluator still lives mostly in the top-level `evaluation/` directory.

That is intentional for now.

The current reorganization is focused on adding structure first, then moving code gradually once the new layout is stable.

## Migration idea

A future migration can move code into subareas like:

```text
visual/
├── classification/
├── captioning/
├── vqa/
├── detection/
├── segmentation/
└── grounding/
```

## What should stay visual-specific

Examples of code that should remain in `visual/` instead of `shared/`:
- IoU implementations for boxes or masks
- AP / mAP logic for object detection
- Dice / IoU / VC metrics for segmentation
- image-region conversion helpers
- visual grounding evaluation

## Why this split matters

Visual evaluation and retrieval evaluation may share some reporting logic or ranking utilities, but they are still different enough that a dedicated visual area keeps the code easier to maintain.
