# ppeval

`ppeval` is a growing evaluation repository for visual models, vision-language models, retrieval systems, RAG, and GraphRAG.

The repo is being organized around a **single evaluation framework** with separate subareas for different evaluation targets:

- **visual evaluation** for classification, captioning, VQA, detection, segmentation, and grounding
- **retrieval evaluation** for ranking, RAG, and GraphRAG
- **shared utilities** for schemas, metrics, and reporting helpers

## Current structure

```text
repo/
├── README.md
├── requirements-base.txt
├── requirements-visual.txt
├── requirements-retrieval.txt
└── evaluation/
    ├── README.md
    ├── shared/
    ├── visual/
    └── retrieval/
```

## Important note about the current visual evaluator

The existing visual evaluation code is **still the active implementation** and has not been moved yet. This is intentional.

The current reorganization is a **safe scaffolding pass**:
- add structure
- add documentation
- add split requirements
- add starter retrieval / RAG / GraphRAG evaluation modules
- avoid breaking existing visual evaluation scripts

After the structure settles, the next step is to gradually migrate existing visual evaluation code into the `evaluation/visual/` area.

## Main areas

### 1. Visual evaluation
This branch is for image and VLM-style tasks such as:
- classification
- captioning
- VQA
- HBB / OBB detection
- segmentation
- change detection
- visual grounding

### 2. Retrieval evaluation
This branch is for text retrieval and generation systems such as:
- dense retrieval
- hybrid retrieval
- RAG
- GraphRAG

### 3. Shared evaluation utilities
This area is for logic that can be reused across both visual and retrieval tasks, such as:
- ranking metrics
- JSON schemas
- reporting helpers
- aggregation utilities

## Requirements

The repo now uses split requirements files:

- `requirements-base.txt`
- `requirements-visual.txt`
- `requirements-retrieval.txt`

Typical installs:

```bash
pip install -r requirements-visual.txt
```

or

```bash
pip install -r requirements-retrieval.txt
```

## Suggested roadmap

### Near term
- keep the current visual evaluator working as-is
- expand retrieval / RAG / GraphRAG evaluation modules
- add examples for each evaluation family

### Next cleanup step
- migrate the current visual evaluator into `evaluation/visual/`
- move shared utilities into `evaluation/shared/`
- reduce legacy import fallbacks
- add tests

## Status

This repository is currently best described as:
- practical
- research-oriented
- in active reorganization

The goal of the current layout is to let the repo grow beyond VLM-only evaluation without forcing visual and retrieval evaluation into the same flat folder.
