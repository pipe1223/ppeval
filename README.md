# Unified evaluation project

This package has been reorganized to support more than visual or VLM evaluation.

## Main structure

```text
evaluation/
├── evaluation.py                # unified router
├── evaluation_combined.py       # compatibility wrapper
├── evaluation_separated.py      # compatibility wrapper
├── visual/
│   └── legacy/                  # moved visual evaluator
├── retrieval/
│   ├── ranking/
│   ├── rag/
│   └── graphrag/
└── shared/
    └── metrics/
```

## What was changed

- Moved the current visual evaluator into `evaluation/visual/legacy/`
- Left compatibility wrappers at the old paths
- Added retrieval / RAG / GraphRAG evaluators
- Added shared ranking and text metrics
- Added split requirements files

## Typical usage

### Visual combined
```bash
python evaluation/evaluation.py path/to/combined.json --mode visual-combined
```

### Visual separate
```bash
python evaluation/evaluation.py path/to/results.json --gt path/to/gt.json --mode visual-separate
```

### RAG
```bash
python evaluation/evaluation.py evaluation/retrieval/examples/rag_sample.json --mode rag
```

### GraphRAG
```bash
python evaluation/evaluation.py evaluation/retrieval/examples/graphrag_sample.json --mode graphrag
```
