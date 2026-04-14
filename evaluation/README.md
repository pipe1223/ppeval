# Evaluation framework

This directory now supports multiple evaluation families:

- `visual/legacy/` for the moved visual and VLM evaluator
- `retrieval/` for retrieval, RAG, and GraphRAG
- `shared/` for reusable metrics

The existing visual code was preserved and moved, while compatibility wrappers were left in place so older import paths and commands are less likely to break.
