# RAG evaluation

This directory is for evaluating retrieval-augmented generation workflows.

## Practical evaluation layers

A useful RAG evaluation usually covers three separate questions:

### 1. Did retrieval fetch the right evidence?
Examples:
- Recall@k
- Precision@k
- MRR
- nDCG
- Hit@k

### 2. Did the system answer correctly?
Examples:
- exact match
- token F1
- semantic judge score

### 3. Was the answer grounded in evidence?
Examples:
- citation support
- faithfulness
- answer-context consistency

## Starter module

The included `evaluate_rag.py` is a lightweight baseline evaluator that currently computes:
- retrieval ranking metrics
- exact match
- token-level F1
- support rate from a boolean field when available

It is intentionally simple so it can be extended to your own dataset and judging setup.

## Suggested input schema

Each sample can contain fields like:

```json
{
  "question": "What is ...?",
  "retrieved_ids": ["chunk-1", "chunk-9", "chunk-2"],
  "gold_ids": ["chunk-2", "chunk-9"],
  "answer": "...",
  "gold_answer": "...",
  "is_supported": true
}
```

You can later expand this with:
- retrieved text chunks
- citation spans
- judge scores
- hallucination labels
