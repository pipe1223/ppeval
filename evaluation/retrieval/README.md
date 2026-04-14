# Retrieval evaluation

This directory is for evaluating retrieval systems and retrieval-augmented generation workflows.

Planned scope:
- classic ranking evaluation
- dense retrieval
- hybrid retrieval
- RAG evaluation
- GraphRAG evaluation

## Main subareas

```text
retrieval/
├── README.md
├── ranking/
├── rag/
└── graphrag/
```

## Retrieval evaluation layers

A practical retrieval or RAG evaluation usually has multiple layers:

### 1. Retrieval quality
Examples:
- Recall@k
- Precision@k
- Hit@k
- MRR
- nDCG
- Average Precision

### 2. Answer quality
Examples:
- exact match
- token F1
- semantic similarity
- task success rate

### 3. Grounding / support quality
Examples:
- was the answer supported by retrieved context?
- were the cited chunks correct?
- was the answer faithful to evidence?

### 4. Graph-specific quality
For GraphRAG, add checks such as:
- node retrieval recall
- edge retrieval recall
- path hit rate
- subgraph usefulness

## Design principle

The retrieval area should stay separate from visual evaluation even when some metrics overlap.

Shared pieces can later be promoted into `evaluation/shared/`, but retrieval-specific task logic should remain here.
