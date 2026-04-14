# GraphRAG evaluation

This directory is for evaluating graph-assisted retrieval and generation systems.

## What makes GraphRAG different from plain RAG

GraphRAG systems often retrieve more than flat text chunks. They may retrieve or expand through:
- nodes
- edges
- communities
- paths
- subgraphs

Because of that, GraphRAG evaluation should usually include both standard retrieval metrics and graph-specific checks.

## Useful GraphRAG metrics

### Retrieval-style metrics
- Recall@k
- Precision@k
- MRR
- nDCG

### Graph-specific metrics
- node recall
- edge recall
- path hit rate
- subgraph coverage
- reasoning path usefulness

## Starter module

The included `evaluate_graphrag.py` is a lightweight baseline evaluator that computes:
- standard ranking metrics on retrieved context ids
- node recall from `retrieved_node_ids` vs `gold_node_ids`
- edge recall from `retrieved_edge_ids` vs `gold_edge_ids`
- path hit rate from exact path matches when path annotations are available

## Suggested input schema

Each sample can contain fields like:

```json
{
  "question": "What connects A to B?",
  "retrieved_ids": ["ctx-1", "ctx-4"],
  "gold_ids": ["ctx-4"],
  "retrieved_node_ids": ["n1", "n2", "n3"],
  "gold_node_ids": ["n2", "n5"],
  "retrieved_edge_ids": ["e1", "e2"],
  "gold_edge_ids": ["e2"],
  "retrieved_paths": [["n1", "n2", "n5"]],
  "gold_paths": [["n1", "n2", "n5"]]
}
```

This starter evaluator is meant as a clean base, not a final GraphRAG judge.
