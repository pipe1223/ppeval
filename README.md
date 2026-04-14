# PP Evaluation

**PP Evaluation** is a unified evaluation framework for modern AI systems across **perception**, **retrieval**, and **production-quality** assessment.

In this project, **PP** stands for **Perception & Provenance**:

- **Perception** covers visual and multimodal tasks such as classification, captioning, VQA, detection, segmentation, and grounding.
- **Provenance** covers retrieval-grounded systems such as search, **RAG**, and **GraphRAG**, where answer quality depends on evidence quality, traceability, and support.

The goal of PP Evaluation is to provide a practical, extensible toolkit that is useful for both **research benchmarking** and **engineering regression testing**.

---

## Why this project exists

Evaluation in modern AI development is fragmented.

A visual model may need classification, detection, and segmentation metrics.  
A RAG system may need retrieval quality, answer correctness, faithfulness, and citation support.  
A production system may also need latency, token usage, cost, pairwise comparison, and human review.

PP Evaluation brings these workflows into one repository with a shared structure:

- **visual / VLM evaluation**
- **retrieval / RAG / GraphRAG evaluation**
- **shared core metrics**
- **online and engineering-oriented evaluation utilities**

---

## What PP Evaluation supports

### Visual and VLM evaluation
- classification
- captioning
- VQA
- HBB detection
- OBB detection
- segmentation
- change detection
- visual grounding

### Retrieval and grounded generation evaluation
- ranking metrics
- RAG evaluation
- GraphRAG evaluation
- answer correctness
- context precision / recall
- citation precision / recall
- faithfulness proxy
- node / edge / path-level graph retrieval checks

### Engineering and online evaluation
- pairwise comparison
- rule-based validation
- LLM judge adapter interface
- human review queue generation
- latency / token / cost summaries

---

## Project structure

```text
PP_Evaluation/
├── README.md
├── requirements-base.txt
├── requirements-visual.txt
├── requirements-retrieval.txt
├── requirements-all.txt
├── examples/
│   ├── online/
│   └── retrieval/
├── evaluation/
│   ├── __init__.py
│   ├── evaluation.py
│   ├── evaluation_combined.py
│   ├── evaluation_separated.py
│   ├── result_reader.py
│   ├── result_gt_reader.py
│   ├── gt_reader.py
│   ├── shared/
│   ├── retrieval/
│   ├── online/
│   ├── visual/
│   │   └── legacy/
│   └── eval_core/
└── tianhuieval/
```

---

## Design principles

### 1. One framework, multiple evaluation families
PP Evaluation is not limited to a single benchmark style. It is designed to support different AI workflows under a shared layout.

### 2. Backward compatibility for existing visual code
The repository preserves the original visual evaluator and keeps compatibility wrappers at the old paths, so existing usage can continue while the project evolves.

### 3. Shared core metrics
Common metrics such as precision / recall / F1, ranking metrics, text correctness, and system summaries are implemented in shared modules so they can be reused across task families.

### 4. Research + engineering
The project is intended to support both:
- **offline evaluation** for model comparison and benchmarking
- **engineering evaluation** for regression testing, system monitoring, and review workflows

---

## Installation

### Base dependencies

```bash
pip install -r requirements-base.txt
```

### Visual evaluation dependencies

```bash
pip install -r requirements-visual.txt
```

### Retrieval and online evaluation dependencies

```bash
pip install -r requirements-retrieval.txt
```

### Full installation

```bash
pip install -r requirements-all.txt
```

---

## Core entry points

### Unified router

The main router automatically detects the evaluation schema and dispatches to the correct evaluation flow.

```bash
python evaluation/evaluation.py <input.json> --mode auto
```

Supported modes:

- `auto`
- `combined`
- `separate`
- `rag`
- `graphrag`
- `pairwise`
- `system`

### Visual combined evaluation

```bash
python evaluation/evaluation_combined.py path/to/combined_eval.json --iou 0.5
```

### Visual separate evaluation

```bash
python evaluation/evaluation_separated.py   --pred path/to/results.json   --gt path/to/gt.json
```

### RAG evaluation

```bash
python evaluation/retrieval/rag/evaluate_rag.py   examples/retrieval/rag_samples.json
```

### GraphRAG evaluation

```bash
python evaluation/retrieval/graphrag/evaluate_graphrag.py   examples/retrieval/graphrag_samples.json
```

### Pairwise evaluation

```bash
python evaluation/online/evaluate_pairwise.py   examples/online/pairwise_samples.json
```

### System metrics evaluation

```bash
python evaluation/online/evaluate_system.py   examples/online/system_records.json
```

---

## Evaluation families

## 1. Shared core metrics

Location:

```text
evaluation/shared/metrics/
```

Includes:
- core precision / recall / F1 style utilities
- ranking metrics
- text correctness helpers
- system-level summaries

Representative modules:
- `evaluation/shared/metrics/core.py`
- `evaluation/shared/metrics/ranking.py`
- `evaluation/shared/metrics/text.py`
- `evaluation/shared/metrics/system.py`

These modules are intended to be reusable across visual, retrieval, and engineering evaluation.

---

## 2. Visual evaluation

Location:

```text
evaluation/visual/legacy/
```

This contains the preserved visual evaluator, including:
- combined JSON evaluation
- separate prediction-vs-GT evaluation
- readers
- eval core modules for visual tasks

Compatibility wrappers are preserved at:
- `evaluation/evaluation_combined.py`
- `evaluation/evaluation_separated.py`
- `evaluation/result_reader.py`
- `evaluation/result_gt_reader.py`
- `evaluation/gt_reader.py`
- `evaluation/eval_core/*`

This allows older imports and scripts to keep working while the project structure becomes broader.

---

## 3. Retrieval, RAG, and GraphRAG evaluation

Location:

```text
evaluation/retrieval/
```

### Ranking
General ranking metrics such as:
- Precision@k
- Recall@k
- HitRate@k
- MRR
- Average Precision
- nDCG

### RAG
RAG evaluation includes:
- retrieval ranking quality
- context precision
- context recall
- answer correctness
- citation precision / recall
- support rate
- deterministic faithfulness proxy

### GraphRAG
GraphRAG evaluation includes:
- ranking metrics
- node recall
- edge recall
- path hit rate
- subgraph coverage
- multi-hop answer success

Representative modules:
- `evaluation/retrieval/ranking/metrics.py`
- `evaluation/retrieval/rag/metrics.py`
- `evaluation/retrieval/rag/evaluate_rag.py`
- `evaluation/retrieval/graphrag/metrics.py`
- `evaluation/retrieval/graphrag/evaluate_graphrag.py`

---

## 4. Online and engineering evaluation

Location:

```text
evaluation/online/
```

This part of the project supports engineering-oriented assessment workflows.

Included components:
- pairwise comparison
- rule-based validators
- LLM judge adapter interface
- human review queue generation and aggregation
- system metric reporting

Representative modules:
- `evaluation/online/pairwise.py`
- `evaluation/online/validators.py`
- `evaluation/online/llm_judge.py`
- `evaluation/online/human_review.py`
- `evaluation/online/evaluate_pairwise.py`
- `evaluation/online/evaluate_system.py`

---

## Examples

Example input files are included under:

```text
examples/retrieval/
examples/online/
```

Current examples:
- `examples/retrieval/rag_samples.json`
- `examples/retrieval/graphrag_samples.json`
- `examples/online/pairwise_samples.json`
- `examples/online/system_records.json`

These are useful as both smoke-test inputs and starter schemas for integrating your own data.

---

## Compatibility with older imports

Some original code paths relied on imports under:

```text
tianhuieval/
```

This project keeps a compatibility alias package so those older imports continue to resolve to the local implementation.

That means PP Evaluation can evolve structurally without forcing an immediate rewrite of every legacy visual script.

---

## Intended use cases

PP Evaluation is well suited for:

- vision-language model benchmarking
- remote sensing and visual understanding tasks
- retrieval system evaluation
- RAG / GraphRAG experiments
- model comparison and ablation studies
- engineering regression tests
- review workflows for model outputs
- system-level tracking of latency, token usage, and cost

---

