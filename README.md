# PP Evaluation

**PP Evaluation** is a unified evaluation toolkit for modern AI systems that need to be measured across both **perception** and **provenance** tasks.

In this project, **PP** stands for **Perception & Provenance**:
- **Perception** covers visual and vision-language evaluation such as classification, captioning, VQA, detection, grounding, segmentation, and change detection.
- **Provenance** covers evidence-aware retrieval workflows such as RAG and GraphRAG, where answer quality must be evaluated together with retrieval quality and grounding quality.

The goal of PP Evaluation is to give one project a clean, extensible structure for evaluating:
- visual / VLM outputs,
- retrieval systems,
- RAG pipelines,
- GraphRAG pipelines,
- and future task families that share common reporting, ranking, and text-matching logic.

---

## Why this project exists

Most evaluation repos start narrow and become messy as new task families are added. Visual evaluation, retrieval evaluation, RAG evaluation, and graph-based retrieval often end up split across unrelated scripts with incompatible input formats and duplicated metrics.

PP Evaluation is designed to avoid that drift by organizing the code into:
- **visual evaluation**,
- **retrieval evaluation**,
- and **shared metrics/utilities**.

It also preserves backwards compatibility for older visual workflows by keeping compatibility wrappers at the original paths.

---

## What PP Evaluation supports today

### Visual / VLM evaluation
The current visual stack supports combined and separate evaluation flows for tasks such as:
- image / region / pixel classification,
- captioning,
- VQA,
- horizontal bounding-box detection,
- oriented bounding-box detection,
- visual grounding,
- segmentation,
- change detection.

### Retrieval / RAG evaluation
The retrieval stack includes:
- ranking metrics for retrieved IDs,
- lightweight RAG evaluation,
- lightweight GraphRAG evaluation.

### Shared metrics
The shared layer includes reusable utilities such as:
- `Precision@k`,
- `Recall@k`,
- `HitRate@k`,
- `MRR`,
- `Average Precision`,
- `nDCG@k`,
- `Exact Match`,
- token-level `F1`.

---

## Project structure

```text
.
├── README.md
├── requirements.txt
├── requirements-base.txt
├── requirements-visual.txt
├── requirements-retrieval.txt
└── evaluation/
    ├── evaluation.py                 # unified router for visual / RAG / GraphRAG
    ├── evaluation_combined.py        # compatibility wrapper
    ├── evaluation_separated.py       # compatibility wrapper
    ├── result_reader.py              # compatibility wrapper
    ├── result_gt_reader.py           # compatibility wrapper
    ├── gt_reader.py                  # compatibility wrapper
    ├── eval_core/                    # compatibility wrappers for old visual imports
    ├── visual/
    │   ├── README.md
    │   └── legacy/                   # preserved visual evaluator implementation
    ├── retrieval/
    │   ├── README.md
    │   ├── examples/
    │   ├── ranking/
    │   ├── rag/
    │   └── graphrag/
    └── shared/
        ├── metrics/
        ├── schemas/
        └── utils/
```

### Design notes
- `evaluation/visual/legacy/` contains the moved visual evaluator.
- The old entry points still exist as wrappers so existing code is less likely to break.
- `evaluation/retrieval/` contains retrieval-family evaluators.
- `evaluation/shared/` is where reusable metrics and helpers belong as the framework grows.

---

## Installation

### Base installation

```bash
pip install -r requirements-base.txt
```

### Full visual stack

```bash
pip install -r requirements-visual.txt
```

### Retrieval stack

```bash
pip install -r requirements-retrieval.txt
```

### Default install

The default `requirements.txt` currently points to the visual stack:

```bash
pip install -r requirements.txt
```

---

## Quick start

### 1. Unified router

The main entry point for the whole project is:

```bash
python evaluation/evaluation.py <path-to-json>
```

You can let the router infer the evaluation family automatically, or force a specific mode.

### 2. Visual evaluation: combined JSON

Use this when predictions and ground truth are stored in the same file.

```bash
python evaluation/evaluation.py path/to/combined.json --mode visual-combined
```

### 3. Visual evaluation: separate result + GT

Use this when predictions and GT are stored separately.

```bash
python evaluation/evaluation.py path/to/results.json \
  --gt path/to/gt.json \
  --mode visual-separate
```

You can also pass multiple GT files or a GT directory.

### 4. RAG evaluation

```bash
python evaluation/evaluation.py evaluation/retrieval/examples/rag_sample.json --mode rag
```

### 5. GraphRAG evaluation

```bash
python evaluation/evaluation.py evaluation/retrieval/examples/graphrag_sample.json --mode graphrag
```

---

## Evaluation modes

### Visual combined mode
This mode evaluates a JSON file that contains both:
- model predictions (`answer`), and
- ground truth (`gt`).

This is useful when exporting a single benchmark file for direct scoring.

### Visual separate mode
This mode evaluates:
- a result-only JSON file, plus
- one or more GT JSON files.

This is useful for larger evaluation pipelines where inference and benchmarking are separated.

### RAG mode
The RAG evaluator is designed for sample records that may contain fields such as:
- `retrieved_ids`
- `gold_ids`
- `answer`
- `gold_answer`
- `is_supported`

It reports retrieval quality and answer quality together.

### GraphRAG mode
The GraphRAG evaluator extends RAG-style scoring with graph-aware fields such as:
- `retrieved_node_ids`
- `gold_node_ids`
- `retrieved_edge_ids`
- `gold_edge_ids`
- `retrieved_paths`
- `gold_paths`

This makes it possible to score not just whether the right context was retrieved, but whether the right graph structure was retrieved.

---

## Metrics

### Retrieval metrics
Used by retrieval, RAG, and GraphRAG workflows:
- `Precision@k`
- `Recall@k`
- `HitRate@k`
- `MRR`
- `AP`
- `nDCG@k`

### Text answer metrics
Used by RAG and GraphRAG answer scoring:
- `ExactMatch`
- token-level `F1`

### RAG-specific output
The current RAG evaluator can also compute:
- `SupportRate` (when `is_supported` labels are provided)

### GraphRAG-specific output
The current GraphRAG evaluator can compute:
- `NodeRecall`
- `EdgeRecall`
- `PathHitRate`

### Visual metrics
The visual legacy evaluator supports task-specific outputs such as:
- Accuracy / Precision / Recall / F1 for classification and VQA,
- BLEU / METEOR / ROUGE / CIDEr for captioning,
- mAP and per-class AP for detection,
- mDice / mIoU / VC-style metrics for segmentation and change detection.

---

## Input conventions

### Visual combined schema
A combined visual file typically looks like this at the top level:

```json
{
  "info": {
    "task": "PIX_SEG",
    "model": "your_model",
    "dataset": "your_dataset"
  },
  "data": [
    {
      "image": ["path/to/image.png"],
      "question": "Segment out road in the image",
      "answer": [],
      "gt": []
    }
  ]
}
```

### Visual separate schema
A separate visual result file typically contains:
- `task_config`
- `results`

with GT stored in separate JSON files.

### RAG schema
A minimal RAG sample can look like:

```json
{
  "question": "Who wrote the paper?",
  "retrieved_ids": ["chunk-1", "chunk-2", "chunk-3"],
  "gold_ids": ["chunk-2"],
  "answer": "Alice Smith",
  "gold_answer": "Alice Smith",
  "is_supported": true
}
```

### GraphRAG schema
A minimal GraphRAG sample can look like:

```json
{
  "question": "How is A connected to C?",
  "retrieved_ids": ["ctx-1", "ctx-5"],
  "gold_ids": ["ctx-5"],
  "retrieved_node_ids": ["A", "B", "C"],
  "gold_node_ids": ["B", "C"],
  "retrieved_edge_ids": ["A-B", "B-C"],
  "gold_edge_ids": ["B-C"],
  "retrieved_paths": [["A", "B", "C"]],
  "gold_paths": [["A", "B", "C"]],
  "answer": "A connects to C through B.",
  "gold_answer": "A connects to C through B."
}
```

---

## Backwards compatibility

A core design requirement of this reorganization is **not breaking the existing visual evaluator**.

That is why the project still exposes older paths such as:
- `evaluation/evaluation_combined.py`
- `evaluation/evaluation_separated.py`
- `evaluation/result_reader.py`
- `evaluation/result_gt_reader.py`
- `evaluation/gt_reader.py`
- `evaluation/eval_core/*`

These paths now act as compatibility wrappers that point to the moved implementation under `evaluation/visual/legacy/`.

This means you can modernize the project structure without forcing all old scripts to be rewritten immediately.

---

## Extending the framework

PP Evaluation is structured so new evaluation families can be added cleanly.

### Add a new visual task
If the task is specific to image, region, or multimodal perception, it should typically go under:

```text
evaluation/visual/
```

### Add a new retrieval task
If the task is evidence retrieval, answer grounding, or graph-based context retrieval, it should typically go under:

```text
evaluation/retrieval/
```

### Add a reusable metric
If a metric can be shared across task families, it should usually go under:

```text
evaluation/shared/
```

This structure keeps the codebase scalable as the project grows beyond its original visual-only scope.

---

## Current status

PP Evaluation is already useful as a working framework, but it should still be viewed as an evolving engineering/research project.

Current strengths:
- unified structure,
- preserved visual compatibility,
- reusable ranking and text metrics,
- built-in RAG and GraphRAG starter evaluators.

Current limitations:
- some visual components still live under a `legacy` namespace,
- some domain-specific visual modules are specialized and not yet fully normalized,
- schemas are still lightweight rather than fully formalized,
- test coverage can be expanded further.

---

## Suggested roadmap

Recommended next steps for the project:
- migrate more reusable logic from `visual/legacy/` into `shared/`,
- formalize schemas under `evaluation/shared/schemas/`,
- add more benchmark examples for each task family,
- expand RAG / GraphRAG judges beyond lightweight starter metrics,
- add automated tests for visual and retrieval families,
- add CI and release packaging when the structure stabilizes.

---

## Naming rationale

**PP Evaluation = Perception & Provenance Evaluation**

That name works well because the project covers two important sides of modern AI system evaluation:
- **Perception**: what the model sees, understands, classifies, describes, localizes, or segments.
- **Provenance**: where the answer comes from, whether the right evidence was retrieved, and whether the answer is grounded in that evidence.

This makes the name broad enough for visual evaluation, RAG, GraphRAG, and future multimodal evaluation work.

---

## License

Add your preferred license here.

---

## Citation

If you use PP Evaluation in a paper, system report, or benchmark release, add the preferred citation block here.
