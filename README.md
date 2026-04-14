# PP Evaluation

**PP Evaluation** stands for **Perception & Provenance Evaluation**.

It is a unified evaluation project for AI systems used in both research and engineering workflows.

- **Perception** covers visual, multimodal, and VLM-style tasks such as classification, captioning, VQA, detection, segmentation, and grounding.
- **Provenance** covers retrieval-grounded systems such as search, RAG, and GraphRAG, where answer quality depends on evidence quality and traceability.

The project is designed to gather useful evaluation methods that are commonly used across modern AI development rather than staying limited to a single benchmark style.

## Main goals

PP Evaluation aims to support four practical evaluation layers:

### Phase 1 — Shared core metrics
- classification-style precision / recall / F1
- exact match
- token F1
- ranking metrics such as Recall@k, MRR, nDCG, AP
- system metrics such as latency, token usage, cost, and success rate

### Phase 2 — RAG metrics
- retrieval ranking quality
- answer correctness
- context precision / recall
- citation precision / recall
- support rate
- deterministic faithfulness proxy

### Phase 3 — GraphRAG metrics
- node recall
- edge recall
- path hit rate
- subgraph coverage
- multi-hop answer success

### Phase 4 — Online / judge-based evaluation
- pairwise comparison
- rule-based validators
- LLM judge adapter interface
- human review queue generation and aggregation

## Project structure

```text
PP_Evaluation/
├── README.md
├── requirements-base.txt
├── requirements-visual.txt
├── requirements-retrieval.txt
├── requirements-all.txt
├── tianhuieval/
├── examples/
│   ├── retrieval/
│   └── online/
└── evaluation/
    ├── __init__.py
    ├── README.md
    ├── evaluation.py
    ├── evaluation_combined.py
    ├── evaluation_separated.py
    ├── result_gt_reader.py
    ├── result_reader.py
    ├── gt_reader.py
    ├── eval_core/
    ├── shared/
    ├── online/
    ├── retrieval/
    └── visual/
        └── legacy/
```

## Visual evaluation

The uploaded project already included a working visual evaluator. That implementation is preserved under:

```text
evaluation/visual/legacy/
```

Compatibility wrappers are left at the old paths so older commands continue to work:

- `evaluation/evaluation_combined.py`
- `evaluation/evaluation_separated.py`
- `evaluation/result_reader.py`
- `evaluation/result_gt_reader.py`
- `evaluation/gt_reader.py`
- `evaluation/eval_core/*`

## Retrieval evaluation

New retrieval modules are included for:

- generic ranking evaluation
- RAG evaluation
- GraphRAG evaluation

The key new modules are:

- `evaluation/retrieval/ranking/metrics.py`
- `evaluation/retrieval/rag/metrics.py`
- `evaluation/retrieval/rag/evaluate_rag.py`
- `evaluation/retrieval/graphrag/metrics.py`
- `evaluation/retrieval/graphrag/evaluate_graphrag.py`

## Online / engineering evaluation

The project now also includes:

- `evaluation/online/pairwise.py`
- `evaluation/online/validators.py`
- `evaluation/online/llm_judge.py`
- `evaluation/online/human_review.py`
- `evaluation/online/evaluate_pairwise.py`
- `evaluation/online/evaluate_system.py`

These are intended for engineering-style regression testing and review workflows.

## Installation

### Base installation

```bash
pip install -r requirements-base.txt
```

### Visual evaluation dependencies

```bash
pip install -r requirements-visual.txt
```

### Retrieval / online evaluation dependencies

```bash
pip install -r requirements-retrieval.txt
```

### Everything

```bash
pip install -r requirements-all.txt
```

## Examples

### Unified router

The main router is:

```bash
python evaluation/evaluation.py <input.json> --mode auto
```

Supported routing modes:
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
python evaluation/evaluation_separated.py --pred path/to/results.json --gt path/to/gt.json
```

### RAG evaluation

```bash
python evaluation/retrieval/rag/evaluate_rag.py examples/retrieval/rag_samples.json
```

### GraphRAG evaluation

```bash
python evaluation/retrieval/graphrag/evaluate_graphrag.py examples/retrieval/graphrag_samples.json
```

### Pairwise evaluation

```bash
python evaluation/online/evaluate_pairwise.py examples/online/pairwise_samples.json
```

### System metrics summary

```bash
python evaluation/online/evaluate_system.py examples/online/system_records.json
```

## Notes on compatibility

The original uploaded code relied in several places on imports under the name `tianhuieval`.

To avoid breaking that behavior during the reorganization, this project includes a compatibility alias package:

```text
tianhuieval/
```

That alias points older imports to the local legacy visual modules.

## Recommended next steps

If you want to continue evolving this project, the next high-value improvements are:

1. add unit tests for the new retrieval / online modules
2. add richer RAG faithfulness and judge-based metrics
3. add GraphRAG graph-construction quality metrics
4. add benchmark dataset schemas for each evaluation family
5. gradually modernize the visual legacy implementation into a non-legacy `visual/` structure

## License

Add your preferred license here.
