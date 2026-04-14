# VLM/Visual

A lightweight multitask evaluation toolkit for vision-language model (VLM) and visual model outputs.

`ppeval` is designed for evaluating predictions across several visual understanding tasks with a single, JSON-based workflow. The current codebase supports two main evaluation modes:

1. **Combined evaluation**  
   A single JSON file contains both model predictions (`answer`) and ground truth (`gt`).

2. **Separated evaluation**  
   Model outputs and ground truth are stored in different JSON files and matched by image path.

The repository is especially useful for research workflows where a model may produce outputs for classification, captioning, VQA, detection, segmentation, or change detection, and you want one evaluation entry point rather than a different script for every task.

---

## What this repo currently supports

### Task families

The evaluator routes metrics by task tag.

#### Classification
- `IMG_CLS`
- `REG_CLS_HBB`
- `REG_CLS_OBB`
- `PIX_CLS`

#### Captioning
- `IMG_CAP`
- `IMG_CAP_DETAILED`
- `REG_CAP`

#### VQA
- `IMG_CT`
- `IMG_VQA`
- `VQA_numComp`
- `VQA_presence`
- `VQA_ruralUrban`

#### Detection / Grounding
- `REG_DET_HBB`
- `REG_DET_OBB`
- `REG_VG`

#### Segmentation / Change Detection
- `PIX_SEG`
- `PIX_CHG`

### Metric outputs by task

Depending on task type, the evaluator returns one or more of the following:

- **Classification / VQA**
  - Accuracy
  - Precision
  - Recall
  - F1
  - Per-class metrics for classification

- **Captioning**
  - BLEU
  - METEOR
  - ROUGE
  - ROUGE-L
  - CIDEr

- **HBB Detection**
  - mAP
  - AP per class
  - GT count per class
  - macro Precision / Recall / F1
  - configurable IoU threshold

- **Segmentation / Change Detection**
  - mDice
  - mIoU
  - mVC8
  - mVC16
  - optional per-class aggregation when available

---

## Repository layout

```text
evaluation/
├── evaluation_combined.py
├── evaluation_separated.py
├── result_gt_reader.py
├── result_reader.py
├── gt_reader.py
└── eval_core/
    ├── const.py
    ├── classification.py
    ├── detection.py
    ├── detection_hbb.py
    └── segmentation.py
```

### Main files

- `evaluation/evaluation_combined.py`  
  Main entry point for evaluating a single JSON that already contains both prediction and GT.

- `evaluation/evaluation_separated.py`  
  Entry point for evaluating model result JSON against one or more GT JSON files.

- `evaluation/result_gt_reader.py`  
  Reader for combined result+GT JSON files.

- `evaluation/result_reader.py`  
  Reader for result-only JSON files.

- `evaluation/gt_reader.py`  
  Reader for GT JSON files. Supports both single-task GT and mixed-task GT with per-item `task_type`.

- `evaluation/eval_core/*`  
  Core metric logic for classification, HBB detection, segmentation, and task constants.

---

## Current status

This repository already has a good evaluation core, but it is still closer to a research/internal prototype than a fully packaged public library.

Important current status notes:

- There is **no packaged installation config yet** (`setup.py`, `pyproject.toml`, or existing `requirements.txt` was not found).
- The code is **script-oriented**, so the main workflow is to run files directly with Python.
- Some evaluation branches in `evaluation_combined.py` and `evaluation_separated.py` still **fall back to external `tianhuieval` modules** when local modules are not available.

In practice, that means:

- **Classification**, **segmentation**, and **HBB detection** are largely implemented inside this repo.
- **Captioning**, **VQA**, and **OBB / VG evaluation** currently appear to rely on legacy fallback imports from `tianhuieval` unless you add local replacements.

So if you want full out-of-the-box support for all task families, either:

1. keep your legacy `tianhuieval` environment available, or
2. add local modules for:
   - `eval_core/captioning.py`
   - `eval_core/vqa.py`
   - `eval_core/detection_obb.py`

---

## Installation

This repo is not yet packaged, so the simplest workflow is a normal Python virtual environment.

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run from the `evaluation/` directory

```bash
cd evaluation
```

---

## Recommended Python version

Python 3.10 or newer is recommended.

---

## Requirements and dependency notes

The included `requirements.txt` covers the imports that are currently visible in this repository.

It includes packages used directly by the current codebase, such as:

- `numpy`
- `matplotlib`
- `scikit-learn`
- `imbalanced-learn`
- `pycocotools`
- `scikit-image`
- `tqdm`
- `seaborn`
- `opencv-python`

### Important note about legacy fallback modules

`requirements.txt` does **not** install `tianhuieval`.

That dependency is not bundled in this repo and may be private, local, or from another codebase. If your current evaluation flow still depends on `tianhuieval`, you will need to keep that environment available or port the missing modules into this repository.

---

## Evaluation mode 1: Combined JSON

Use this when predictions and GT are already stored in the same JSON file.

### Expected top-level schema

```json
{
  "info": {
    "task": "PIX_SEG",
    "model": "your_model_name",
    "dataset": "your_dataset"
  },
  "data": [
    {
      "image": ["path/to/image.png"],
      "crop": [0, 0, 512, 512],
      "question": "Segment out road in the image",
      "answer": [],
      "gt": []
    }
  ]
}
```

### Notes

- `image` may be:
  - a string
  - `["img_path"]`
  - `["img_t1", "img_t2"]` for paired-image tasks such as change detection
- `crop` may be:
  - omitted
  - empty (`[]`)
  - a 4-value list `[x1, y1, x2, y2]`
- `data` may be:
  - a list
  - a dict keyed by id
- `question` is optional, but useful for extracting class names in some tasks

### Run combined evaluation

```bash
cd evaluation
python evaluation_combined.py path/to/combined_eval.json --iou 0.5
```

### Example output structure

```json
{
  "info": {
    "task": "PIX_SEG",
    "dataset": "example_dataset",
    "model": "example_model",
    "n_eval": 100
  },
  "results": {
    "mDICE": 72.31,
    "mIoU": 61.05,
    "mVC8": 58.00,
    "mVC16": 67.00
  }
}
```

The exact result keys depend on task type.

---

## Evaluation mode 2: Separated result JSON and GT JSON

Use this when model predictions and GT are stored separately.

### Result-only JSON format

The repo supports two result-only formats.

#### A. New recommended dict-based format

```json
{
  "info": {
    "model": "your_model_name",
    "dataset": "your_dataset"
  },
  "data": {
    "image/Image1.tif": {
      "task_config": {
        "image_path": "image/Image1.tif",
        "task_type": "IMG_CLS"
      },
      "results": {
        "merged_results": {
          "prediction": ["airport"]
        }
      }
    }
  }
}
```

#### B. Old list-based format

```json
{
  "info": {
    "model": "your_model_name",
    "dataset": "your_dataset"
  },
  "data": [
    {
      "task_config": {
        "image_path": "image/Image1.tif",
        "task_type": "IMG_CLS"
      },
      "results": {
        "merged_results": {
          "prediction": ["airport"]
        }
      }
    }
  ]
}
```

### GT JSON format

The GT reader supports both dict-based and list-based GT layouts.

#### Recommended dict-based GT format

```json
{
  "info": {
    "dataset": "your_dataset",
    "task": "IMG_CLS"
  },
  "data": {
    "image/Image1.tif": {
      "image_path": "image/Image1.tif",
      "image2_path": null,
      "gt": [
        {
          "label": "airport",
          "question": "What is the scene category?"
        }
      ]
    }
  }
}
```

### Mixed-task GT support

A GT file can also contain per-item `task_type` inside `gt[]`, which allows one GT file to store multiple tasks.

Example:

```json
{
  "info": {
    "dataset": "your_dataset"
  },
  "data": {
    "image/Image1.tif": {
      "image_path": "image/Image1.tif",
      "gt": [
        {
          "label": "airport",
          "question": "What is the scene category?",
          "task_type": "IMG_CLS"
        }
      ]
    }
  }
}
```

### Run separated evaluation with explicit GT files

```bash
cd evaluation
python evaluation_separated.py   --pred path/to/results.json   --gt path/to/gt_cls.json   --gt path/to/gt_seg.json   --result-key merged_results   --iou 0.5
```

### Run separated evaluation with a GT directory

```bash
cd evaluation
python evaluation_separated.py   --pred path/to/results.json   --gt-dir path/to/gt_folder   --result-key merged_results   --iou 0.5
```

### Matching behavior

Prediction entries are matched to GT primarily by:

1. exact image path
2. basename fallback if full paths differ

This makes the evaluator more tolerant when predictions and GT use slightly different path roots.

---

## Reader utilities

The repo contains useful readers for inspecting files before evaluation.

### Inspect a combined file

```bash
cd evaluation
python result_gt_reader.py path/to/combined_eval.json --summary --print-first
```

### Inspect a result-only file

```bash
cd evaluation
python result_reader.py path/to/results.json --summary --print-keys
```

### Inspect a GT file

```bash
cd evaluation
python gt_reader.py path/to/gt.json --summary --split --print-first
```

These utilities are helpful when debugging:

- wrong task tags
- missing result keys
- path mismatches
- mixed-task GT files
- malformed geometry fields

---

## Prediction formats by task

### Text-like tasks
For classification, captioning, and VQA, predictions are generally stored as text or text-like items under:

```json
{
  "results": {
    "merged_results": {
      "prediction": ["answer text"]
    }
  }
}
```

### Detection-like tasks
Detection predictions commonly use objects with geometry and labels:

```json
{
  "results": {
    "merged_results": {
      "prediction": [
        {
          "coord": [10, 20, 100, 120],
          "label": "vehicle",
          "confidence": 0.93
        }
      ]
    }
  }
}
```

### Combined HBB / polygon-like tasks
In combined mode, `answer` and `gt` may be stored as lists directly, and the evaluator converts them into the expected internal representation for each task.

---

## Typical workflows

### Workflow A: quick evaluation from one file
Use combined mode when you already merged prediction and GT into one JSON for one task.

```bash
cd evaluation
python evaluation_combined.py sample.json
```

### Workflow B: benchmark a result file against GT
Use separated mode when you have a result file exported from inference and one or more GT files.

```bash
cd evaluation
python evaluation_separated.py --pred result.json --gt-dir gt/
```

### Workflow C: inspect before evaluating
Use the reader scripts first if the evaluator reports missing GT, unsupported task, or malformed geometry.

```bash
cd evaluation
python result_reader.py result.json --summary --print-keys
python gt_reader.py gt.json --summary --split
```

---

## Known limitations

At the current stage, users should be aware of the following:

1. **Not fully packaged yet**  
   The repo currently behaves like a script collection rather than a pip-installable library.

2. **Partial legacy fallback dependency**  
   Some tasks still depend on `tianhuieval` fallback imports.

3. **No official examples folder yet**  
   The code supports multiple JSON formats, but the repo would benefit from ready-made example files.

4. **Some modules still reflect research iteration**
   A few files mix older script-style logic with newer reusable library-style logic.

5. **Dependency list is practical, not guaranteed complete for every legacy branch**
   The included `requirements.txt` reflects current visible imports in this repo. If you activate legacy fallback paths, you may need additional packages from your older environment.

---

## Recommended next improvements

To make this repo easier to share and reuse, the next highest-value improvements are:

1. Add an `examples/` folder with:
   - one combined JSON example
   - one result-only JSON example
   - one GT JSON example

2. Add missing local task modules so all tasks run without `tianhuieval`:
   - `eval_core/captioning.py`
   - `eval_core/vqa.py`
   - `eval_core/detection_obb.py`

3. Add:
   - `pyproject.toml`
   - test files
   - unit tests for each task family

4. Clean up:
   - duplicated helper functions
   - missing imports
   - legacy script-only code paths

---

## Minimal quickstart

```bash
git clone <your-repo-url>
cd ppeval
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd evaluation
python result_reader.py path/to/results.json --summary
python gt_reader.py path/to/gt.json --summary
python evaluation_separated.py --pred path/to/results.json --gt path/to/gt.json
```

---

## License

Add your preferred license here, for example:

- MIT
- Apache-2.0
- BSD-3-Clause

---

## Citation

If you use this repo in a paper or project, add your preferred citation block here.
