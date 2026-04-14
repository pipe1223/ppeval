# Legacy visual evaluator

This directory contains the original visual evaluation implementation that shipped with the uploaded project.

It includes:
- combined result+GT evaluation
- separate result-vs-GT evaluation
- readers for results and GT files
- visual eval core modules for captioning, classification, detection, segmentation, and VQA

Top-level wrappers in `evaluation/` keep the original entry points working.
