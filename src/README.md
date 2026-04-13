# Source Code Guide

This is a short guide to help new users understand where the main code lives.

If you are new to this repository, start with:

1. [src/train_patient_held_out/](/home/yehyun/Landmark-based-2D-3D-Registration-Uncertainty/src/train_patient_held_out/README.md)
2. [src/train_patient_held_out/main.py](/home/yehyun/Landmark-based-2D-3D-Registration-Uncertainty/src/train_patient_held_out/main.py)
3. [src/data/1_extract_content.py](/home/yehyun/Landmark-based-2D-3D-Registration-Uncertainty/src/data/1_extract_content.py)
4. [src/data/2_project.py](/home/yehyun/Landmark-based-2D-3D-Registration-Uncertainty/src/data/2_project.py)

## Main Idea

The repository has one primary path and a few supporting modules.

Primary path:
- `src/train_patient_held_out/`

Optional real-image path:
- `src/train_deepfluoro_real/`

## Folder Overview

### `src/data/`

This folder prepares the dataset.

Important files:
- `1_extract_content.py`
  - extracts the DeepFluoro archive
  - writes CT volumes, segmentations, 3D landmarks, and projection images
- `2_project.py`
  - generates DRR images
  - writes 2D landmark CSV files
  - writes pose parameter CSV files

### `src/train/`

This folder contains shared training utilities used by the main pipelines.

Important files:
- `model.py`
  - defines the main U-Net landmark detector
- `utils.py`
  - seed and argument helpers
- `visualization.py`
  - training overlays and plots

### `src/test/`

This folder contains shared testing and evaluation logic.

Important files:
- `model.py`
  - dropout-enabled U-Net used for MC-dropout
- `data.py`
  - saves and reloads prediction CSV files
  - computes uncertainty from MC samples
- `uncertainty.py`
  - filtering and weighting logic for uncertainty-aware evaluation
- `pose_estimation.py`
  - pose solvers used during testing and finetuning
- `result.py`
  - saves final test summary CSV files

### `src/train_patient_held_out/`

This is the main package for the baseline reproduction workflow.

Important files:
- `main.py`
  - central entrypoint
  - dispatches training, finetuning, and testing
- `data_loader.py`
  - builds held-out train/val/test splits
  - loads DRR images and landmark labels
- `train.py`
  - supervised landmark detector training
- `test.py`
  - deterministic testing
  - MC-dropout prediction
  - final CSV export
- `finetune.py`
  - pose-aware finetuning
- `log.py`
  - W&B logging

### `src/train_deepfluoro_real/`

This package is for the optional real fluoroscopy experiments.

Important files:
- `main.py`
  - entrypoint for real-image training and testing
- `data_loader.py`
  - builds real-image manifests and mixed-domain loaders
- `train.py`
  - training loop for the real-image experiments
- `test.py`
  - evaluation on held-out real projections

### `src/deepfluoro_real/`

This folder contains lower-level utilities used by the real-image path.

Important files:
- `io.py`
  - reads calibration, landmarks, and poses from the DeepFluoro HDF5 file
- `detector.py`
  - detector preprocessing and inference bridge for raw fluoroscopy images
- `pose_estimation.py`
  - geometry and pose estimation for the real-image setting

### `src/sweep/`

This folder contains utilities for combining per-run results into sweep summaries.

## Recommended Reading Order

If you are trying to learn the project quickly, use this order:

1. `src/train_patient_held_out/main.py`
2. `src/train_patient_held_out/data_loader.py`
3. `src/train_patient_held_out/train.py`
4. `src/train_patient_held_out/test.py`
5. `src/test/uncertainty.py`
6. `src/test/pose_estimation.py`
7. `src/train_patient_held_out/finetune.py`

If you are specifically interested in the real-image experiments, read:

1. `src/train_deepfluoro_real/main.py`
2. `src/train_deepfluoro_real/data_loader.py`
3. `src/train_deepfluoro_real/test.py`
4. `src/deepfluoro_real/`

## Practical Advice For Newcomers

- Start from the patient-held-out path, not the optional real-image path.
- Treat `src/train/` and `src/test/` as shared support code.
- Use the shell scripts in `script/` to see how experiments were launched.
- Use `analysis/` only after you understand which CSV files are produced by testing.

For overall project setup and the full workflow, see the root [README.md](/home/yehyun/Landmark-based-2D-3D-Registration-Uncertainty/README.md).
