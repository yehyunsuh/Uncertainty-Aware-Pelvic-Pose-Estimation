# Landmark-based-2D-3D-Registration-Uncertainty

## Introduction

This repository contains the code for uncertainty-aware landmark-based 2D/3D pelvic pose estimation from DeepFluoro data.

The main reproduction path is:
- data extraction
- DRR generation
- patient-held-out training
- MC-dropout testing
- finetuning
- final result aggregation

Optional real-image experiments are also included:
- [src/train_deepfluoro_real/](/home/yehyun/Landmark-based-2D-3D-Registration-Uncertainty/src/train_deepfluoro_real/README.md)

## Directory Structure

```text
.
├── README.md
├── data/
│   ├── README.md
│   └── DeepFluoro/
├── script/
├── src/
│   ├── data/
│   ├── test/
│   ├── train/
│   ├── train_patient_held_out/
│   ├── train_deepfluoro_real/
│   ├── deepfluoro_real/
│   └── sweep/
├── analysis/
├── model_weight/
├── results/
└── visualizations/
```

## Active Code Paths

- Data preparation: [data/README.md](/home/yehyun/Landmark-based-2D-3D-Registration-Uncertainty/data/README.md)
- DRR Image Experiments: [src/train_patient_held_out/](/home/yehyun/Landmark-based-2D-3D-Registration-Uncertainty/src/train_patient_held_out/README.md)
- Fluoro Image Experiments: [src/train_deepfluoro_real/](/home/yehyun/Landmark-based-2D-3D-Registration-Uncertainty/src/train_deepfluoro_real/README.md)

## Prerequisites

- Python 3.10
- PyTorch with CUDA support
- DeepFluoro dataset archive

Example environment setup:

```bash
conda create -n landmark python=3.10 -y
conda activate landmark
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python h5py numpy nibabel tqdm scipy diffdrr albumentations wandb segmentation_models_pytorch scikit-learn seaborn matplotlib pandas
```

## Dataset And Setup

Download the DeepFluoro archive into `data/`, then run:

- `script/1_data.sh`

This prepares the extracted dataset and baseline DRRs.

## Reproduction Workflow

Use the canonical shell launchers in `script/`:

1. Data preparation
   - `script/1_data.sh`
2. Patient-held-out training
   - `script/2_train_patient_held_out.sh`
3. MC-dropout testing and filtering sweeps
   - `script/3_test_patient_held_out_dropout.sh`
4. Finetuning
   - `script/4_finetune_patient_held_out.sh`
5. Finetuned testing
   - `script/5_test_finetune*.sh`
6. Optional real-image experiments
   - `script/6_train_deepfluoro_real_manual*.sh`
   - `script/7_test_deepfluoro_real_manual.sh`

## Outputs

The main outputs are written under:
- `model_weight/`
- `results/`
- `visualizations/`

Patient-held-out test runs write CSV summaries under:
- `visualizations/patient_held_out/<run>/prediction_<...>/csv_results/`
- `visualizations/patient_held_out/<run>/prediction_<...>/final_results/`

## Analysis

Main aggregation and figure utilities are in:
- [analysis/](/home/yehyun/Landmark-based-2D-3D-Registration-Uncertainty/analysis)
- [src/sweep/main.py](/home/yehyun/Landmark-based-2D-3D-Registration-Uncertainty/src/sweep/main.py)
- [src/sweep/combine_patients.py](/home/yehyun/Landmark-based-2D-3D-Registration-Uncertainty/src/sweep/combine_patients.py)

Promoted patient-held-out analysis scripts include:
- `analysis/make_patient_held_out_figures.py`
- `analysis/make_patient_held_out_finetune_figures.py`
- `analysis/make_patient_held_out_finetune_grad_vs_nograd_figures.py`
- `analysis/make_patient_held_out_tables.py`

## More Detailed Docs

- Main held-out pipeline: [src/train_patient_held_out/README.md](/home/yehyun/Landmark-based-2D-3D-Registration-Uncertainty/src/train_patient_held_out/README.md)
- Optional real-image experiments: [src/train_deepfluoro_real/README.md](/home/yehyun/Landmark-based-2D-3D-Registration-Uncertainty/src/train_deepfluoro_real/README.md)

## Reference

If you found this code useful, please cite:

```bibtex
@inproceedings{suh2026landmark,
  title={Landmark Detection Uncertainty as a Reliability Weight for Robust Landmark-based 2D/3D Pelvic Pose Estimation},
  author={Suh, Yehyun and Schott, Brayden and Mo, Chou and Martin, John Ryan and Moyer, Daniel},
  booktitle={Medical Imaging with Deep Learning},
  year={2026}
}
```
