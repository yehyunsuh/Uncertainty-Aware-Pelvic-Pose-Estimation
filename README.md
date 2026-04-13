# Landmark-based-2D-3D-Registration-Uncertainty

## Introduction

An uncertainty-aware landmark-based 2D/3D pelvic registration framework that uses MC-dropout-derived reliability to improve pose estimation in both synthetic and real fluoroscopy.

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
│   └── deepfluoro_real/
├── analysis/
├── model_weight/
├── results/
└── visualizations/
```
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

## Dataset
Download the DeepFluoro archive into `data/` using
```
wget --no-check-certificate -O data/ipcai_2020_full_res_data.zip "http://archive.data.jhu.edu/api/access/datafile/:persistentId/?persistentId=doi:10.7281/T1/IFSXNV/EAN9GH"
```

## Reproduction Workflow

Use the canonical shell launchers in `script/`:

1. Data preparation
   - `script/1_data.sh`
3. Patient-held-out training
   - `script/2_train_patient_held_out.sh`
4. MC-dropout testing and filtering sweeps
   - `script/3_test_patient_held_out_dropout.sh`
5. Finetuning
   - `script/4_finetune_patient_held_out.sh`
6. Finetuned testing
   - `script/5_test_finetune*.sh`
7. Fluoroscopy image experiments
   - `script/6_train_deepfluoro_real_manual*.sh`
   - `script/7_test_deepfluoro_real_manual.sh`

## Outputs

The main outputs are written under:
- `model_weight/`
- `results/`
- `visualizations/`

Test runs write CSV summaries under:
- `visualizations/patient_held_out/<run>/prediction_<...>/csv_results/`
- `visualizations/patient_held_out/<run>/prediction_<...>/final_results/`

## Reference

If you found this code useful, please cite this [paper](https://openreview.net/pdf?id=t40yShfMhk):

```bibtex
@inproceedings{suh2026landmark,
  title={Landmark Detection Uncertainty as a Reliability Weight for Robust Landmark-based 2D/3D Pelvic Pose Estimation},
  author={Suh, Yehyun and Schott, Brayden and Mo, Chou and Martin, John Ryan and Moyer, Daniel},
  booktitle={Medical Imaging with Deep Learning},
  year={2026}
}
```
