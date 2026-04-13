# DeepFluoro Real-Image Experiments

This package is optional and is not required to reproduce the original paper-aligned patient-held-out DRR pipeline.

It supports newer real fluoroscopy experiments described in `2026_MIDL_Landmark_Prediction_Uncertainty (2).pdf`.

Main entrypoint:

```bash
python -m src.train_deepfluoro_real.main
```

## Supported Regimes

- `--source_domain synthetic --init_mode imagenet`
  - train a held-out synthetic DRR detector from ImageNet initialization
- `--source_domain real --init_mode synthetic`
  - finetune on real DeepFluoro starting from a newly trained synthetic detector
- `--source_domain real --init_mode imagenet`
  - direct real-image baseline

## Important Files

- `main.py`: CLI and specimen-wise dispatch
- `data_loader.py`: real-image manifest generation and domain-aware loading
- `train.py`: supervised training loop
- `test.py`: held-out evaluation on real projections

Related raw-geometry and evaluation utilities live in:
- `src/deepfluoro_real/`

## Typical Commands

Train synthetic held-out model:

```bash
python -m src.train_deepfluoro_real.main --train_mode --preprocess \
  --source_domain synthetic --init_mode imagenet \
  --specimen_id 17-1882 --wandb_name 17-1882_synthetic_imagenet
```

Finetune synthetic to real:

```bash
python -m src.train_deepfluoro_real.main --train_mode --preprocess \
  --source_domain real --init_mode synthetic \
  --specimen_id 17-1882 \
  --synthetic_weight_path model_weight/train_deepfluoro_real/17-1882_synthetic_imagenet_dist.pth \
  --wandb_name 17-1882_real_from_synthetic
```

Direct ImageNet to real baseline:

```bash
python -m src.train_deepfluoro_real.main --train_mode --preprocess \
  --source_domain real --init_mode imagenet \
  --specimen_id 17-1882 --wandb_name 17-1882_real_imagenet
```

## Canonical Script References

- `script/6_train_deepfluoro_real_manual*.sh`
- `script/7_test_deepfluoro_real_manual.sh`

## Notes

- This path uses real projection images from `gt_projections/` and JSON landmarks from `gt_landmarks_2D/`.
- It is maintained separately from the original patient-held-out synthetic DRR paper path.
