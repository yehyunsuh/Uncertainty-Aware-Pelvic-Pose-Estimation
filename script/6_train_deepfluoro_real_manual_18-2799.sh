python3 -m src.train_deepfluoro_real.main \
  --train_mode \
  --preprocess \
  --source_domain synthetic \
  --init_mode imagenet \
  --specimen_id 18-2799 \
  --wandb \
  --wandb_project "DeepFluoro Synthetic Held-Out" \
  --wandb_name 18-2799_synthetic_imagenet

python3 -m src.train_deepfluoro_real.main \
  --train_mode \
  --preprocess \
  --source_domain real \
  --init_mode synthetic \
  --specimen_id 18-2799 \
  --synthetic_weight_path model_weight/train_deepfluoro_real/18-2799_synthetic_imagenet_dist.pth \
  --wandb \
  --wandb_project "DeepFluoro Real Held-Out" \
  --wandb_name 18-2799_real_from_synthetic_aug_geom_v1

python3  -m src.train_deepfluoro_real.main \
  --train_mode \
  --preprocess \
  --source_domain real \
  --init_mode imagenet \
  --specimen_id 18-2799 \
  --wandb \
  --wandb_project "DeepFluoro Real Held-Out" \
  --wandb_name 18-2799_real_imagenet
