python3 -m src.train_deepfluoro_real.main \
  --test_mode \
  --source_domain real \
  --init_mode synthetic \
  --specimen_id 17-1882 \
  --wandb_name 17-1882_real_from_synthetic_aug_geom_v1 \
  --checkpoint_metric dist

python3 -m src.train_deepfluoro_real.main \
  --test_mode \
  --source_domain real \
  --init_mode synthetic \
  --specimen_id 17-1905 \
  --wandb_name 17-1905_real_from_synthetic_aug_geom_v1 \
  --checkpoint_metric dist

python3 -m src.train_deepfluoro_real.main \
  --test_mode \
  --source_domain real \
  --init_mode synthetic \
  --specimen_id 18-0725 \
  --wandb_name 18-0725_real_from_synthetic_aug_geom_v1 \
  --checkpoint_metric dist

python3 -m src.train_deepfluoro_real.main \
  --test_mode \
  --source_domain real \
  --init_mode synthetic \
  --specimen_id 18-1109 \
  --wandb_name 18-1109_real_from_synthetic_aug_geom_v1 \
  --checkpoint_metric dist

python3 -m src.train_deepfluoro_real.main \
  --test_mode \
  --source_domain real \
  --init_mode synthetic \
  --specimen_id 18-2799 \
  --wandb_name 18-2799_real_from_synthetic_aug_geom_v1 \
  --checkpoint_metric dist

python3 -m src.train_deepfluoro_real.main \
  --test_mode \
  --source_domain real \
  --init_mode synthetic \
  --specimen_id 18-2800 \
  --wandb_name 18-2800_real_from_synthetic_aug_geom_v1 \
  --checkpoint_metric dist
