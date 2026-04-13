# DeepFluoro Real Debug Status

This note records the current status of the raw DeepFluoro real-image path implemented under `src/deepfluoro_real/`.

The raw-geometry debugging stage is complete. The repository now also contains a first predicted-landmark baseline on raw DeepFluoro fluoroscopy images. The raw pose solver remains the same validated solver used in the GT-correspondence stage.

## Scope

The current code covers:

1. raw-geometry reprojection sanity checking
2. raw-geometry pose-recovery sanity checking
3. full-dataset GT-correspondence verification across all real projections
4. detector inference on raw fluoroscopy images
5. explicit detector-to-raw-coordinate conversion
6. predicted-landmark registration baseline on the full real dataset

It now **does** run the trained landmark detector on raw fluoroscopy images, but only as a first no-weight/no-filter baseline.

## Raw DeepFluoro Convention Used

The current implementation uses the raw HDF5 data directly from:

- `data/ipcai_2020_full_res_data.h5`

The main convention choices are:

- raw 3D target points are read from `vol-landmarks`
- raw 2D target points are read from `gt-landmarks`
- raw pelvis pose is read from `gt-poses/cam-to-pelvis-vol`
- global calibration is read from `proj-params/intrinsic` and `proj-params/extrinsic`
- `rot-180-for-up` is respected for **display orientation**, but reprojection error is computed in the raw detector coordinate system

These are implemented in:

- `src/deepfluoro_real/io.py`
- `src/deepfluoro_real/convention.py`
- `src/deepfluoro_real/projection.py`

## Verified Projection Formula

The raw HDF5 projection convention was verified numerically.

The working world-to-camera transform for projection is:

- `world_to_camera = proj_params.extrinsic @ inv(cam_to_pelvis_vol)`

The projection matrix is then:

- `P = K @ world_to_camera[:3, :]`

This is implemented in:

- `src/deepfluoro_real/convention.py`
  - `camera_to_pelvis_to_world_to_camera()`
- `src/deepfluoro_real/projection.py`
  - `projection_matrix()`
  - `project_points()`

This matches the raw DeepFluoro pinhole projection behavior and avoids the synthetic repo convention that uses depth along `y`.

## Image And Detector Orientation

The code distinguishes between:

- **raw detector coordinates** for numeric reprojection and pose recovery
- **upright display orientation** for saved overlay images

If `rot-180-for-up` is set:

- the saved overlay image is rotated by 180 degrees
- the overlay points are rotated by 180 degrees
- the underlying reprojection metric is still computed against the raw stored 2D landmarks

This is implemented in:

- `src/deepfluoro_real/convention.py`
  - `rotate_image_for_upright_display()`
  - `rotate_points_for_upright_display()`
- `src/deepfluoro_real/debug_one_case.py`
  - `run_case()`

## Pose Recovery Path

The new pose recovery path is separate from the existing synthetic solver.

It currently does:

1. DLT initialization from raw 2D-3D correspondences
2. nonlinear reprojection refinement with `scipy.optimize.least_squares`
3. comparison of recovered pose against the stored raw HDF5 pelvis pose
4. mTRE computation over the raw H5 `vol-landmarks`

This is implemented in:

- `src/deepfluoro_real/pose_estimation.py`
  - `estimate_world_to_camera_dlt()`
  - `refine_world_to_camera()`
  - `estimate_pose_from_correspondences()`
  - `mtre_mm()`

## Detector Preprocessing Audit

The held-out detector path was traced before adding the real-image pipeline.

The detector uses:

- checkpoint format: plain PyTorch `state_dict`
- architecture: `smp.Unet` with `resnet101` encoder and `in_channels=3`
- detector preprocessing:
  - read image with OpenCV
  - convert `BGR -> RGB`
  - `Resize(512, 512)`
  - `Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))`
  - `InvertImg(p=1.0)`
  - `ToTensorV2()`
- decoding:
  - apply `sigmoid`
  - take heatmap argmax
  - decode to `(x, y)` coordinates in resized model space

These are implemented in:

- `src/train/model.py`
  - `UNet()`
- `src/train_patient_held_out/main.py`
  - `landmark_prediction_train()`
- `src/train_patient_held_out/data_loader.py`
  - `SegmentationDataset.__getitem__()`
- `src/train_patient_held_out/test.py`
  - `test_model()`

Important training detail:

- the standard detector training path also used `VerticalFlip(p=0.3)`
- validation in that path also receives the same flip because of the current loader condition

## Real-Image Detector Bridge

The real-image detector bridge is implemented in:

- `src/deepfluoro_real/detector.py`
  - `load_detector_model()`
  - `normalize_raw_image_to_uint8()`
  - `preprocess_raw_image()`
  - `raw_to_model_coords()`
  - `model_to_raw_coords()`
  - `infer_landmarks()`

The bridge keeps the detector preprocessing explicit:

1. raw DeepFluoro fluoroscopy image is loaded from the HDF5 file
2. raw float image is converted to `uint8` using percentile windowing
3. grayscale image is converted to 3-channel RGB
4. the original detector preprocessing is applied
5. detector outputs are decoded in resized model space
6. decoded coordinates are mapped back to raw detector coordinates by explicit linear scaling

The current default intensity conversion is:

- `percentile_1_99`

This was chosen as the simplest possible baseline to reuse the synthetic detector on real images without modifying the validated raw solver.

## Predicted-Landmark Evaluation Path

The predicted-landmark real-image baseline is implemented in:

- `src/deepfluoro_real/eval_predicted_landmarks.py`
  - `_save_bridge_figure()`
  - `_write_predicted_landmarks_csv()`
  - `_save_pose_overlay()`
  - `_summarize_results()`
  - `main()`

Per case, this path computes:

- predicted landmarks in model space
- predicted landmarks mapped back to raw detector coordinates
- GT landmark pixel error in raw detector space
- GT upper-bound pose result
- predicted-landmark pose result using the validated `estimate_pose_from_correspondences()`
- runtime for preprocessing, detector inference, pose solve, and total predicted pipeline

The raw solver itself remains in:

- `src/deepfluoro_real/pose_estimation.py`
  - `estimate_pose_from_correspondences()`

and was not modified for the predicted-landmark baseline.

## Current Outputs

### GT-Correspondence Path

The GT debug runner saves, per case:

- reprojection overlay image
- per-landmark reprojection error CSV
- summary JSON
- recovered pose JSON

It also saves aggregate outputs:

- `cases.csv`
- `per_specimen_summary.json`
- `overall_summary.json`
- `overall_summary.png`

This is implemented in:

- `src/deepfluoro_real/debug_one_case.py`
  - `run_case()`
  - `summarize_cases()`

### Predicted-Landmark Path

The predicted-landmark runner saves, per case when artifacts are enabled:

- `bridge_overlay.png`
- `predicted_landmarks.csv`
- `predicted_pose_overlay.png`
- `gt_pose_result.json`
- `pred_pose_result.json`
- `summary.json`

It also saves aggregate outputs:

- `cases.csv`
- `per_specimen_summary.json`
- `overall_summary.json`
- `overall_summary.png`

## Commands Verified

Single-case GT debug:

```bash
script/debug_deepfluoro_real.sh
```

Equivalent direct command:

```bash
python3 -m src.deepfluoro_real.debug_one_case \
  --specimen_ids 17-1882 \
  --projection_ids 000 \
  --output_dir visualizations/deepfluoro_real_debug_single
```

Small multi-case validation:

```bash
python3 -m src.deepfluoro_real.debug_one_case \
  --specimen_ids 17-1882 17-1905 18-0725 \
  --max_cases_per_specimen 2 \
  --output_dir visualizations/deepfluoro_real_debug_multi
```

Full GT-correspondence validation:

```bash
python3 -m src.deepfluoro_real.debug_one_case \
  --all_specimens \
  --all_projections \
  --output_dir visualizations/deepfluoro_real_debug_all
```

Few-case predicted-landmark validation:

```bash
scripts/debug_deepfluoro_real_predicted_few.sh
```

Equivalent direct command:

```bash
conda run -n landmark python -m src.deepfluoro_real.eval_predicted_landmarks \
  --all_specimens \
  --max_cases_per_specimen 2 \
  --output_dir visualizations/deepfluoro_real_predicted_few
```

Full-dataset predicted-landmark baseline:

```bash
scripts/debug_deepfluoro_real_predicted_all.sh
```

Equivalent direct command:

```bash
conda run -n landmark python -m src.deepfluoro_real.eval_predicted_landmarks \
  --all_specimens \
  --all_projections \
  --skip_case_artifacts \
  --output_dir visualizations/deepfluoro_real_predicted_all
```

## Current Verified Results

### Single Case

For:

- `17-1882 / 000`

the saved summary showed:

- GT reprojection mean error: about `0.00027 px`
- GT reprojection median error: about `0.00030 px`
- recovered pose rotation difference: `0.0 deg`
- recovered pose translation difference: about `3.8e-05 mm`
- recovered pose mTRE: about `3.8e-05 mm`

Output files:

- `visualizations/deepfluoro_real_debug_single/17-1882/000/reprojection_overlay.png`
- `visualizations/deepfluoro_real_debug_single/17-1882/000/summary.json`
- `visualizations/deepfluoro_real_debug_single/17-1882/000/pose_recovery.json`

### Multi-Case Check

For:

- `17-1882`
- `17-1905`
- `18-0725`

with `2` projections per specimen, the aggregate summary showed:

- `num_cases = 6`
- `valid_cases = 6`
- `invalid_cases = 0`
- mean GT reprojection error: about `0.00040 px`
- mean recovered-pose mTRE: about `0.00013 mm`
- mean recovered-pose rotation difference: about `0.0030 deg`
- mean recovered-pose translation difference: about `0.00013 mm`

Aggregate files:

- `visualizations/deepfluoro_real_debug_multi/_aggregate/cases.csv`
- `visualizations/deepfluoro_real_debug_multi/_aggregate/per_specimen_summary.json`
- `visualizations/deepfluoro_real_debug_multi/_aggregate/overall_summary.json`

### Full GT-Correspondence Validation

For all real DeepFluoro projections:

- `num_cases = 366`
- `valid_cases = 366`
- `invalid_cases = 0`
- mean GT reprojection error: about `0.00057 px`
- mean recovered-pose mTRE: about `0.00014 mm`
- mean recovered-pose rotation difference: about `0.00067 deg`
- mean recovered-pose translation difference: about `0.00014 mm`

Aggregate files:

- `visualizations/deepfluoro_real_debug_all/_aggregate/cases.csv`
- `visualizations/deepfluoro_real_debug_all/_aggregate/per_specimen_summary.json`
- `visualizations/deepfluoro_real_debug_all/_aggregate/overall_summary.json`
- `visualizations/deepfluoro_real_debug_all/_aggregate/overall_summary.png`

This validates the raw geometry conventions and the raw solver on GT 2D-3D correspondences across the full dataset.

### Few-Case Predicted-Landmark Validation

For `2` projections per specimen (`12` cases total), the few-case predicted-landmark baseline showed:

- `valid_pred_cases = 12`
- `invalid_pred_cases = 0`
- mean landmark error: about `995 px`
- mean predicted-pose mTRE: about `1439 mm`
- mean predicted-pose rotation difference: about `110 deg`
- mean predicted-pose translation difference: about `1448 mm`

Representative bridge output:

- `visualizations/deepfluoro_real_predicted_few/17-1882/000/bridge_overlay.png`

Interpretation:

- the detector-to-raw-coordinate bridge appears geometrically coherent
- the detector predictions themselves are poor on raw fluoroscopy

### Full-Dataset Predicted-Landmark Baseline

For all `366` real DeepFluoro projections with the no-weight/no-filter baseline:

- `valid_pred_cases = 366`
- `invalid_pred_cases = 0`
- mean predicted landmark error: about `919 px`
- median predicted landmark error: about `980 px`
- mean predicted-pose mTRE: about `1261 mm`
- median predicted-pose mTRE: about `1366 mm`
- 95th percentile predicted-pose mTRE: about `3789 mm`
- mean predicted-pose rotation difference: about `120 deg`
- mean predicted-pose translation difference: about `1291 mm`

Runtime summary per image:

- preprocessing: about `39.0 ms`
- detector inference: about `237.2 ms`
- predicted pose solve: about `12.5 ms`
- total predicted pipeline: about `288.8 ms`

Aggregate files:

- `visualizations/deepfluoro_real_predicted_all/_aggregate/cases.csv`
- `visualizations/deepfluoro_real_predicted_all/_aggregate/per_specimen_summary.json`
- `visualizations/deepfluoro_real_predicted_all/_aggregate/overall_summary.json`
- `visualizations/deepfluoro_real_predicted_all/_aggregate/overall_summary.png`

This is the first complete predicted-landmark real-image baseline. It is numerically complete, but registration quality is poor because the detector does not transfer well from synthetic DRRs to raw fluoroscopy.

## What Has Been Verified

At this point, the following are verified:

- raw DeepFluoro calibration can be used directly for projection
- raw stored pelvis pose is geometrically consistent with the stored raw 2D landmarks
- the projection direction and transform composition are now traced and implemented safely
- detector upright rotation can be handled as a display concern without corrupting raw reprojection math
- pose can be recovered back from raw 2D-3D correspondences with negligible error on the full dataset
- mTRE can be computed safely from raw 3D target points in the raw geometry path
- detector preprocessing, checkpoint format, decoding, and output coordinate convention have been traced from the held-out inference path
- predicted detector coordinates can be mapped back into raw DeepFluoro detector coordinates explicitly and reproducibly
- the first no-weight/no-filter predicted-landmark baseline has been executed on all 366 real projections
- the main failure mode is currently detector domain transfer, not the raw solver

## What Is Not Implemented Yet

The following steps are still pending:

- confidence-thresholded predicted-landmark pose solving on the raw-image path
- top-k landmark dropping / DS on the raw-image path
- comparing uncertainty-aware weighting variants on real images
- testing alternative detector checkpoints, including finetuned variants, on the same real-image path
- testing alternative real-image preprocessing / intensity conversion strategies
- producing paper-ready real-image benchmark tables

## Recommended Next Step

The next implementation step should be:

1. keep the current raw solver unchanged
2. improve detector-to-real transfer first, because the current no-weight baseline has very large landmark error
3. test one or more finetuned detector checkpoints on the same raw-image path
4. then add confidence thresholding / DS / CW variants on top of the same validated coordinate bridge

## Next Experiments

The recommended experiment order from this point is:

1. compare the current base held-out checkpoints against the available finetuned detector checkpoints on the same real-image path
2. if none of the existing checkpoints transfer adequately, train or adapt a detector specifically for raw real fluoroscopy images
3. once landmark quality is materially improved, add confidence thresholding on the current raw-image baseline
4. then add top-k landmark dropping / DS
5. then add continuous weighting / CW if the landmark confidence / uncertainty signal is stable enough to support it
6. only after the detector quality is acceptable, produce benchmark tables and figures intended for paper use

## Current Practical Conclusion

The current evidence supports the following conclusion:

- the raw DeepFluoro geometry path is validated
- the raw pose solver is not the main bottleneck
- the current synthetic-trained detector does not transfer well enough to raw fluoroscopy images
- for the real-image setting, a new detector or a detector adapted specifically to real fluoroscopy is likely required

In other words, the next major improvement should target landmark detection on real images, not a rewrite of the validated raw pose solver.

## Related Notes

For dataset-side calibration and frame-convention notes traced earlier, also see:

- `data/mTRE.md`
