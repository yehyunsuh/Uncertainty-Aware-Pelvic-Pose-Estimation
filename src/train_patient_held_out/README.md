# Patient-Held-Out Pipeline

This is the primary paper-aligned implementation.

Main entrypoint:

```bash
python -m src.train_patient_held_out.main
```

## Supported Modes

- `--train_mode`: supervised landmark detector training
- `--finetune_mode`: pose-aware finetuning from a trained detector
- `--test_mode`: deterministic prediction, MC-dropout uncertainty, result export, and summary generation

## Inputs

Expected data for each specimen under `data/DeepFluoro/<specimen_id>/`:
- `<specimen_id>_CT.nii.gz`
- `<specimen_id>_Landmarks_3D.npy`
- generated DRR projection folder
- generated DRR landmark CSV folder
- generated DRR pose-parameter CSV

The preprocessing stage writes split CSVs under:
- `landmark_prediction_csv/patient_held_out/`

## Core Files

- `main.py`: CLI and mode dispatch
- `data_loader.py`: held-out split construction and DRR loading
- `train.py`: supervised heatmap training
- `test.py`: deterministic testing, MC-dropout sampling, CSV export, summary export
- `finetune.py`: differentiable pose-aware finetuning
- `log.py`: W&B integration

## Standard Training Convention

Use run names such as:
- `17-1882`
- `17-1905`
- `18-0725`
- `18-1109`
- `18-2799`
- `18-2800`

This keeps outputs aligned with the existing `results/` and `visualizations/` layout.

## Standard Test Outputs

Under:

```text
visualizations/patient_held_out/<run>/prediction_<dropout>_<model_weight_name>*/
```

the code writes:
- `csv_results/ground_truth.csv`
- `csv_results/predictions_<dropout>.csv`
- `csv_results/mc_predictions_<dropout>.csv`
- `csv_results/final_clustered_results.csv`
- `final_results/test_results_summary_<k>_perImage_<visibility>.csv`
- `final_results/run_summary_<suffix>.csv`

## Canonical Script References

- `script/1_data.sh`
- `script/2_train_patient_held_out*.sh`
- `script/3_test_patient_held_out_dropout.sh`
- `script/3_test_patient_held_out_landmark*.sh`
- `script/4_finetune_patient_held_out.sh`
- `script/5_test_finetune*.sh`

## Important Reproduction Constraints

- Use the patient-held-out specimen IDs listed above
- Train one model per held-out specimen
- Run test mode after training to produce the CSV artifacts required by downstream aggregation
