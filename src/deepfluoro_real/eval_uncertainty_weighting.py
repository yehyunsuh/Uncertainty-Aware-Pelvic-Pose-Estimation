from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

from src.deepfluoro_real.convention import camera_to_pelvis_to_world_to_camera
from src.deepfluoro_real.detector import (
    DetectorConfig,
    checkpoint_path_for_specimen,
    infer_landmarks,
    preprocess_raw_image,
)
from src.deepfluoro_real.io import list_projection_ids, list_specimen_ids, load_case
from src.deepfluoro_real.pose_estimation import (
    estimate_pose_from_correspondences,
    estimate_pose_from_correspondences_weighted,
    mtre_mm,
)
from src.test.model import UNet_with_dropout
from src.train.model import UNet


def _select_cases(
    h5_path: str,
    specimen_ids: list[str],
    projection_ids: list[str] | None,
    max_cases_per_specimen: int,
) -> list[tuple[str, str]]:
    selected = []
    for specimen_id in specimen_ids:
        all_projection_ids = projection_ids or list_projection_ids(h5_path, specimen_id)
        for projection_id in all_projection_ids[:max_cases_per_specimen]:
            selected.append((specimen_id, projection_id))
    return selected


def _build_model_args(cfg: DetectorConfig, dropout_rate: float | None = None) -> SimpleNamespace:
    args = SimpleNamespace(
        encoder_depth=cfg.encoder_depth,
        decoder_channels=list(cfg.decoder_channels),
        n_landmarks=cfg.n_landmarks,
    )
    if dropout_rate is not None:
        args.dropout_rate = dropout_rate
    return args


def _load_base_model(specimen_id: str, device: torch.device, cfg: DetectorConfig) -> torch.nn.Module:
    checkpoint_path = checkpoint_path_for_specimen(specimen_id, cfg)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {checkpoint_path}")
    model = UNet(_build_model_args(cfg), str(device))
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _load_dropout_model(
    specimen_id: str,
    device: torch.device,
    cfg: DetectorConfig,
    dropout_rate: float,
) -> torch.nn.Module:
    checkpoint_path = checkpoint_path_for_specimen(specimen_id, cfg)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {checkpoint_path}")
    model = UNet_with_dropout(_build_model_args(cfg, dropout_rate=dropout_rate), str(device))
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d)):
            module.train()
    return model


@torch.no_grad()
def _infer_mc_raw_coords(
    model: torch.nn.Module,
    prep,
    device: torch.device,
    n_simulations: int,
    sim_batch_size: int,
) -> np.ndarray:
    image = prep.image_tensor.to(device)
    sims_remaining = n_simulations
    all_coords = []
    while sims_remaining > 0:
        current_batch = min(sim_batch_size, sims_remaining)
        batch = image.repeat(current_batch, 1, 1, 1)
        logits = model(batch)
        probs = torch.sigmoid(logits)
        bsz, channels, _, width = probs.shape
        flat = probs.view(bsz, channels, -1)
        max_idx = flat.argmax(dim=2)

        coords_model = torch.zeros((bsz, channels, 2), device=device, dtype=torch.float32)
        for b in range(bsz):
            for c in range(channels):
                idx1d = int(max_idx[b, c].item())
                y, x = divmod(idx1d, width)
                coords_model[b, c] = torch.tensor([x, y], device=device, dtype=torch.float32)

        coords_model_np = coords_model.detach().cpu().numpy()
        coords_raw_np = coords_model_np.copy()
        if prep.apply_horizontal_flip:
            coords_raw_np[:, :, 0] = (prep.model_width - 1) - coords_raw_np[:, :, 0]
        coords_raw_np[:, :, 0] /= prep.scale_x
        coords_raw_np[:, :, 1] /= prep.scale_y
        if prep.apply_rot180:
            coords_raw_np[:, :, 0] = (prep.raw_width - 1) - coords_raw_np[:, :, 0]
            coords_raw_np[:, :, 1] = (prep.raw_height - 1) - coords_raw_np[:, :, 1]

        all_coords.append(coords_raw_np)
        sims_remaining -= current_batch

    return np.concatenate(all_coords, axis=0)


def _compute_deviation(mc_coords_raw: np.ndarray) -> np.ndarray:
    mean_pt = mc_coords_raw.mean(axis=0)
    diffs = mc_coords_raw - mean_pt[None, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    return np.sqrt(np.mean(sq_dists, axis=0)).astype(np.float64)


def _visible_gt_mask(case) -> np.ndarray:
    mask_x = (case.landmarks_2d[:, 0] >= 0.0) & (case.landmarks_2d[:, 0] < case.calibration.num_cols)
    mask_y = (case.landmarks_2d[:, 1] >= 0.0) & (case.landmarks_2d[:, 1] < case.calibration.num_rows)
    return mask_x & mask_y


def _filtered_mask(visible_mask: np.ndarray, deviation: np.ndarray, top_k: int) -> np.ndarray:
    filtered = visible_mask.copy()
    if top_k <= 0:
        return filtered
    valid_indices = np.where(visible_mask & np.isfinite(deviation))[0]
    if len(valid_indices) <= top_k:
        return filtered
    sorted_valid = valid_indices[np.argsort(deviation[valid_indices])]
    drop_indices = sorted_valid[-top_k:]
    filtered[drop_indices] = False
    return filtered


def _continuous_weights(deviation: np.ndarray, beta: float) -> np.ndarray:
    valid = np.isfinite(deviation)
    if not valid.any():
        return np.ones_like(deviation, dtype=np.float64)
    filled = deviation.copy()
    filled[~valid] = np.nanmedian(filled[valid])
    weights = np.exp(-beta * filled)
    max_val = float(np.max(weights))
    if max_val > 0:
        weights /= max_val
    return weights.astype(np.float64)


def _pose_summary(pose_values: list[float]) -> tuple[float | None, float | None, float | None]:
    if not pose_values:
        return None, None, None
    values = np.asarray(pose_values, dtype=np.float64)
    return float(values.mean()), float(np.median(values)), float(np.percentile(values, 95))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DS/CW uncertainty weighting on raw DeepFluoro images.")
    parser.add_argument("--h5_path", default="data/ipcai_2020_full_res_data.h5")
    parser.add_argument("--output_dir", default="visualizations/deepfluoro_real_uncertainty_weighting")
    parser.add_argument("--specimen_ids", nargs="+", default=["17-1882"])
    parser.add_argument("--projection_ids", nargs="*", default=None)
    parser.add_argument("--max_cases_per_specimen", type=int, default=2)
    parser.add_argument("--all_specimens", action="store_true")
    parser.add_argument("--all_projections", action="store_true")
    parser.add_argument("--image_resize", type=int, default=512)
    parser.add_argument("--n_landmarks", type=int, default=14)
    parser.add_argument("--encoder_depth", type=int, default=5)
    parser.add_argument("--decoder_channels", nargs="+", type=int, default=[256, 128, 64, 32, 16])
    parser.add_argument("--checkpoint_suffix", type=str, default="hard_dist")
    parser.add_argument("--model_weight_dir", type=str, default="model_weight")
    parser.add_argument("--model_type", type=str, default="patient_held_out")
    parser.add_argument("--intensity_mode", choices=["percentile_1_99", "minmax"], default="percentile_1_99")
    parser.add_argument("--apply_invert", action="store_true")
    parser.add_argument("--disable_horizontal_flip", action="store_true")
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--n_simulations", type=int, default=100)
    parser.add_argument("--sim_batch_size", type=int, default=25)
    parser.add_argument("--uncertainty_weight_beta", type=float, default=0.01)
    parser.add_argument("--k_values", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    specimen_ids = args.specimen_ids
    if args.all_specimens:
        specimen_ids = list_specimen_ids(args.h5_path)

    max_cases_per_specimen = args.max_cases_per_specimen
    if args.all_projections:
        max_cases_per_specimen = 10**9

    cfg = DetectorConfig(
        image_resize=args.image_resize,
        n_landmarks=args.n_landmarks,
        encoder_depth=args.encoder_depth,
        decoder_channels=tuple(args.decoder_channels),
        checkpoint_suffix=args.checkpoint_suffix,
        model_weight_dir=args.model_weight_dir,
        model_type=args.model_type,
        intensity_mode=args.intensity_mode,
        apply_invert=args.apply_invert,
        apply_horizontal_flip=not args.disable_horizontal_flip,
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    case_keys = _select_cases(
        h5_path=args.h5_path,
        specimen_ids=specimen_ids,
        projection_ids=args.projection_ids,
        max_cases_per_specimen=max_cases_per_specimen,
    )

    models_base: dict[str, torch.nn.Module] = {}
    models_dropout: dict[str, torch.nn.Module] = {}

    per_case_rows: list[dict] = []
    method_metrics = {
        ("no_weights", 0): {"rotation": [], "translation": [], "mtre": [], "success": 0},
    }
    for k in args.k_values:
        method_metrics[("discrete_selection", k)] = {"rotation": [], "translation": [], "mtre": [], "success": 0}
        method_metrics[("continuous_weighting", k)] = {"rotation": [], "translation": [], "mtre": [], "success": 0}

    for specimen_id, projection_id in tqdm(case_keys, desc="Uncertainty weighting"):
        case = load_case(args.h5_path, specimen_id, projection_id)

        if specimen_id not in models_base:
            models_base[specimen_id] = _load_base_model(specimen_id, device, cfg)
            models_dropout[specimen_id] = _load_dropout_model(specimen_id, device, cfg, args.dropout_rate)

        prep = preprocess_raw_image(case.image, cfg, apply_rot180=case.rot_180_for_up)
        detection = infer_landmarks(models_base[specimen_id], prep, device)
        mc_coords_raw = _infer_mc_raw_coords(
            models_dropout[specimen_id],
            prep,
            device,
            n_simulations=args.n_simulations,
            sim_batch_size=args.sim_batch_size,
        )

        deviation = _compute_deviation(mc_coords_raw)
        visible_mask = _visible_gt_mask(case)
        continuous_weights = _continuous_weights(deviation, args.uncertainty_weight_beta)
        gt_world_to_camera = case.calibration.extrinsic @ np.linalg.inv(case.cam_to_pelvis_vol)

        base_mask = visible_mask.copy()
        try:
            if int(base_mask.sum()) < 6:
                raise ValueError(f"fewer than 6 visible landmarks: {int(base_mask.sum())}")
            pose_no_weights = estimate_pose_from_correspondences(
                points_3d=case.landmarks_3d[base_mask],
                points_2d=detection.coords_raw[base_mask],
                intrinsic=case.calibration.intrinsic,
                dataset_extrinsic=case.calibration.extrinsic,
                gt_cam_to_pelvis_vol=case.cam_to_pelvis_vol,
            )
            pred_world_to_camera = camera_to_pelvis_to_world_to_camera(
                pose_no_weights.cam_to_pelvis_vol, case.calibration.extrinsic
            )
            mtre_eval = mtre_mm(case.landmarks_3d, gt_world_to_camera, pred_world_to_camera)
            method_metrics[("no_weights", 0)]["rotation"].append(float(pose_no_weights.rotation_diff_deg))
            method_metrics[("no_weights", 0)]["translation"].append(float(pose_no_weights.translation_diff_mm))
            method_metrics[("no_weights", 0)]["mtre"].append(float(mtre_eval))
            method_metrics[("no_weights", 0)]["success"] += 1
            per_case_rows.append(
                {
                    "specimen_id": specimen_id,
                    "projection_id": projection_id,
                    "method": "no_weights",
                    "k": 0,
                    "visible_landmarks": int(base_mask.sum()),
                    "rotation_diff_deg": float(pose_no_weights.rotation_diff_deg),
                    "translation_diff_mm": float(pose_no_weights.translation_diff_mm),
                    "mtre_mm": float(mtre_eval),
                }
            )
        except Exception:
            per_case_rows.append(
                {
                    "specimen_id": specimen_id,
                    "projection_id": projection_id,
                    "method": "no_weights",
                    "k": 0,
                    "visible_landmarks": int(base_mask.sum()),
                    "rotation_diff_deg": np.nan,
                    "translation_diff_mm": np.nan,
                    "mtre_mm": np.nan,
                }
            )

        for k in args.k_values:
            filtered = _filtered_mask(visible_mask, deviation, k)
            filtered_count = int(filtered.sum())

            try:
                if filtered_count < 6:
                    raise ValueError(f"fewer than 6 retained landmarks: {filtered_count}")
                pose_ds = estimate_pose_from_correspondences(
                    points_3d=case.landmarks_3d[filtered],
                    points_2d=detection.coords_raw[filtered],
                    intrinsic=case.calibration.intrinsic,
                    dataset_extrinsic=case.calibration.extrinsic,
                    gt_cam_to_pelvis_vol=case.cam_to_pelvis_vol,
                )
                pred_world_to_camera_ds = camera_to_pelvis_to_world_to_camera(
                    pose_ds.cam_to_pelvis_vol, case.calibration.extrinsic
                )
                mtre_eval_ds = mtre_mm(case.landmarks_3d, gt_world_to_camera, pred_world_to_camera_ds)
                method_metrics[("discrete_selection", k)]["rotation"].append(float(pose_ds.rotation_diff_deg))
                method_metrics[("discrete_selection", k)]["translation"].append(float(pose_ds.translation_diff_mm))
                method_metrics[("discrete_selection", k)]["mtre"].append(float(mtre_eval_ds))
                method_metrics[("discrete_selection", k)]["success"] += 1
                per_case_rows.append(
                    {
                        "specimen_id": specimen_id,
                        "projection_id": projection_id,
                        "method": "discrete_selection",
                        "k": int(k),
                        "visible_landmarks": int(visible_mask.sum()),
                        "retained_landmarks": filtered_count,
                        "rotation_diff_deg": float(pose_ds.rotation_diff_deg),
                        "translation_diff_mm": float(pose_ds.translation_diff_mm),
                        "mtre_mm": float(mtre_eval_ds),
                    }
                )
            except Exception:
                per_case_rows.append(
                    {
                        "specimen_id": specimen_id,
                        "projection_id": projection_id,
                        "method": "discrete_selection",
                        "k": int(k),
                        "visible_landmarks": int(visible_mask.sum()),
                        "retained_landmarks": filtered_count,
                        "rotation_diff_deg": np.nan,
                        "translation_diff_mm": np.nan,
                        "mtre_mm": np.nan,
                    }
                )

            try:
                if filtered_count < 6:
                    raise ValueError(f"fewer than 6 retained landmarks: {filtered_count}")
                pose_cw = estimate_pose_from_correspondences_weighted(
                    points_3d=case.landmarks_3d[filtered],
                    points_2d=detection.coords_raw[filtered],
                    weights=continuous_weights[filtered],
                    intrinsic=case.calibration.intrinsic,
                    dataset_extrinsic=case.calibration.extrinsic,
                    gt_cam_to_pelvis_vol=case.cam_to_pelvis_vol,
                )
                pred_world_to_camera_cw = camera_to_pelvis_to_world_to_camera(
                    pose_cw.cam_to_pelvis_vol, case.calibration.extrinsic
                )
                mtre_eval_cw = mtre_mm(case.landmarks_3d, gt_world_to_camera, pred_world_to_camera_cw)
                method_metrics[("continuous_weighting", k)]["rotation"].append(float(pose_cw.rotation_diff_deg))
                method_metrics[("continuous_weighting", k)]["translation"].append(float(pose_cw.translation_diff_mm))
                method_metrics[("continuous_weighting", k)]["mtre"].append(float(mtre_eval_cw))
                method_metrics[("continuous_weighting", k)]["success"] += 1
                per_case_rows.append(
                    {
                        "specimen_id": specimen_id,
                        "projection_id": projection_id,
                        "method": "continuous_weighting",
                        "k": int(k),
                        "visible_landmarks": int(visible_mask.sum()),
                        "retained_landmarks": filtered_count,
                        "rotation_diff_deg": float(pose_cw.rotation_diff_deg),
                        "translation_diff_mm": float(pose_cw.translation_diff_mm),
                        "mtre_mm": float(mtre_eval_cw),
                    }
                )
            except Exception:
                per_case_rows.append(
                    {
                        "specimen_id": specimen_id,
                        "projection_id": projection_id,
                        "method": "continuous_weighting",
                        "k": int(k),
                        "visible_landmarks": int(visible_mask.sum()),
                        "retained_landmarks": filtered_count,
                        "rotation_diff_deg": np.nan,
                        "translation_diff_mm": np.nan,
                        "mtre_mm": np.nan,
                    }
                )

    case_csv = output_root / "per_case_results.csv"
    with case_csv.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = sorted({key for row in per_case_rows for key in row.keys()})
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_case_rows)

    summary_rows = []
    for (method, k), metrics in method_metrics.items():
        rot_mean, rot_median, _ = _pose_summary(metrics["rotation"])
        trans_mean, trans_median, _ = _pose_summary(metrics["translation"])
        mtre_mean, mtre_median, mtre_p95 = _pose_summary(metrics["mtre"])
        summary_rows.append(
            {
                "method": method,
                "k": int(k),
                "num_cases": len(case_keys),
                "success_cases": int(metrics["success"]),
                "mean_rotation_diff_deg": rot_mean,
                "median_rotation_diff_deg": rot_median,
                "mean_translation_diff_mm": trans_mean,
                "median_translation_diff_mm": trans_median,
                "mean_mtre_mm": mtre_mean,
                "median_mtre_mm": mtre_median,
                "p95_mtre_mm": mtre_p95,
            }
        )

    summary_rows.sort(key=lambda row: (row["method"], row["k"]))
    summary_csv = output_root / "summary_by_method_k.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with (output_root / "summary_by_method_k.json").open("w", encoding="utf-8") as json_file:
        json.dump(summary_rows, json_file, indent=2)


if __name__ == "__main__":
    main()
