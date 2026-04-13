import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.deepfluoro_real.io import load_case
from src.deepfluoro_real.pose_estimation import (
    estimate_pose_from_correspondences,
    estimate_pose_from_correspondences_weighted,
)
from src.train_deepfluoro_real.data_loader import LANDMARK_NAMES, dataloader


def _coerce_landmarks_tensor(landmarks) -> torch.Tensor:
    if isinstance(landmarks, torch.Tensor):
        gt_coords = landmarks.detach().clone().float()
    else:
        gt_coords = torch.as_tensor(landmarks, dtype=torch.float32)

    if gt_coords.ndim == 2:
        gt_coords = gt_coords.unsqueeze(0)

    return gt_coords


def _unwrap_singleton(value):
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _raw_coords_from_model(coords: np.ndarray, image_width: int, image_height: int, image_resize: int) -> np.ndarray:
    scale_x = image_width / float(image_resize)
    scale_y = image_height / float(image_resize)
    raw = coords.astype(np.float32).copy()
    raw[:, 0] *= scale_x
    raw[:, 1] *= scale_y
    return raw


def _projection_id_from_image_name(image_name: str) -> str:
    return Path(image_name).stem.split("_")[-1]


def _pose_result_or_nan() -> dict[str, object]:
    return {
        "pose_success": False,
        "pose_message": "not_run",
        "pose_rotation_diff_deg": np.nan,
        "pose_translation_diff_mm": np.nan,
        "pose_mtre_mm": np.nan,
        "pose_reprojection_error_mean_px": np.nan,
        "pose_reprojection_error_median_px": np.nan,
        "weighted_pose_success": False,
        "weighted_pose_message": "not_run",
        "weighted_pose_rotation_diff_deg": np.nan,
        "weighted_pose_translation_diff_mm": np.nan,
        "weighted_pose_mtre_mm": np.nan,
        "weighted_pose_reprojection_error_mean_px": np.nan,
        "weighted_pose_reprojection_error_median_px": np.nan,
    }


def evaluate_test_set(args, model, device):
    test_loader = dataloader(args, data_type="test")
    model.eval()

    image_rows = []
    landmark_rows = []
    pose_rows = []
    all_dists = []
    all_raw_dists = []

    with torch.no_grad():
        for image, specimen_id, image_name, landmarks in tqdm(test_loader, desc="Testing"):
            image = image.to(device)
            gt_coords = _coerce_landmarks_tensor(landmarks).to(device)

            zero_mask = (gt_coords == 0).all(dim=2)
            gt_coords[zero_mask] = float("nan")

            outputs = model(image)
            probs = torch.sigmoid(outputs)
            batch_size, channels, _, width = probs.shape
            probs_flat = probs.view(batch_size, channels, -1)
            max_indices = probs_flat.argmax(dim=2)

            pred_coords = torch.zeros((batch_size, channels, 2), device=device)
            max_scores = torch.gather(probs_flat, 2, max_indices.unsqueeze(2)).squeeze(2)

            for b in range(batch_size):
                for c in range(channels):
                    index = max_indices[b, c].item()
                    y, x = divmod(index, width)
                    pred_coords[b, c] = torch.tensor([x, y], device=device)

            dists = torch.norm(pred_coords - gt_coords, dim=2)
            valid_mask = ~torch.isnan(gt_coords).any(dim=2)
            dists[~valid_mask] = float("nan")
            all_dists.append(dists.cpu())

            specimen_str = str(_unwrap_singleton(specimen_id))
            image_name_str = str(_unwrap_singleton(image_name))
            projection_id = _projection_id_from_image_name(image_name_str)

            case = load_case(args.h5_path, specimen_str, projection_id)
            gt_coords_np = gt_coords[0].detach().cpu().numpy()
            pred_coords_np = pred_coords[0].detach().cpu().numpy()
            max_scores_np = max_scores[0].detach().cpu().numpy()

            gt_coords_raw = _raw_coords_from_model(
                gt_coords_np,
                image_width=case.calibration.num_cols,
                image_height=case.calibration.num_rows,
                image_resize=args.image_resize,
            )
            pred_coords_raw = _raw_coords_from_model(
                pred_coords_np,
                image_width=case.calibration.num_cols,
                image_height=case.calibration.num_rows,
                image_resize=args.image_resize,
            )

            raw_dists = np.linalg.norm(pred_coords_raw - gt_coords_raw, axis=1)
            raw_dists[~valid_mask[0].detach().cpu().numpy()] = np.nan
            all_raw_dists.append(torch.from_numpy(raw_dists))

            valid_np = valid_mask[0].detach().cpu().numpy()
            pose_result = _pose_result_or_nan()
            num_pose_points = int(valid_np.sum())

            if num_pose_points >= 6:
                try:
                    pose_estimate = estimate_pose_from_correspondences(
                        points_3d=case.landmarks_3d[valid_np],
                        points_2d=pred_coords_raw[valid_np].astype(np.float64),
                        intrinsic=case.calibration.intrinsic,
                        dataset_extrinsic=case.calibration.extrinsic,
                        gt_cam_to_pelvis_vol=case.cam_to_pelvis_vol,
                    )
                    pose_result.update(
                        {
                            "pose_success": bool(pose_estimate.optimization_success),
                            "pose_message": str(pose_estimate.optimization_message),
                            "pose_rotation_diff_deg": float(pose_estimate.rotation_diff_deg),
                            "pose_translation_diff_mm": float(pose_estimate.translation_diff_mm),
                            "pose_mtre_mm": float(pose_estimate.mtre_mm),
                            "pose_reprojection_error_mean_px": float(
                                pose_estimate.reprojection_error_mean_px
                            ),
                            "pose_reprojection_error_median_px": float(
                                pose_estimate.reprojection_error_median_px
                            ),
                        }
                    )
                except Exception as exc:
                    pose_result["pose_message"] = str(exc)

                try:
                    weighted_pose = estimate_pose_from_correspondences_weighted(
                        points_3d=case.landmarks_3d[valid_np],
                        points_2d=pred_coords_raw[valid_np].astype(np.float64),
                        weights=max_scores_np[valid_np].astype(np.float64),
                        intrinsic=case.calibration.intrinsic,
                        dataset_extrinsic=case.calibration.extrinsic,
                        gt_cam_to_pelvis_vol=case.cam_to_pelvis_vol,
                    )
                    pose_result.update(
                        {
                            "weighted_pose_success": bool(weighted_pose.optimization_success),
                            "weighted_pose_message": str(weighted_pose.optimization_message),
                            "weighted_pose_rotation_diff_deg": float(
                                weighted_pose.rotation_diff_deg
                            ),
                            "weighted_pose_translation_diff_mm": float(
                                weighted_pose.translation_diff_mm
                            ),
                            "weighted_pose_mtre_mm": float(weighted_pose.mtre_mm),
                            "weighted_pose_reprojection_error_mean_px": float(
                                weighted_pose.reprojection_error_mean_px
                            ),
                            "weighted_pose_reprojection_error_median_px": float(
                                weighted_pose.reprojection_error_median_px
                            ),
                        }
                    )
                except Exception as exc:
                    pose_result["weighted_pose_message"] = str(exc)
            else:
                pose_result["pose_message"] = "fewer_than_6_visible_landmarks"
                pose_result["weighted_pose_message"] = "fewer_than_6_visible_landmarks"

            image_rows.append(
                {
                    "specimen_id": specimen_str,
                    "image_name": image_name_str,
                    "projection_id": projection_id,
                    "mean_dist": torch.nanmean(dists[0]).item(),
                    "mean_dist_raw": float(np.nanmean(raw_dists)),
                    "n_visible_landmarks": int(valid_mask[0].sum().item()),
                }
            )
            pose_rows.append(
                {
                    "specimen_id": specimen_str,
                    "image_name": image_name_str,
                    "projection_id": projection_id,
                    "n_visible_landmarks": num_pose_points,
                    **pose_result,
                }
            )

            for landmark_idx in range(channels):
                gt_x, gt_y = gt_coords[0, landmark_idx].detach().cpu().tolist()
                pred_x, pred_y = pred_coords[0, landmark_idx].detach().cpu().tolist()
                gt_x_raw, gt_y_raw = gt_coords_raw[landmark_idx].tolist()
                pred_x_raw, pred_y_raw = pred_coords_raw[landmark_idx].tolist()
                landmark_rows.append(
                    {
                        "specimen_id": specimen_str,
                        "image_name": image_name_str,
                        "projection_id": projection_id,
                        "landmark_index": landmark_idx,
                        "landmark_name": LANDMARK_NAMES[landmark_idx],
                        "gt_x": gt_x,
                        "gt_y": gt_y,
                        "pred_x": pred_x,
                        "pred_y": pred_y,
                        "gt_x_raw": gt_x_raw,
                        "gt_y_raw": gt_y_raw,
                        "pred_x_raw": pred_x_raw,
                        "pred_y_raw": pred_y_raw,
                        "max_score": max_scores[0, landmark_idx].item(),
                        "dist": dists[0, landmark_idx].item(),
                        "dist_raw": float(raw_dists[landmark_idx]),
                        "visible": bool(valid_mask[0, landmark_idx].item()),
                    }
                )

    dist_tensor = torch.cat(all_dists, dim=0) if all_dists else torch.empty((0, args.n_landmarks))
    raw_dist_tensor = torch.stack(all_raw_dists, dim=0) if all_raw_dists else torch.empty((0, args.n_landmarks))
    valid_distances = dist_tensor[~torch.isnan(dist_tensor)] if dist_tensor.numel() else torch.empty(0)
    valid_raw_distances = (
        raw_dist_tensor[~torch.isnan(raw_dist_tensor)] if raw_dist_tensor.numel() else torch.empty(0)
    )
    pose_df = pd.DataFrame(pose_rows)

    summary = {
        "specimen_id": args.specimen_id,
        "run_name": args.wandb_name,
        "checkpoint_path": args.test_weight_path,
        "mean_dist": torch.nanmean(dist_tensor).item() if dist_tensor.numel() else float("nan"),
        "mean_dist_raw": valid_raw_distances.mean().item() if valid_raw_distances.numel() else float("nan"),
        "median_dist": torch.nanmedian(dist_tensor).item() if dist_tensor.numel() else float("nan"),
        "median_dist_raw": torch.nanmedian(raw_dist_tensor).item() if raw_dist_tensor.numel() else float("nan"),
        "std_dist": valid_distances.std(unbiased=False).item() if valid_distances.numel() else float("nan"),
        "std_dist_raw": valid_raw_distances.std(unbiased=False).item() if valid_raw_distances.numel() else float("nan"),
        "n_images": len(image_rows),
        "n_visible_landmarks": int(torch.sum(~torch.isnan(dist_tensor)).item()) if dist_tensor.numel() else 0,
        "n_pose_success": int(pose_df["pose_success"].sum()) if not pose_df.empty else 0,
        "n_weighted_pose_success": int(pose_df["weighted_pose_success"].sum()) if not pose_df.empty else 0,
        "mean_pose_rotation_diff_deg": float(pose_df.loc[pose_df["pose_success"], "pose_rotation_diff_deg"].mean()) if not pose_df.empty else float("nan"),
        "mean_pose_translation_diff_mm": float(pose_df.loc[pose_df["pose_success"], "pose_translation_diff_mm"].mean()) if not pose_df.empty else float("nan"),
        "mean_pose_mtre_mm": float(pose_df.loc[pose_df["pose_success"], "pose_mtre_mm"].mean()) if not pose_df.empty else float("nan"),
        "mean_weighted_pose_rotation_diff_deg": float(pose_df.loc[pose_df["weighted_pose_success"], "weighted_pose_rotation_diff_deg"].mean()) if not pose_df.empty else float("nan"),
        "mean_weighted_pose_translation_diff_mm": float(pose_df.loc[pose_df["weighted_pose_success"], "weighted_pose_translation_diff_mm"].mean()) if not pose_df.empty else float("nan"),
        "mean_weighted_pose_mtre_mm": float(pose_df.loc[pose_df["weighted_pose_success"], "weighted_pose_mtre_mm"].mean()) if not pose_df.empty else float("nan"),
    }

    out_dir = os.path.join(args.result_dir, "test_results")
    os.makedirs(out_dir, exist_ok=True)

    image_df = pd.DataFrame(image_rows)
    landmark_df = pd.DataFrame(landmark_rows)
    summary_df = pd.DataFrame([summary])

    image_df.to_csv(os.path.join(out_dir, "per_image_results.csv"), index=False)
    landmark_df.to_csv(os.path.join(out_dir, "per_landmark_results.csv"), index=False)
    pose_df.to_csv(os.path.join(out_dir, "per_image_pose_results.csv"), index=False)
    summary_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    per_landmark_summary = (
        landmark_df.groupby(["landmark_index", "landmark_name"], dropna=False)[["dist", "dist_raw"]]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )
    per_landmark_summary.columns = [
        "landmark_index",
        "landmark_name",
        "mean_dist",
        "median_dist",
        "std_dist",
        "n_visible_samples",
        "mean_dist_raw",
        "median_dist_raw",
        "std_dist_raw",
        "n_visible_samples_raw",
    ]
    per_landmark_summary.to_csv(
        os.path.join(out_dir, "per_landmark_summary.csv"),
        index=False,
    )

    print(f"Saved test summary to {os.path.join(out_dir, 'summary.csv')}")
    return summary, image_df, landmark_df
