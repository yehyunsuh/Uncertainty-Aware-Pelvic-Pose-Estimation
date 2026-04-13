from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.deepfluoro_real.detector import (
    DetectorConfig,
    checkpoint_path_for_specimen,
    infer_landmarks,
    load_detector_model,
    preprocess_raw_image,
    raw_to_model_coords,
)
from src.deepfluoro_real.io import list_projection_ids, list_specimen_ids, load_case
from src.deepfluoro_real.pose_estimation import estimate_pose_from_correspondences
from src.deepfluoro_real.projection import project_points, reprojection_errors
from src.deepfluoro_real.convention import camera_to_pelvis_to_world_to_camera


LM_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#ffff33", "#a65628", "#f781bf",
    "#999999", "#66c2a5", "#fc8d62", "#8da0cb",
    "#e78ac3", "#a6d854",
]


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


def _normalize_for_display(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float64)
    return np.clip(image / 255.0, 0.0, 1.0)


def _save_bridge_figure(
    case,
    prep,
    gt_model: np.ndarray,
    pred_model: np.ndarray,
    pred_raw: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=180)

    axes[0, 0].imshow(_normalize_for_display(prep.raw_gray), cmap="gray")
    axes[0, 0].set_title("Raw DeepFluoro Detector Image")
    axes[0, 0].set_axis_off()

    axes[0, 1].imshow(_normalize_for_display(prep.model_rgb_u8))
    axes[0, 1].scatter(gt_model[:, 0], gt_model[:, 1], s=16, c="#00c853", label="GT model")
    axes[0, 1].scatter(pred_model[:, 0], pred_model[:, 1], s=16, c="#d50000", marker="x", label="Pred model")
    axes[0, 1].set_title("Model-Space Coordinates")
    axes[0, 1].legend(loc="upper right", fontsize=7, frameon=True)
    axes[0, 1].set_axis_off()

    axes[1, 0].imshow(_normalize_for_display(prep.raw_gray), cmap="gray")
    axes[1, 0].scatter(case.landmarks_2d[:, 0], case.landmarks_2d[:, 1], s=16, c="#00c853", label="GT raw")
    axes[1, 0].scatter(pred_raw[:, 0], pred_raw[:, 1], s=16, c="#d50000", marker="x", label="Pred raw")
    axes[1, 0].set_title("Raw Detector Coordinates")
    axes[1, 0].legend(loc="upper right", fontsize=7, frameon=True)
    axes[1, 0].set_axis_off()

    axes[1, 1].imshow(_normalize_for_display(prep.raw_gray), cmap="gray")
    for idx, (gt_pt, pred_pt) in enumerate(zip(case.landmarks_2d, pred_raw)):
        color = LM_COLORS[idx % len(LM_COLORS)]
        axes[1, 1].plot(
            [gt_pt[0], pred_pt[0]],
            [gt_pt[1], pred_pt[1]],
            color=color,
            linewidth=0.8,
            alpha=0.85,
        )
        axes[1, 1].scatter(gt_pt[0], gt_pt[1], s=12, c=color)
        axes[1, 1].scatter(pred_pt[0], pred_pt[1], s=18, c=color, marker="x")
    axes[1, 1].set_title("Prediction vs GT in Raw Space")
    axes[1, 1].set_axis_off()

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_input_comparison_figure(
    case,
    prep,
    gt_model: np.ndarray,
    pred_model: np.ndarray,
    pred_raw: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=180)

    axes[0].imshow(_normalize_for_display(prep.raw_gray), cmap="gray")
    axes[0].scatter(case.landmarks_2d[:, 0], case.landmarks_2d[:, 1], s=18, c="#00c853", label="GT")
    axes[0].scatter(pred_raw[:, 0], pred_raw[:, 1], s=22, c="#d50000", marker="x", label="Pred")
    axes[0].set_title("Raw Real Image Used")
    axes[0].legend(loc="upper right", fontsize=8, frameon=True)
    axes[0].set_axis_off()

    axes[1].imshow(_normalize_for_display(prep.model_rgb_u8))
    axes[1].scatter(gt_model[:, 0], gt_model[:, 1], s=18, c="#00c853", label="GT")
    axes[1].scatter(pred_model[:, 0], pred_model[:, 1], s=22, c="#d50000", marker="x", label="Pred")
    axes[1].set_title("Image Fed Into Model")
    axes[1].legend(loc="upper right", fontsize=8, frameon=True)
    axes[1].set_axis_off()

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_pose_overlay(
    prep,
    gt_raw: np.ndarray,
    pred_raw: np.ndarray,
    reproj_raw: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    ax.imshow(_normalize_for_display(prep.raw_gray), cmap="gray")
    ax.scatter(gt_raw[:, 0], gt_raw[:, 1], s=16, c="#00c853", label="GT landmarks")
    ax.scatter(pred_raw[:, 0], pred_raw[:, 1], s=18, c="#d50000", marker="x", label="Pred landmarks")
    ax.scatter(reproj_raw[:, 0], reproj_raw[:, 1], s=16, c="#2962ff", marker="+", label="Pred-pose reproj")
    ax.set_title("Predicted Landmarks and Pose Reprojection")
    ax.legend(loc="upper right", fontsize=7, frameon=True)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _write_predicted_landmarks_csv(
    output_path: Path,
    landmark_names: tuple[str, ...],
    gt_raw: np.ndarray,
    pred_model: np.ndarray,
    pred_raw: np.ndarray,
    confidence: np.ndarray,
    pixel_errors: np.ndarray,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "landmark",
                "gt_col_px",
                "gt_row_px",
                "pred_model_col_px",
                "pred_model_row_px",
                "pred_raw_col_px",
                "pred_raw_row_px",
                "confidence",
                "pixel_error_raw",
            ],
        )
        writer.writeheader()
        for name, gt_pt, pred_m, pred_r, conf, err in zip(
            landmark_names, gt_raw, pred_model, pred_raw, confidence, pixel_errors
        ):
            writer.writerow(
                {
                    "landmark": name,
                    "gt_col_px": float(gt_pt[0]),
                    "gt_row_px": float(gt_pt[1]),
                    "pred_model_col_px": float(pred_m[0]),
                    "pred_model_row_px": float(pred_m[1]),
                    "pred_raw_col_px": float(pred_r[0]),
                    "pred_raw_row_px": float(pred_r[1]),
                    "confidence": float(conf),
                    "pixel_error_raw": float(err),
                }
            )


def _summarize_results(case_rows: list[dict], output_root: Path) -> None:
    aggregate_dir = output_root / "_aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    if not case_rows:
        return

    csv_path = aggregate_dir / "cases.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(case_rows[0].keys()))
        writer.writeheader()
        writer.writerows(case_rows)

    valid_pred = [row for row in case_rows if row["pred_pose_success"]]
    invalid_pred = [row for row in case_rows if not row["pred_pose_success"]]
    gt_valid = [row for row in case_rows if row["gt_pose_success"]]

    def pct(values: list[float], q: float) -> float | None:
        if not values:
            return None
        return float(np.percentile(np.asarray(values, dtype=np.float64), q))

    pred_mtre = [row["pred_pose_mtre_mm"] for row in valid_pred]
    pred_rot = [row["pred_pose_rotation_diff_deg"] for row in valid_pred]
    pred_trans = [row["pred_pose_translation_diff_mm"] for row in valid_pred]
    pixel_error = [row["pred_landmark_error_mean_px"] for row in case_rows]
    preprocess_ms = [row["preprocess_runtime_ms"] for row in case_rows]
    infer_ms = [row["detector_inference_ms"] for row in case_rows]
    pose_ms = [row["pred_pose_runtime_ms"] for row in valid_pred]
    total_ms = [row["pred_pipeline_total_ms"] for row in case_rows]

    per_specimen = []
    by_specimen: dict[str, list[dict]] = {}
    for row in case_rows:
        by_specimen.setdefault(row["specimen_id"], []).append(row)
    for specimen_id, rows in sorted(by_specimen.items()):
        pred_ok = [row for row in rows if row["pred_pose_success"]]
        per_specimen.append(
            {
                "specimen_id": specimen_id,
                "num_cases": len(rows),
                "valid_pred_cases": len(pred_ok),
                "invalid_pred_cases": len(rows) - len(pred_ok),
                "mean_pixel_error_px": float(np.mean([row["pred_landmark_error_mean_px"] for row in rows])),
                "mean_pred_mtre_mm": float(np.mean([row["pred_pose_mtre_mm"] for row in pred_ok])) if pred_ok else None,
                "median_pred_mtre_mm": float(np.median([row["pred_pose_mtre_mm"] for row in pred_ok])) if pred_ok else None,
                "mean_preprocess_runtime_ms": float(np.mean([row["preprocess_runtime_ms"] for row in rows])),
                "mean_detector_inference_ms": float(np.mean([row["detector_inference_ms"] for row in rows])),
                "mean_pred_pose_runtime_ms": float(np.mean([row["pred_pose_runtime_ms"] for row in pred_ok])) if pred_ok else None,
                "mean_pred_pipeline_total_ms": float(np.mean([row["pred_pipeline_total_ms"] for row in rows])),
            }
        )

    with (aggregate_dir / "per_specimen_summary.json").open("w", encoding="utf-8") as f:
        json.dump(per_specimen, f, indent=2)

    overall = {
        "num_cases": len(case_rows),
        "valid_pred_cases": len(valid_pred),
        "invalid_pred_cases": len(invalid_pred),
        "valid_gt_upper_bound_cases": len(gt_valid),
        "mean_pred_landmark_error_px": float(np.mean(pixel_error)),
        "median_pred_landmark_error_px": float(np.median(pixel_error)),
        "mean_pred_pose_mtre_mm": float(np.mean(pred_mtre)) if pred_mtre else None,
        "median_pred_pose_mtre_mm": float(np.median(pred_mtre)) if pred_mtre else None,
        "p95_pred_pose_mtre_mm": pct(pred_mtre, 95),
        "mean_pred_pose_rotation_diff_deg": float(np.mean(pred_rot)) if pred_rot else None,
        "mean_pred_pose_translation_diff_mm": float(np.mean(pred_trans)) if pred_trans else None,
        "mean_preprocess_runtime_ms": float(np.mean(preprocess_ms)),
        "median_preprocess_runtime_ms": float(np.median(preprocess_ms)),
        "mean_detector_inference_ms": float(np.mean(infer_ms)),
        "median_detector_inference_ms": float(np.median(infer_ms)),
        "mean_pred_pose_runtime_ms": float(np.mean(pose_ms)) if pose_ms else None,
        "median_pred_pose_runtime_ms": float(np.median(pose_ms)) if pose_ms else None,
        "mean_pred_pipeline_total_ms": float(np.mean(total_ms)),
        "median_pred_pipeline_total_ms": float(np.median(total_ms)),
        "invalid_pred_case_details": [
            {
                "specimen_id": row["specimen_id"],
                "projection_id": row["projection_id"],
                "message": row["pred_pose_message"],
            }
            for row in invalid_pred
        ],
        "preprocessing_caveat": (
            "Detector was trained on 512x512 uint8 synthetic DRRs with Resize+ImageNet Normalize+InvertImg. "
            f"Raw DeepFluoro fluoroscopy images are 1536x1536 float32, so this baseline uses {case_rows[0]['intensity_mode']} "
            "windowing to map raw images to uint8 before detector preprocessing with "
            f"apply_invert={case_rows[0]['apply_invert']} and "
            f"apply_horizontal_flip={case_rows[0]['apply_horizontal_flip']}."
        ),
    }
    with (aggregate_dir / "overall_summary.json").open("w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), dpi=180)

    def sorted_plot(ax, values, color, title, ylabel, log_scale=False):
        values = np.asarray(values, dtype=np.float64)
        values = np.sort(values)
        x = np.arange(1, len(values) + 1)
        if log_scale and len(values):
            positive = values[values > 0]
            floor = float(min(positive.min() * 0.5, 1e-8)) if len(positive) else 1e-8
            values = np.clip(values, floor, None)
            ax.set_yscale("log")
            zero_count = int((np.asarray(values) <= floor).sum())
            if zero_count:
                ax.text(
                    0.03,
                    0.95,
                    f"{zero_count} zeros shown at {floor:.1e}",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
                )
        ax.plot(x, values, color=color, linewidth=1.4)
        ax.scatter(x, values, color=color, s=8, alpha=0.65)
        ax.set_title(title)
        ax.set_xlabel("Sorted case index")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, linewidth=0.6)

    if pred_mtre:
        sorted_plot(axes[0], pred_mtre, "#1565c0", "Predicted-Pose mTRE", "mTRE (mm)")
        sorted_plot(axes[1], pred_rot, "#2e7d32", "Predicted Rotation Error", "Rotation Difference (deg)", log_scale=True)
        sorted_plot(axes[2], pred_trans, "#ef6c00", "Predicted Translation Error", "Translation Difference (mm)", log_scale=True)
    else:
        for idx in range(3):
            axes[idx].text(0.5, 0.5, "No valid predicted pose cases", ha="center", va="center")
            axes[idx].set_axis_off()

    pixel_err = np.asarray(pixel_error, dtype=np.float64)
    axes[3].hist(pixel_err, bins=min(40, max(12, len(pixel_err) // 5)), color="#6a1b9a", alpha=0.9)
    axes[3].set_title("Mean Landmark Pixel Error")
    axes[3].set_xlabel("Mean pixel error per image (raw px)")
    axes[3].set_ylabel("Cases")
    axes[3].grid(True, alpha=0.25, linewidth=0.6)

    fig.tight_layout()
    fig.savefig(aggregate_dir / "overall_summary.png", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predicted landmarks on raw DeepFluoro images.")
    parser.add_argument("--h5_path", default="data/ipcai_2020_full_res_data.h5")
    parser.add_argument("--output_dir", default="visualizations/deepfluoro_real_predicted_baseline")
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
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_case_artifacts", action="store_true")
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

    models: dict[str, torch.nn.Module] = {}
    case_rows: list[dict] = []

    for specimen_id, projection_id in case_keys:
        case = load_case(args.h5_path, specimen_id, projection_id)
        case_dir = output_root / specimen_id / projection_id
        if not args.skip_case_artifacts:
            case_dir.mkdir(parents=True, exist_ok=True)

        if specimen_id not in models:
            models[specimen_id] = load_detector_model(specimen_id, device, cfg)

        import time
        full_eval_start = time.perf_counter()
        preprocess_start = time.perf_counter()
        prep = preprocess_raw_image(case.image, cfg, apply_rot180=case.rot_180_for_up)
        preprocess_runtime_ms = (time.perf_counter() - preprocess_start) * 1000.0
        detection = infer_landmarks(models[specimen_id], prep, device)
        pixel_errors = reprojection_errors(detection.coords_raw, case.landmarks_2d)
        gt_model = raw_to_model_coords(case.landmarks_2d, prep)

        gt_pose_start = time.perf_counter()
        gt_pose = estimate_pose_from_correspondences(
            points_3d=case.landmarks_3d,
            points_2d=case.landmarks_2d,
            intrinsic=case.calibration.intrinsic,
            dataset_extrinsic=case.calibration.extrinsic,
            gt_cam_to_pelvis_vol=case.cam_to_pelvis_vol,
        )
        gt_pose_runtime_ms = (time.perf_counter() - gt_pose_start) * 1000.0

        pred_pose_success = True
        pred_pose_message = "ok"
        pred_pose_dict = {
            "pred_pose_rotation_diff_deg": np.nan,
            "pred_pose_translation_diff_mm": np.nan,
            "pred_pose_mtre_mm": np.nan,
            "pred_pose_reprojection_error_mean_px": np.nan,
            "pred_pose_reprojection_error_median_px": np.nan,
            "pred_pose_runtime_ms": np.nan,
        }
        pred_reprojected = np.full_like(case.landmarks_2d, np.nan)
        pred_pose_runtime_ms = np.nan

        try:
            pred_pose_start = time.perf_counter()
            pred_pose = estimate_pose_from_correspondences(
                points_3d=case.landmarks_3d,
                points_2d=detection.coords_raw,
                intrinsic=case.calibration.intrinsic,
                dataset_extrinsic=case.calibration.extrinsic,
                gt_cam_to_pelvis_vol=case.cam_to_pelvis_vol,
            )
            pred_pose_runtime_ms = (time.perf_counter() - pred_pose_start) * 1000.0
            pred_world_to_camera = camera_to_pelvis_to_world_to_camera(
                pred_pose.cam_to_pelvis_vol, case.calibration.extrinsic
            )
            pred_reprojected = project_points(
                case.landmarks_3d, case.calibration.intrinsic, pred_world_to_camera
            )
            pred_pose_dict = {
                "pred_pose_rotation_diff_deg": pred_pose.rotation_diff_deg,
                "pred_pose_translation_diff_mm": pred_pose.translation_diff_mm,
                "pred_pose_mtre_mm": pred_pose.mtre_mm,
                "pred_pose_reprojection_error_mean_px": pred_pose.reprojection_error_mean_px,
                "pred_pose_reprojection_error_median_px": pred_pose.reprojection_error_median_px,
                "pred_pose_runtime_ms": float(pred_pose_runtime_ms),
            }
        except Exception as exc:
            pred_pose_success = False
            pred_pose_message = str(exc)
            pred_pose = None

        pred_pipeline_total_ms = float(preprocess_runtime_ms + detection.inference_runtime_ms)
        if pred_pose_success and pred_pose is not None:
            pred_pipeline_total_ms += float(pred_pose_runtime_ms)
        full_eval_runtime_ms = (time.perf_counter() - full_eval_start) * 1000.0

        if not args.skip_case_artifacts:
            _save_bridge_figure(
                case=case,
                prep=prep,
                gt_model=gt_model,
                pred_model=detection.coords_model,
                pred_raw=detection.coords_raw,
                output_path=case_dir / "bridge_overlay.png",
            )
            _save_input_comparison_figure(
                case=case,
                prep=prep,
                gt_model=gt_model,
                pred_model=detection.coords_model,
                pred_raw=detection.coords_raw,
                output_path=case_dir / "input_vs_model_overlay.png",
            )
            _write_predicted_landmarks_csv(
                output_path=case_dir / "predicted_landmarks.csv",
                landmark_names=case.landmark_names,
                gt_raw=case.landmarks_2d,
                pred_model=detection.coords_model,
                pred_raw=detection.coords_raw,
                confidence=detection.confidence,
                pixel_errors=pixel_errors,
            )
            _save_pose_overlay(
                prep=prep,
                gt_raw=case.landmarks_2d,
                pred_raw=detection.coords_raw,
                reproj_raw=pred_reprojected,
                output_path=case_dir / "predicted_pose_overlay.png",
            )
            with (case_dir / "gt_pose_result.json").open("w", encoding="utf-8") as f:
                json.dump({**asdict(gt_pose), "world_to_camera": gt_pose.world_to_camera.tolist(), "cam_to_pelvis_vol": gt_pose.cam_to_pelvis_vol.tolist()}, f, indent=2)
            if pred_pose_success and pred_pose is not None:
                with (case_dir / "pred_pose_result.json").open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            **asdict(pred_pose),
                            "world_to_camera": pred_pose.world_to_camera.tolist(),
                            "cam_to_pelvis_vol": pred_pose.cam_to_pelvis_vol.tolist(),
                            "preprocess_runtime_ms": float(preprocess_runtime_ms),
                            "detector_inference_ms": float(detection.inference_runtime_ms),
                            "gt_pose_runtime_ms": float(gt_pose_runtime_ms),
                            "pred_pose_runtime_ms": float(pred_pose_runtime_ms),
                            "pred_pipeline_total_ms": float(pred_pipeline_total_ms),
                            "full_eval_runtime_ms": float(full_eval_runtime_ms),
                        },
                        f,
                        indent=2,
                    )
            with (case_dir / "summary.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "specimen_id": specimen_id,
                        "projection_id": projection_id,
                        "checkpoint_path": str(checkpoint_path_for_specimen(specimen_id, cfg)),
                        "intensity_mode": cfg.intensity_mode,
                        "apply_invert": bool(cfg.apply_invert),
                        "apply_horizontal_flip": bool(cfg.apply_horizontal_flip),
                        "pred_landmark_error_mean_px": float(pixel_errors.mean()),
                        "pred_landmark_error_median_px": float(np.median(pixel_errors)),
                        "preprocess_runtime_ms": float(preprocess_runtime_ms),
                        "detector_inference_ms": float(detection.inference_runtime_ms),
                        "gt_pose_runtime_ms": float(gt_pose_runtime_ms),
                        "pred_pose_success": bool(pred_pose_success),
                        "pred_pose_message": pred_pose_message,
                        "pred_pipeline_total_ms": float(pred_pipeline_total_ms),
                        "full_eval_runtime_ms": float(full_eval_runtime_ms),
                        **pred_pose_dict,
                    },
                    f,
                    indent=2,
                )

        row = {
            "specimen_id": specimen_id,
            "projection_id": projection_id,
            "checkpoint_path": str(checkpoint_path_for_specimen(specimen_id, cfg)),
            "intensity_mode": cfg.intensity_mode,
            "apply_invert": bool(cfg.apply_invert),
            "apply_horizontal_flip": bool(cfg.apply_horizontal_flip),
            "pred_landmark_error_mean_px": float(pixel_errors.mean()),
            "pred_landmark_error_median_px": float(np.median(pixel_errors)),
            "pred_landmark_error_max_px": float(pixel_errors.max()),
            "pred_landmark_confidence_mean": float(detection.confidence.mean()),
            "preprocess_runtime_ms": float(preprocess_runtime_ms),
            "detector_inference_ms": float(detection.inference_runtime_ms),
            "gt_pose_success": bool(gt_pose.optimization_success),
            "gt_pose_runtime_ms": float(gt_pose_runtime_ms),
            "gt_pose_rotation_diff_deg": float(gt_pose.rotation_diff_deg),
            "gt_pose_translation_diff_mm": float(gt_pose.translation_diff_mm),
            "gt_pose_mtre_mm": float(gt_pose.mtre_mm),
            "pred_pose_success": bool(pred_pose_success),
            "pred_pose_message": pred_pose_message,
            "pred_pipeline_total_ms": float(pred_pipeline_total_ms),
            "full_eval_runtime_ms": float(full_eval_runtime_ms),
            **pred_pose_dict,
        }
        case_rows.append(row)

    _summarize_results(case_rows, output_root)


if __name__ == "__main__":
    main()
