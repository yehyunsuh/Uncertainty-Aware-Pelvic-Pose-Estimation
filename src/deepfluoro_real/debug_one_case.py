from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.deepfluoro_real.convention import (
    camera_to_pelvis_to_world_to_camera,
    rotate_image_for_upright_display,
    rotate_points_for_upright_display,
)
from src.deepfluoro_real.io import list_projection_ids, list_specimen_ids, load_case
from src.deepfluoro_real.pose_estimation import estimate_pose_from_correspondences
from src.deepfluoro_real.projection import project_points, reprojection_errors


def _normalize_for_display(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float64)
    p1, p99 = np.percentile(image, [1, 99])
    if p99 <= p1:
        p1 = float(image.min())
        p99 = float(image.max())
    scaled = np.clip((image - p1) / max(p99 - p1, 1e-6), 0.0, 1.0)
    return scaled


def _save_overlay(
    image: np.ndarray,
    gt_points_2d: np.ndarray,
    projected_points_2d: np.ndarray,
    landmark_names: tuple[str, ...],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    ax.imshow(_normalize_for_display(image), cmap="gray")
    ax.scatter(gt_points_2d[:, 0], gt_points_2d[:, 1], c="#00c853", s=18, label="GT 2D")
    ax.scatter(
        projected_points_2d[:, 0],
        projected_points_2d[:, 1],
        c="#d50000",
        s=18,
        marker="x",
        label="Projected 3D",
    )
    for name, gt_point, pred_point in zip(landmark_names, gt_points_2d, projected_points_2d):
        ax.plot(
            [gt_point[0], pred_point[0]],
            [gt_point[1], pred_point[1]],
            color="#ff9100",
            linewidth=0.7,
            alpha=0.8,
        )
        ax.text(gt_point[0] + 4, gt_point[1] + 4, name, fontsize=6, color="white")
    ax.set_xlim(0, image.shape[1] - 1)
    ax.set_ylim(image.shape[0] - 1, 0)
    ax.set_title("DeepFluoro Raw Geometry Reprojection")
    ax.legend(loc="upper right", fontsize=7, frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _write_landmark_error_csv(
    output_path: Path,
    landmark_names: tuple[str, ...],
    gt_points_2d: np.ndarray,
    projected_points_2d: np.ndarray,
    errors_px: np.ndarray,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "landmark",
                "gt_col_px",
                "gt_row_px",
                "projected_col_px",
                "projected_row_px",
                "error_px",
            ],
        )
        writer.writeheader()
        for name, gt_point, proj_point, error in zip(
            landmark_names, gt_points_2d, projected_points_2d, errors_px
        ):
            writer.writerow(
                {
                    "landmark": name,
                    "gt_col_px": float(gt_point[0]),
                    "gt_row_px": float(gt_point[1]),
                    "projected_col_px": float(proj_point[0]),
                    "projected_row_px": float(proj_point[1]),
                    "error_px": float(error),
                }
            )


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


def run_case(
    h5_path: str,
    specimen_id: str,
    projection_id: str,
    output_root: Path,
    upright_display: bool,
    save_case_artifacts: bool,
) -> dict:
    case = load_case(h5_path, specimen_id, projection_id)
    world_to_camera = camera_to_pelvis_to_world_to_camera(
        case.cam_to_pelvis_vol, case.calibration.extrinsic
    )
    projected_points_2d = project_points(
        case.landmarks_3d, case.calibration.intrinsic, world_to_camera
    )
    gt_points_2d = case.landmarks_2d.copy()
    display_image = case.image
    display_projected = projected_points_2d
    display_gt = gt_points_2d
    if upright_display:
        display_image = rotate_image_for_upright_display(case.image, case.rot_180_for_up)
        display_projected = rotate_points_for_upright_display(
            projected_points_2d,
            width=case.calibration.num_cols,
            height=case.calibration.num_rows,
            rot_180_for_up=case.rot_180_for_up,
        )
        display_gt = rotate_points_for_upright_display(
            gt_points_2d,
            width=case.calibration.num_cols,
            height=case.calibration.num_rows,
            rot_180_for_up=case.rot_180_for_up,
        )

    errors_px = reprojection_errors(projected_points_2d, gt_points_2d)
    pose_start = time.perf_counter()
    pose_estimate = estimate_pose_from_correspondences(
        points_3d=case.landmarks_3d,
        points_2d=case.landmarks_2d,
        intrinsic=case.calibration.intrinsic,
        dataset_extrinsic=case.calibration.extrinsic,
        gt_cam_to_pelvis_vol=case.cam_to_pelvis_vol,
    )
    pose_runtime_ms = (time.perf_counter() - pose_start) * 1000.0

    summary = {
        "specimen_id": specimen_id,
        "projection_id": projection_id,
        "rot_180_for_up": case.rot_180_for_up,
        "num_landmarks": int(len(case.landmark_names)),
        "reprojection_error_mean_px": float(errors_px.mean()),
        "reprojection_error_median_px": float(np.median(errors_px)),
        "reprojection_error_max_px": float(errors_px.max()),
        "pose_recovery_rotation_diff_deg": pose_estimate.rotation_diff_deg,
        "pose_recovery_translation_diff_mm": pose_estimate.translation_diff_mm,
        "pose_recovery_reprojection_error_mean_px": pose_estimate.reprojection_error_mean_px,
        "pose_recovery_reprojection_error_median_px": pose_estimate.reprojection_error_median_px,
        "pose_recovery_mtre_mm": pose_estimate.mtre_mm,
        "pose_recovery_runtime_ms": float(pose_runtime_ms),
        "pose_recovery_success": pose_estimate.optimization_success,
        "pose_recovery_message": pose_estimate.optimization_message,
        "landmark_point_set": "raw_h5_vol-landmarks",
        "display_mode": "upright" if upright_display else "raw",
    }

    if save_case_artifacts:
        case_dir = output_root / specimen_id / projection_id
        case_dir.mkdir(parents=True, exist_ok=True)
        _save_overlay(
            image=display_image,
            gt_points_2d=display_gt,
            projected_points_2d=display_projected,
            landmark_names=case.landmark_names,
            output_path=case_dir / "reprojection_overlay.png",
        )
        _write_landmark_error_csv(
            output_path=case_dir / "per_landmark_reprojection_errors.csv",
            landmark_names=case.landmark_names,
            gt_points_2d=case.landmarks_2d,
            projected_points_2d=projected_points_2d,
            errors_px=errors_px,
        )
        with (case_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with (case_dir / "pose_recovery.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    **asdict(pose_estimate),
                    "world_to_camera": pose_estimate.world_to_camera.tolist(),
                    "cam_to_pelvis_vol": pose_estimate.cam_to_pelvis_vol.tolist(),
                    "pose_recovery_runtime_ms": float(pose_runtime_ms),
                },
                f,
                indent=2,
            )
    return summary


def summarize_cases(case_summaries: list[dict], output_root: Path) -> None:
    aggregate_dir = output_root / "_aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    if not case_summaries:
        return

    csv_path = aggregate_dir / "cases.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(case_summaries[0].keys()))
        writer.writeheader()
        writer.writerows(case_summaries)

    valid = [row for row in case_summaries if row["pose_recovery_success"]]
    invalid = [row for row in case_summaries if not row["pose_recovery_success"]]

    def pct(values: list[float], q: float) -> float | None:
        if not values:
            return None
        return float(np.percentile(np.asarray(values, dtype=np.float64), q))

    pose_mtre = [row["pose_recovery_mtre_mm"] for row in valid]
    pose_rot = [row["pose_recovery_rotation_diff_deg"] for row in valid]
    pose_trans = [row["pose_recovery_translation_diff_mm"] for row in valid]
    pose_runtime_ms = [row["pose_recovery_runtime_ms"] for row in valid]
    gt_reproj = [row["reprojection_error_mean_px"] for row in case_summaries]

    by_specimen: dict[str, dict[str, list[float] | int]] = {}
    for row in case_summaries:
        specimen = row["specimen_id"]
        if specimen not in by_specimen:
            by_specimen[specimen] = {
                "mtre_mm": [],
                "runtime_ms": [],
                "valid": 0,
                "invalid": 0,
                "gt_reproj_px": [],
            }
        by_specimen[specimen]["gt_reproj_px"].append(row["reprojection_error_mean_px"])
        if row["pose_recovery_success"]:
            by_specimen[specimen]["mtre_mm"].append(row["pose_recovery_mtre_mm"])
            by_specimen[specimen]["runtime_ms"].append(row["pose_recovery_runtime_ms"])
            by_specimen[specimen]["valid"] += 1
        else:
            by_specimen[specimen]["invalid"] += 1

    per_specimen = []
    for specimen, values in sorted(by_specimen.items()):
        specimen_row = {
            "specimen_id": specimen,
            "valid_cases": int(values["valid"]),
            "invalid_cases": int(values["invalid"]),
            "mean_gt_reprojection_error_px": float(np.mean(values["gt_reproj_px"])),
            "mean_mtre_mm": float(np.mean(values["mtre_mm"])) if values["mtre_mm"] else None,
            "median_mtre_mm": float(np.median(values["mtre_mm"])) if values["mtre_mm"] else None,
            "mean_pose_runtime_ms": float(np.mean(values["runtime_ms"])) if values["runtime_ms"] else None,
            "median_pose_runtime_ms": float(np.median(values["runtime_ms"])) if values["runtime_ms"] else None,
        }
        per_specimen.append(specimen_row)

    with (aggregate_dir / "per_specimen_summary.json").open("w", encoding="utf-8") as f:
        json.dump(per_specimen, f, indent=2)

    overall = {
        "num_cases": len(case_summaries),
        "valid_cases": len(valid),
        "invalid_cases": len(invalid),
        "mean_gt_reprojection_error_px": float(np.mean(gt_reproj)),
        "median_gt_reprojection_error_px": float(np.median(gt_reproj)),
        "mean_pose_recovery_mtre_mm": float(np.mean(pose_mtre)) if pose_mtre else None,
        "median_pose_recovery_mtre_mm": float(np.median(pose_mtre)) if pose_mtre else None,
        "p25_pose_recovery_mtre_mm": pct(pose_mtre, 25),
        "p50_pose_recovery_mtre_mm": pct(pose_mtre, 50),
        "p95_pose_recovery_mtre_mm": pct(pose_mtre, 95),
        "gfr_10mm": float(np.mean(np.asarray(pose_mtre) <= 10.0)) if pose_mtre else None,
        "gfr_5mm": float(np.mean(np.asarray(pose_mtre) <= 5.0)) if pose_mtre else None,
        "mean_pose_recovery_rotation_diff_deg": float(np.mean(pose_rot)) if pose_rot else None,
        "mean_pose_recovery_translation_diff_mm": float(np.mean(pose_trans)) if pose_trans else None,
        "mean_pose_recovery_runtime_ms": float(np.mean(pose_runtime_ms)) if pose_runtime_ms else None,
        "median_pose_recovery_runtime_ms": float(np.median(pose_runtime_ms)) if pose_runtime_ms else None,
        "p95_pose_recovery_runtime_ms": pct(pose_runtime_ms, 95),
        "invalid_cases_detail": [
            {
                "specimen_id": row["specimen_id"],
                "projection_id": row["projection_id"],
                "message": row["pose_recovery_message"],
            }
            for row in invalid
        ],
    }
    with (aggregate_dir / "overall_summary.json").open("w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    valid_rows = [row for row in case_summaries if row["pose_recovery_success"]]
    if valid_rows:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=180)

        mtre = np.asarray([row["pose_recovery_mtre_mm"] for row in valid_rows], dtype=np.float64)
        rot = np.asarray(
            [row["pose_recovery_rotation_diff_deg"] for row in valid_rows], dtype=np.float64
        )
        trans = np.asarray(
            [row["pose_recovery_translation_diff_mm"] for row in valid_rows], dtype=np.float64
        )
        def _sorted_metric_plot(
            ax: plt.Axes,
            values: np.ndarray,
            color: str,
            title: str,
            ylabel: str,
            log_scale: bool = False,
        ) -> None:
            sorted_vals = np.sort(values)
            x = np.arange(1, len(sorted_vals) + 1)
            if log_scale:
                positive = sorted_vals[sorted_vals > 0]
                floor = float(min(positive.min() * 0.5, 1e-8)) if len(positive) else 1e-8
                sorted_vals = np.clip(sorted_vals, floor, None)
                ax.set_yscale("log")
                zero_count = int((values <= 0).sum())
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
            ax.plot(x, sorted_vals, color=color, linewidth=1.4)
            ax.scatter(x, sorted_vals, color=color, s=8, alpha=0.65)
            ax.set_title(title)
            ax.set_xlabel("Sorted case index")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25, linewidth=0.6)

        _sorted_metric_plot(axes[0], mtre, "#1565c0", "mTRE", "mTRE (mm)")
        _sorted_metric_plot(
            axes[1], rot, "#2e7d32", "Rotation Error", "Rotation Difference (deg)", log_scale=True
        )
        _sorted_metric_plot(
            axes[2], trans, "#ef6c00", "Translation Error", "Translation Difference (mm)", log_scale=True
        )

        fig.tight_layout()
        fig.savefig(aggregate_dir / "overall_summary.png", bbox_inches="tight")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug raw DeepFluoro real-image geometry.")
    parser.add_argument(
        "--h5_path",
        default="data/ipcai_2020_full_res_data.h5",
        help="Path to the raw DeepFluoro HDF5 file.",
    )
    parser.add_argument(
        "--specimen_ids",
        nargs="+",
        default=["17-1882"],
        help="One or more specimen ids.",
    )
    parser.add_argument(
        "--projection_ids",
        nargs="*",
        default=None,
        help="Projection ids to evaluate. If omitted, uses the first N per specimen.",
    )
    parser.add_argument(
        "--max_cases_per_specimen",
        type=int,
        default=1,
        help="Maximum number of projection ids to evaluate per specimen when --projection_ids is omitted.",
    )
    parser.add_argument(
        "--output_dir",
        default="visualizations/deepfluoro_real_debug",
        help="Directory for overlays and summaries.",
    )
    parser.add_argument(
        "--raw_display",
        action="store_true",
        help="Keep saved overlays in raw detector orientation instead of upright display orientation.",
    )
    parser.add_argument(
        "--skip_case_artifacts",
        action="store_true",
        help="Skip per-case overlays and JSON files; still saves aggregate CSV and summaries.",
    )
    parser.add_argument(
        "--all_specimens",
        action="store_true",
        help="Evaluate every specimen in the HDF5 file.",
    )
    parser.add_argument(
        "--all_projections",
        action="store_true",
        help="Evaluate every projection for each selected specimen.",
    )
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

    case_keys = _select_cases(
        h5_path=args.h5_path,
        specimen_ids=specimen_ids,
        projection_ids=args.projection_ids,
        max_cases_per_specimen=max_cases_per_specimen,
    )

    case_summaries = []
    for specimen_id, projection_id in case_keys:
        try:
            case_summaries.append(
                run_case(
                    h5_path=args.h5_path,
                    specimen_id=specimen_id,
                    projection_id=projection_id,
                    output_root=output_root,
                    upright_display=not args.raw_display,
                    save_case_artifacts=not args.skip_case_artifacts,
                )
            )
        except Exception as exc:
            case_summaries.append(
                {
                    "specimen_id": specimen_id,
                    "projection_id": projection_id,
                    "rot_180_for_up": None,
                    "num_landmarks": 0,
                    "reprojection_error_mean_px": None,
                    "reprojection_error_median_px": None,
                    "reprojection_error_max_px": None,
                    "pose_recovery_rotation_diff_deg": None,
                    "pose_recovery_translation_diff_mm": None,
                    "pose_recovery_reprojection_error_mean_px": None,
                    "pose_recovery_reprojection_error_median_px": None,
                    "pose_recovery_mtre_mm": None,
                    "pose_recovery_runtime_ms": None,
                    "pose_recovery_success": False,
                    "pose_recovery_message": str(exc),
                    "landmark_point_set": "raw_h5_vol-landmarks",
                    "display_mode": "raw" if args.raw_display else "upright",
                }
            )

    summarize_cases(case_summaries, output_root)


if __name__ == "__main__":
    main()
