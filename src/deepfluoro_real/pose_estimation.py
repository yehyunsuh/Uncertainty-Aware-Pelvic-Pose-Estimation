from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from src.deepfluoro_real.convention import world_to_camera_to_camera_to_pelvis
from src.deepfluoro_real.projection import camera_points, project_points, reprojection_errors

MAX_NFEV = 20000


@dataclass(frozen=True)
class PoseEstimate:
    world_to_camera: np.ndarray
    cam_to_pelvis_vol: np.ndarray
    rotation_diff_deg: float
    translation_diff_mm: float
    reprojection_error_mean_px: float
    reprojection_error_median_px: float
    mtre_mm: float
    optimization_cost: float
    optimization_success: bool
    optimization_message: str


def _rodrigues_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(rotvec).as_matrix()


def _matrix_to_rodrigues(rotation: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(rotation).as_rotvec()


def _compose_world_to_camera(rotvec: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _rodrigues_to_matrix(rotvec)
    matrix[:3, 3] = translation
    return matrix


def _decompose_world_to_camera(world_to_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _matrix_to_rodrigues(world_to_camera[:3, :3]), world_to_camera[:3, 3].copy()


def _normalize_image_points(points_2d: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    homog = np.concatenate([points_2d, np.ones((len(points_2d), 1))], axis=1)
    normalized = (np.linalg.inv(intrinsic) @ homog.T).T
    return normalized[:, :2] / normalized[:, 2:3]


def estimate_world_to_camera_dlt(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    if len(points_3d) < 6:
        raise ValueError("At least 6 correspondences are required for DLT initialization.")

    normalized_2d = _normalize_image_points(points_2d, intrinsic)
    a_rows = []
    for point_3d, point_2d in zip(points_3d, normalized_2d):
        x, y = point_2d
        X, Y, Z = point_3d
        Xh = np.array([X, Y, Z, 1.0], dtype=np.float64)
        zeros = np.zeros(4, dtype=np.float64)
        a_rows.append(np.concatenate([Xh, zeros, -x * Xh]))
        a_rows.append(np.concatenate([zeros, Xh, -y * Xh]))
    A = np.stack(a_rows, axis=0)

    _, _, vh = np.linalg.svd(A, full_matrices=False)
    projection = vh[-1].reshape(3, 4)

    if np.linalg.det(projection[:, :3]) < 0:
        projection = -projection

    M = projection[:, :3]
    t = projection[:, 3]
    U, singular_vals, Vt = np.linalg.svd(M)
    rotation = U @ Vt
    if np.linalg.det(rotation) < 0:
        rotation = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
    scale = float(np.mean(singular_vals))
    translation = t / scale

    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = rotation
    world_to_camera[:3, 3] = translation
    return world_to_camera


def _residuals(
    params: np.ndarray,
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    world_to_camera = _compose_world_to_camera(params[:3], params[3:])
    projected = project_points(points_3d, intrinsic, world_to_camera)
    return (projected - points_2d).ravel()


def _residuals_weighted(
    params: np.ndarray,
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    weights: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    world_to_camera = _compose_world_to_camera(params[:3], params[3:])
    projected = project_points(points_3d, intrinsic, world_to_camera)
    residuals = projected - points_2d
    sqrt_weights = np.sqrt(np.asarray(weights, dtype=np.float64)).reshape(-1, 1)
    return (sqrt_weights * residuals).ravel()


def refine_world_to_camera(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    intrinsic: np.ndarray,
    initial_world_to_camera: np.ndarray,
) -> tuple[np.ndarray, least_squares]:
    rotvec, translation = _decompose_world_to_camera(initial_world_to_camera)
    initial = np.concatenate([rotvec, translation], axis=0)
    result = least_squares(
        _residuals,
        initial,
        method="lm",
        args=(points_3d, points_2d, intrinsic),
        max_nfev=MAX_NFEV,
    )
    return _compose_world_to_camera(result.x[:3], result.x[3:]), result


def refine_world_to_camera_weighted(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    weights: np.ndarray,
    intrinsic: np.ndarray,
    initial_world_to_camera: np.ndarray,
) -> tuple[np.ndarray, least_squares]:
    rotvec, translation = _decompose_world_to_camera(initial_world_to_camera)
    initial = np.concatenate([rotvec, translation], axis=0)
    result = least_squares(
        _residuals_weighted,
        initial,
        method="lm",
        args=(points_3d, points_2d, weights, intrinsic),
        max_nfev=MAX_NFEV,
    )
    return _compose_world_to_camera(result.x[:3], result.x[3:]), result


def rotation_difference_degrees(gt_transform: np.ndarray, pred_transform: np.ndarray) -> float:
    delta = pred_transform[:3, :3] @ gt_transform[:3, :3].T
    trace = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def translation_difference_mm(gt_transform: np.ndarray, pred_transform: np.ndarray) -> float:
    return float(np.linalg.norm(pred_transform[:3, 3] - gt_transform[:3, 3]))


def mtre_mm(
    points_3d: np.ndarray,
    gt_world_to_camera: np.ndarray,
    pred_world_to_camera: np.ndarray,
) -> float:
    gt_cam = camera_points(points_3d, gt_world_to_camera)
    pred_cam = camera_points(points_3d, pred_world_to_camera)
    return float(np.linalg.norm(pred_cam - gt_cam, axis=1).mean())


def estimate_pose_from_correspondences(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    intrinsic: np.ndarray,
    dataset_extrinsic: np.ndarray,
    gt_cam_to_pelvis_vol: np.ndarray,
) -> PoseEstimate:
    gt_world_to_camera = dataset_extrinsic @ np.linalg.inv(gt_cam_to_pelvis_vol)
    initial_world_to_camera = estimate_world_to_camera_dlt(points_3d, points_2d, intrinsic)
    refined_world_to_camera, result = refine_world_to_camera(
        points_3d=points_3d,
        points_2d=points_2d,
        intrinsic=intrinsic,
        initial_world_to_camera=initial_world_to_camera,
    )
    refined_cam_to_pelvis = world_to_camera_to_camera_to_pelvis(
        refined_world_to_camera, dataset_extrinsic
    )
    projected = project_points(points_3d, intrinsic, refined_world_to_camera)
    errors = reprojection_errors(projected, points_2d)

    return PoseEstimate(
        world_to_camera=refined_world_to_camera,
        cam_to_pelvis_vol=refined_cam_to_pelvis,
        rotation_diff_deg=rotation_difference_degrees(gt_cam_to_pelvis_vol, refined_cam_to_pelvis),
        translation_diff_mm=translation_difference_mm(gt_cam_to_pelvis_vol, refined_cam_to_pelvis),
        reprojection_error_mean_px=float(errors.mean()),
        reprojection_error_median_px=float(np.median(errors)),
        mtre_mm=mtre_mm(points_3d, gt_world_to_camera, refined_world_to_camera),
        optimization_cost=float(result.cost),
        optimization_success=bool(result.success),
        optimization_message=str(result.message),
    )


def estimate_pose_from_correspondences_weighted(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    weights: np.ndarray,
    intrinsic: np.ndarray,
    dataset_extrinsic: np.ndarray,
    gt_cam_to_pelvis_vol: np.ndarray,
) -> PoseEstimate:
    if len(points_3d) != len(points_2d) or len(points_3d) != len(weights):
        raise ValueError("points_3d, points_2d, and weights must have the same length.")

    gt_world_to_camera = dataset_extrinsic @ np.linalg.inv(gt_cam_to_pelvis_vol)
    initial_world_to_camera = estimate_world_to_camera_dlt(points_3d, points_2d, intrinsic)
    refined_world_to_camera, result = refine_world_to_camera_weighted(
        points_3d=points_3d,
        points_2d=points_2d,
        weights=weights,
        intrinsic=intrinsic,
        initial_world_to_camera=initial_world_to_camera,
    )
    refined_cam_to_pelvis = world_to_camera_to_camera_to_pelvis(
        refined_world_to_camera, dataset_extrinsic
    )
    projected = project_points(points_3d, intrinsic, refined_world_to_camera)
    errors = reprojection_errors(projected, points_2d)

    return PoseEstimate(
        world_to_camera=refined_world_to_camera,
        cam_to_pelvis_vol=refined_cam_to_pelvis,
        rotation_diff_deg=rotation_difference_degrees(gt_cam_to_pelvis_vol, refined_cam_to_pelvis),
        translation_diff_mm=translation_difference_mm(gt_cam_to_pelvis_vol, refined_cam_to_pelvis),
        reprojection_error_mean_px=float(errors.mean()),
        reprojection_error_median_px=float(np.median(errors)),
        mtre_mm=mtre_mm(points_3d, gt_world_to_camera, refined_world_to_camera),
        optimization_cost=float(result.cost),
        optimization_success=bool(result.success),
        optimization_message=str(result.message),
    )
