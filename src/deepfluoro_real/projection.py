from __future__ import annotations

import numpy as np


def projection_matrix(intrinsic: np.ndarray, world_to_camera: np.ndarray) -> np.ndarray:
    return intrinsic @ world_to_camera[:3, :]


def project_points(
    points_3d: np.ndarray,
    intrinsic: np.ndarray,
    world_to_camera: np.ndarray,
) -> np.ndarray:
    points_h = np.concatenate([points_3d, np.ones((len(points_3d), 1))], axis=1)
    proj = (projection_matrix(intrinsic, world_to_camera) @ points_h.T).T
    depth = proj[:, 2:3]
    return proj[:, :2] / depth


def reprojection_errors(projected_points_2d: np.ndarray, target_points_2d: np.ndarray) -> np.ndarray:
    return np.linalg.norm(projected_points_2d - target_points_2d, axis=1)


def camera_points(points_3d: np.ndarray, world_to_camera: np.ndarray) -> np.ndarray:
    points_h = np.concatenate([points_3d, np.ones((len(points_3d), 1))], axis=1)
    cam = (world_to_camera @ points_h.T).T
    return cam[:, :3]

