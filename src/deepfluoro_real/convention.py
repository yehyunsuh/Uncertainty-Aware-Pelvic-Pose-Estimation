from __future__ import annotations

import numpy as np


def camera_to_pelvis_to_world_to_camera(
    cam_to_pelvis_vol: np.ndarray,
    dataset_extrinsic: np.ndarray,
) -> np.ndarray:
    """Convert the raw H5 pose into the world-to-camera extrinsic used for projection.

    This follows the raw DeepFluoro convention verified numerically:
    P = K @ (E @ inv(T_cam_to_pelvis_vol))
    where E is proj-params/extrinsic.
    """
    return dataset_extrinsic @ np.linalg.inv(cam_to_pelvis_vol)


def world_to_camera_to_camera_to_pelvis(
    world_to_camera: np.ndarray,
    dataset_extrinsic: np.ndarray,
) -> np.ndarray:
    """Invert camera_to_pelvis_to_world_to_camera()."""
    return np.linalg.inv(world_to_camera) @ dataset_extrinsic


def rotate_image_for_upright_display(image: np.ndarray, rot_180_for_up: bool) -> np.ndarray:
    if not rot_180_for_up:
        return image
    return np.rot90(image, k=2)


def rotate_points_for_upright_display(
    points_2d: np.ndarray,
    width: int,
    height: int,
    rot_180_for_up: bool,
) -> np.ndarray:
    if not rot_180_for_up:
        return points_2d.copy()
    rotated = points_2d.copy()
    rotated[:, 0] = (width - 1) - rotated[:, 0]
    rotated[:, 1] = (height - 1) - rotated[:, 1]
    return rotated

