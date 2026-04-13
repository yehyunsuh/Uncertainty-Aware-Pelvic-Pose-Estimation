from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class ProjectionCalibration:
    intrinsic: np.ndarray
    extrinsic: np.ndarray
    num_cols: int
    num_rows: int
    pixel_col_spacing_mm: float
    pixel_row_spacing_mm: float


@dataclass(frozen=True)
class DeepFluoroCase:
    specimen_id: str
    projection_id: str
    image: np.ndarray
    landmark_names: tuple[str, ...]
    landmarks_2d: np.ndarray
    landmarks_3d: np.ndarray
    cam_to_pelvis_vol: np.ndarray
    rot_180_for_up: bool
    calibration: ProjectionCalibration


def _sorted_landmark_names(group: h5py.Group) -> list[str]:
    return sorted(group.keys())


def _read_landmarks_3d(group: h5py.Group, names: list[str]) -> np.ndarray:
    points = []
    for name in names:
        point = np.asarray(group[name][()]).reshape(-1)
        points.append(point[:3].astype(np.float64))
    return np.stack(points, axis=0)


def _read_landmarks_2d(group: h5py.Group, names: list[str]) -> np.ndarray:
    points = []
    for name in names:
        point = np.asarray(group[name][()]).reshape(-1)
        points.append(point[:2].astype(np.float64))
    return np.stack(points, axis=0)


def load_global_calibration(h5_file: h5py.File) -> ProjectionCalibration:
    proj_params = h5_file["proj-params"]
    return ProjectionCalibration(
        intrinsic=np.asarray(proj_params["intrinsic"][()], dtype=np.float64),
        extrinsic=np.asarray(proj_params["extrinsic"][()], dtype=np.float64),
        num_cols=int(proj_params["num-cols"][()]),
        num_rows=int(proj_params["num-rows"][()]),
        pixel_col_spacing_mm=float(proj_params["pixel-col-spacing"][()]),
        pixel_row_spacing_mm=float(proj_params["pixel-row-spacing"][()]),
    )


def list_projection_ids(h5_path: str | Path, specimen_id: str) -> list[str]:
    with h5py.File(h5_path, "r") as h5_file:
        return sorted(h5_file[specimen_id]["projections"].keys())


def list_specimen_ids(h5_path: str | Path) -> list[str]:
    with h5py.File(h5_path, "r") as h5_file:
        return sorted(key for key in h5_file.keys() if key != "proj-params")


def load_case(h5_path: str | Path, specimen_id: str, projection_id: str) -> DeepFluoroCase:
    with h5py.File(h5_path, "r") as h5_file:
        specimen = h5_file[specimen_id]
        projection = specimen["projections"][projection_id]

        landmark_names = _sorted_landmark_names(specimen["vol-landmarks"])
        gt_landmark_names = _sorted_landmark_names(projection["gt-landmarks"])
        if landmark_names != gt_landmark_names:
            raise ValueError(
                f"Landmark name mismatch for {specimen_id}/{projection_id}: "
                f"{landmark_names} != {gt_landmark_names}"
            )

        return DeepFluoroCase(
            specimen_id=specimen_id,
            projection_id=projection_id,
            image=np.asarray(projection["image/pixels"][()], dtype=np.float64),
            landmark_names=tuple(landmark_names),
            landmarks_2d=_read_landmarks_2d(projection["gt-landmarks"], landmark_names),
            landmarks_3d=_read_landmarks_3d(specimen["vol-landmarks"], landmark_names),
            cam_to_pelvis_vol=np.asarray(
                projection["gt-poses"]["cam-to-pelvis-vol"][()], dtype=np.float64
            ),
            rot_180_for_up=bool(projection["rot-180-for-up"][()]),
            calibration=load_global_calibration(h5_file),
        )
