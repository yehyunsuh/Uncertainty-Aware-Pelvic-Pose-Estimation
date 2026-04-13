from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from src.train.model import UNet


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DetectorConfig:
    image_resize: int = 512
    n_landmarks: int = 14
    encoder_depth: int = 5
    decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16)
    checkpoint_suffix: str = "hard_dist"
    model_weight_dir: str = "model_weight"
    model_type: str = "patient_held_out"
    intensity_mode: str = "percentile_1_99"
    apply_invert: bool = False
    apply_horizontal_flip: bool = True


@dataclass(frozen=True)
class PreprocessedImage:
    raw_gray: np.ndarray
    raw_rgb_u8: np.ndarray
    model_rgb_u8: np.ndarray
    image_tensor: torch.Tensor
    raw_height: int
    raw_width: int
    model_height: int
    model_width: int
    scale_x: float
    scale_y: float
    apply_horizontal_flip: bool
    apply_rot180: bool


@dataclass(frozen=True)
class DetectionResult:
    logits: np.ndarray
    probs: np.ndarray
    coords_model: np.ndarray
    coords_raw: np.ndarray
    confidence: np.ndarray
    inference_runtime_ms: float


def _detector_args(cfg: DetectorConfig) -> SimpleNamespace:
    return SimpleNamespace(
        encoder_depth=cfg.encoder_depth,
        decoder_channels=list(cfg.decoder_channels),
        n_landmarks=cfg.n_landmarks,
    )


def checkpoint_path_for_specimen(specimen_id: str, cfg: DetectorConfig) -> Path:
    return Path(cfg.model_weight_dir) / cfg.model_type / f"{specimen_id}_{cfg.checkpoint_suffix}.pth"


def load_detector_model(specimen_id: str, device: torch.device, cfg: DetectorConfig) -> torch.nn.Module:
    checkpoint_path = checkpoint_path_for_specimen(specimen_id, cfg)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {checkpoint_path}")

    model = UNet(_detector_args(cfg), str(device))
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def normalize_raw_image_to_uint8(raw_image: np.ndarray, intensity_mode: str = "percentile_1_99") -> np.ndarray:
    image = np.asarray(raw_image, dtype=np.float32)
    if intensity_mode == "percentile_1_99":
        lo, hi = np.percentile(image, [1.0, 99.0])
    elif intensity_mode == "minmax":
        lo = float(image.min())
        hi = float(image.max())
    else:
        raise ValueError(f"Unknown intensity_mode: {intensity_mode}")

    if hi <= lo:
        lo = float(image.min())
        hi = float(image.max())

    scaled = np.clip((image - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return np.round(scaled * 255.0).astype(np.uint8)


def preprocessing_transform(image_resize: int) -> A.Compose:
    transforms: list[A.BasicTransform] = [
        A.Resize(image_resize, image_resize),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return A.Compose(transforms + [ToTensorV2()])


def resize_keypoints_transform(image_resize: int) -> A.Compose:
    return A.Compose(
        [A.Resize(image_resize, image_resize)],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def rotate_points_180(points: np.ndarray, width: int, height: int) -> np.ndarray:
    rotated = np.asarray(points, dtype=np.float32).copy()
    rotated[:, 0] = (width - 1) - rotated[:, 0]
    rotated[:, 1] = (height - 1) - rotated[:, 1]
    return rotated


def preprocess_raw_image(
    raw_image: np.ndarray,
    cfg: DetectorConfig,
    apply_rot180: bool = False,
) -> PreprocessedImage:
    raw_gray_u8 = normalize_raw_image_to_uint8(raw_image, cfg.intensity_mode)
    raw_rgb_u8 = cv2.cvtColor(raw_gray_u8, cv2.COLOR_GRAY2RGB)
    working_rgb_u8 = raw_rgb_u8
    if apply_rot180:
        working_rgb_u8 = np.ascontiguousarray(np.rot90(working_rgb_u8, k=2))

    model_rgb_u8 = cv2.resize(
        working_rgb_u8,
        (cfg.image_resize, cfg.image_resize),
        interpolation=cv2.INTER_LINEAR,
    )
    if cfg.apply_horizontal_flip:
        model_rgb_u8 = np.ascontiguousarray(model_rgb_u8[:, ::-1])
    if cfg.apply_invert:
        model_rgb_u8 = 255 - model_rgb_u8
    transform = preprocessing_transform(cfg.image_resize)
    transformed = transform(image=model_rgb_u8)
    image_tensor = transformed["image"].float().unsqueeze(0)
    raw_height, raw_width = raw_gray_u8.shape[:2]
    model_height, model_width = model_rgb_u8.shape[:2]

    return PreprocessedImage(
        raw_gray=raw_gray_u8,
        raw_rgb_u8=raw_rgb_u8,
        model_rgb_u8=model_rgb_u8,
        image_tensor=image_tensor,
        raw_height=raw_height,
        raw_width=raw_width,
        model_height=model_height,
        model_width=model_width,
        scale_x=model_width / raw_width,
        scale_y=model_height / raw_height,
        apply_horizontal_flip=cfg.apply_horizontal_flip,
        apply_rot180=apply_rot180,
    )


def raw_to_model_coords(points_raw: np.ndarray, prep: PreprocessedImage) -> np.ndarray:
    points_model = np.asarray(points_raw, dtype=np.float32).copy()
    if prep.apply_rot180:
        points_model = rotate_points_180(points_model, prep.raw_width, prep.raw_height)
    points_model[:, 0] *= prep.scale_x
    points_model[:, 1] *= prep.scale_y
    if prep.apply_horizontal_flip:
        points_model[:, 0] = (prep.model_width - 1) - points_model[:, 0]
    return points_model


def model_to_raw_coords(points_model: np.ndarray, prep: PreprocessedImage) -> np.ndarray:
    points_raw = np.asarray(points_model, dtype=np.float32).copy()
    if prep.apply_horizontal_flip:
        points_raw[:, 0] = (prep.model_width - 1) - points_raw[:, 0]
    points_raw[:, 0] /= prep.scale_x
    points_raw[:, 1] /= prep.scale_y
    if prep.apply_rot180:
        points_raw = rotate_points_180(points_raw, prep.raw_width, prep.raw_height)
    return points_raw


@torch.no_grad()
def infer_landmarks(
    model: torch.nn.Module,
    prep: PreprocessedImage,
    device: torch.device,
) -> DetectionResult:
    import time

    image = prep.image_tensor.to(device)
    start = time.perf_counter()
    logits = model(image)
    probs = torch.sigmoid(logits)
    runtime_ms = (time.perf_counter() - start) * 1000.0

    probs_np = probs[0].detach().cpu().numpy()
    _, h, w = probs_np.shape
    flat = probs_np.reshape(probs_np.shape[0], -1)
    max_idx = flat.argmax(axis=1)
    max_vals = flat.max(axis=1)

    coords_model = np.zeros((probs_np.shape[0], 2), dtype=np.float32)
    for c, idx in enumerate(max_idx):
        y, x = divmod(int(idx), w)
        coords_model[c] = np.array([x, y], dtype=np.float32)

    coords_raw = model_to_raw_coords(coords_model, prep)
    return DetectionResult(
        logits=logits[0].detach().cpu().numpy(),
        probs=probs_np,
        coords_model=coords_model,
        coords_raw=coords_raw,
        confidence=max_vals.astype(np.float32),
        inference_runtime_ms=float(runtime_ms),
    )
