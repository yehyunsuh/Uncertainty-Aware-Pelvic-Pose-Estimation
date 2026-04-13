from __future__ import annotations

import csv
import json
import os
from glob import glob
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import binary_dilation
from torch.utils.data import DataLoader, Dataset


LANDMARK_NAMES = (
    "ASIS-l",
    "ASIS-r",
    "FH-l",
    "FH-r",
    "GSN-l",
    "GSN-r",
    "IOF-l",
    "IOF-r",
    "IPS-l",
    "IPS-r",
    "MOF-l",
    "MOF-r",
    "SPS-l",
    "SPS-r",
)

REAL_MANIFEST_DIRNAME = "train_deepfluoro_real"


def _real_manifest_dir(data_dir: str, specimen_id: str) -> Path:
    return Path(data_dir) / specimen_id / "landmark_prediction_csv" / REAL_MANIFEST_DIRNAME


def _read_real_landmarks(json_path: Path) -> list[tuple[int, int]]:
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    landmarks: list[tuple[int, int]] = []
    for name in LANDMARK_NAMES:
        value = data.get(name)
        if value is None:
            landmarks.append((-1, -1))
            continue

        x = int(round(float(value[0][0])))
        y = int(round(float(value[1][0])))
        landmarks.append((x, y))

    return landmarks


def _make_rows_from_real_images(specimen_dir: Path) -> list[list[int | str]]:
    specimen_id = specimen_dir.name
    image_paths = sorted((specimen_dir / "gt_projections").glob("*.png"))
    rows: list[list[int | str]] = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        json_name = f"{image_path.stem}_landmarks_2D.json"
        json_path = specimen_dir / "gt_landmarks_2D" / json_name
        landmarks = _read_real_landmarks(json_path)
        flat_landmarks = [coord for point in landmarks for coord in point]

        rows.append(
            [
                specimen_id,
                image_path.name,
                int(image.shape[1]),
                int(image.shape[0]),
                len(LANDMARK_NAMES),
                *flat_landmarks,
            ]
        )

    return rows


def _split_dataframe(df: pd.DataFrame, train_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy()

    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if len(shuffled) == 1:
        return shuffled.copy(), shuffled.iloc[0:0].copy()

    train_end = int(round(len(shuffled) * train_ratio))
    train_end = max(1, min(train_end, len(shuffled) - 1))

    train_df = shuffled.iloc[:train_end].reset_index(drop=True)
    val_df = shuffled.iloc[train_end:].reset_index(drop=True)
    return train_df, val_df


def _save_manifest(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def preprocessing(args) -> None:
    specimen_dirs = sorted(Path(args.data_dir).glob("*"))
    for specimen_dir in specimen_dirs:
        if not specimen_dir.is_dir():
            continue

        rows = _make_rows_from_real_images(specimen_dir)
        if not rows:
            continue

        columns = [
            "Case ID",
            "Image Name",
            "Image Width",
            "Image Height",
            "Number of Landmarks",
        ]
        for idx in range(args.n_landmarks):
            columns += [f"Landmark {idx + 1} x", f"Landmark {idx + 1} y"]

        full_df = pd.DataFrame(rows, columns=columns)
        train_df, val_df = _split_dataframe(
            full_df,
            train_ratio=args.real_train_ratio,
            seed=args.seed,
        )

        manifest_dir = _real_manifest_dir(args.data_dir, specimen_dir.name)
        _save_manifest(train_df, manifest_dir / "real_train.csv")
        _save_manifest(val_df, manifest_dir / "real_val.csv")
        _save_manifest(full_df, manifest_dir / "real_all.csv")

        print(
            f"Prepared real manifests for {specimen_dir.name}: "
            f"train={len(train_df)}, val={len(val_df)}, all={len(full_df)}"
        )


def _build_transform(args, split: str, domain: str) -> A.Compose:
    transforms: list = []

    if split == "train":
        if domain == "synthetic":
            transforms.append(A.InvertImg(p=1.0))
        transforms.append(
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent={"x": (-0.06, 0.06), "y": (-0.06, 0.06)},
                rotate=(0, 0),
                shear=(0, 0),
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fit_output=False,
                p=0.7,
            )
        )
        transforms.append(
            A.Rotate(
                limit=(0, 359),
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                p=args.rotation_prob,
            )
        )
        transforms.append(
            A.RandomResizedCrop(
                size=(args.image_resize, args.image_resize),
                scale=(0.85, 1.0),
                ratio=(0.9, 1.1),
                p=0.3,
            )
        )
        transforms.append(
            A.Perspective(
                scale=(0.02, 0.05),
                keep_size=True,
                fit_output=False,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=0.2,
            )
        )
        transforms.append(A.OneOf([A.GaussianBlur(blur_limit=(3, 5), p=1.0), A.MotionBlur(blur_limit=5, p=1.0)], p=0.15))
        transforms.append(
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(0.04, 0.10),
                hole_width_range=(0.04, 0.10),
                fill=0,
                p=0.12,
            )
        )
    elif domain == "synthetic":
        transforms.append(A.InvertImg(p=1.0))

    transforms.append(A.Resize(args.image_resize, args.image_resize))
    transforms.extend(
        [
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ]
    )

    return A.Compose(
        transforms,
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


class DeepFluoroSegmentationDataset(Dataset):
    def __init__(
        self,
        args,
        split: str,
        finetune: bool = False,
    ) -> None:
        self.args = args
        self.split = split
        self.finetune = finetune
        self.samples: list[dict[str, object]] = []

        specimen_dirs = sorted(glob(f"{self.args.data_dir}/*"))
        for specimen_path in specimen_dirs:
            specimen_id = os.path.basename(specimen_path)
            if split == "test":
                if specimen_id != self.args.specimen_id:
                    continue
                self.samples.extend(self._load_real_samples(specimen_id, manifest_split="all"))
                continue

            if specimen_id == self.args.specimen_id:
                continue

            if self.args.source_domain in {"real", "mixed"}:
                self.samples.extend(self._load_real_samples(specimen_id, manifest_split=split))
            if self.args.source_domain in {"synthetic", "mixed"}:
                self.samples.extend(self._load_synthetic_samples(specimen_id, split=split))

        self.transforms = {
            "real": _build_transform(args, split=split, domain="real"),
            "synthetic": _build_transform(args, split=split, domain="synthetic"),
        }

    def _load_real_samples(self, specimen_id: str, manifest_split: str) -> list[dict[str, object]]:
        manifest_path = _real_manifest_dir(self.args.data_dir, specimen_id) / f"real_{manifest_split}.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing real manifest: {manifest_path}. Run with --preprocess first."
            )

        samples: list[dict[str, object]] = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader)
            for row in reader:
                image_name = row[1]
                coords = list(map(int, row[5:]))
                landmarks = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
                samples.append(
                    {
                        "domain": "real",
                        "specimen_id": specimen_id,
                        "image_name": image_name,
                        "image_path": Path(self.args.data_dir) / specimen_id / "gt_projections" / image_name,
                        "landmarks": landmarks,
                    }
                )
        return samples

    def _load_synthetic_samples(self, specimen_id: str, split: str) -> list[dict[str, object]]:
        csv_path = (
            Path(self.args.data_dir)
            / specimen_id
            / "landmark_prediction_csv"
            / self.args.synthetic_model_type
            / f"{split}_label_{self.args.task_type}.csv"
        )
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing synthetic CSV: {csv_path}")

        samples: list[dict[str, object]] = []
        with csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader)
            for row in reader:
                image_name = row[1]
                coords = list(map(int, row[5:]))
                landmarks = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
                samples.append(
                    {
                        "domain": "synthetic",
                        "specimen_id": specimen_id,
                        "image_name": image_name,
                        "image_path": (
                            Path(self.args.data_dir)
                            / specimen_id
                            / f"drr_projections_{self.args.task_type}"
                            / f"{specimen_id}_{image_name}"
                        ),
                        "landmarks": landmarks,
                    }
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image_path = sample["image_path"]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        landmarks = list(sample["landmarks"])
        domain = str(sample["domain"])

        transformed = self.transforms[domain](image=image, keypoints=landmarks)
        image_tensor = transformed["image"]
        transformed_landmarks = list(transformed["keypoints"])

        height, width = image_tensor.shape[1:]
        masks = np.zeros((self.args.n_landmarks, height, width), dtype=np.uint8)

        for k, (x, y) in enumerate(transformed_landmarks):
            if landmarks[k][0] == -1 or landmarks[k][1] == -1:
                transformed_landmarks[k] = (0, 0)
                continue

            x = int(round(x))
            y = int(round(y))
            transformed_landmarks[k] = (x, y)

            if self.split in {"train", "val"} and 0 <= y < height and 0 <= x < width:
                masks[k, y, x] = 1
                masks[k] = binary_dilation(
                    masks[k], iterations=self.args.dilation_iters
                ).astype(np.uint8)

        mask_tensor = torch.from_numpy(masks).float()
        sample_name = f"{sample['specimen_id']}/{sample['image_name']}"

        if self.split in {"train", "val"}:
            return image_tensor, mask_tensor, sample_name, transformed_landmarks

        return image_tensor, str(sample["specimen_id"]), str(sample["image_name"]), transformed_landmarks


def dataloader(args, data_type: str = "train", epoch: int = 0):
    if args.preprocess and data_type == "train" and epoch == 0:
        preprocessing(args)

    if data_type == "train":
        train_dataset = DeepFluoroSegmentationDataset(args, split="train")
        val_dataset = DeepFluoroSegmentationDataset(args, split="val")

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
        )

        print(f"Train size: {len(train_loader.dataset)}")
        print(f"Validation size: {len(val_loader.dataset)}")
        return train_loader, val_loader

    if data_type == "test":
        test_dataset = DeepFluoroSegmentationDataset(args, split="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
        )
        print(f"Test size: {len(test_loader.dataset)}")
        return test_loader

    raise ValueError("data_type must be 'train' or 'test'")
