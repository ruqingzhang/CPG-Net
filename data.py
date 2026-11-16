"""Manifest-driven paired RGB/lesion-mask datasets for supervised experiments."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


class LesionManifestDataset(Dataset):
    """Load aligned images, binary lesion masks, and zero-based disease labels.

    Required CSV columns: image,mask,label,split. Paths are resolved relative
    to --data-root (or the manifest directory if --data-root is omitted).
    """

    def __init__(
        self,
        manifest: str | Path,
        *,
        split: str,
        num_classes: int,
        image_size: int = 256,
        data_root: str | Path | None = None,
        mask_threshold: int | None = None,
        mask_ignore_value: int | None = None,
        augment: bool = False,
    ) -> None:
        self.manifest = Path(manifest).expanduser().resolve()
        if not self.manifest.is_file():
            raise FileNotFoundError(f"Manifest does not exist: {self.manifest}")
        self.base_directory = Path(data_root).expanduser().resolve() if data_root else self.manifest.parent
        self.num_classes = num_classes
        self.image_size = image_size
        self.mask_threshold = mask_threshold
        self.mask_ignore_value = mask_ignore_value
        self.augment = augment and split == "train"
        if mask_ignore_value is not None and not 0 <= mask_ignore_value <= 255:
            raise ValueError("mask_ignore_value must be a uint8 value")
        if mask_threshold is not None and not 0 <= mask_threshold <= 255:
            raise ValueError("mask_threshold must be a uint8 value")

        with self.manifest.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"image", "mask", "label", "split"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"Manifest requires columns: {','.join(sorted(required))}")
            self.records = []
            for row_number, row in enumerate(reader, start=2):
                if row["split"].strip().lower() != split.lower():
                    continue
                try:
                    label = int(row["label"])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"Invalid class label at CSV row {row_number}") from error
                if not 0 <= label < num_classes:
                    raise ValueError(f"Class label {label} is outside [0,{num_classes - 1}]")
                image = self._resolve(row["image"])
                mask = self._resolve(row["mask"])
                if not image.is_file() or not mask.is_file():
                    raise FileNotFoundError(f"Missing paired sample at CSV row {row_number}: {image} / {mask}")
                self.records.append((image, mask, label))
        if not self.records:
            raise ValueError(f"No records found for split '{split}'")

    def _resolve(self, path: str) -> Path:
        if not path or not path.strip():
            raise ValueError("Manifest paths cannot be empty")
        value = Path(path.strip()).expanduser()
        return (value if value.is_absolute() else self.base_directory / value).resolve()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path, mask_path, label = self.records[index]
        with Image.open(image_path) as handle:
            image = handle.convert("RGB").resize(
                (self.image_size, self.image_size), resample=Image.Resampling.BILINEAR
            )
            image_array = np.asarray(image, dtype=np.float32).copy() / 255.0
        with Image.open(mask_path) as handle:
            mask = handle.convert("L").resize(
                (self.image_size, self.image_size), resample=Image.Resampling.NEAREST
            )
            raw_mask = np.asarray(mask, dtype=np.uint8).copy()

        # Binary masks encoded as 0/1 or 0/255 are both supported.
        threshold_values = (
            raw_mask[raw_mask != self.mask_ignore_value]
            if self.mask_ignore_value is not None
            else raw_mask
        )
        detected_max = int(threshold_values.max()) if threshold_values.size else 0
        threshold = (
            self.mask_threshold
            if self.mask_threshold is not None
            else (0 if detected_max <= 1 else 127)
        )
        mask_array = (raw_mask > threshold).astype(np.uint8)
        if self.mask_ignore_value is not None:
            mask_array[raw_mask == self.mask_ignore_value] = 255
        if self.augment:
            if random.random() < 0.5:
                image_array = np.flip(image_array, axis=1)
                mask_array = np.flip(mask_array, axis=1)
            if random.random() < 0.5:
                image_array = np.flip(image_array, axis=0)
                mask_array = np.flip(mask_array, axis=0)
        image_array = np.ascontiguousarray(image_array)
        mask_array = np.ascontiguousarray(mask_array)
        # Standard ImageNet-channel scaling; optional checkpoint initialization is not implied.
        means = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        stds = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (image_array - means) / stds
        return {
            "image": torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))),
            "mask": torch.from_numpy(mask_array.astype(np.int64)),
            "label": torch.tensor(label, dtype=torch.long),
            "name": image_path.name,
        }


def seed_worker(worker_id: int) -> None:
    """Align NumPy and Python RNG streams with a PyTorch data-worker seed."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    seed: int,
    shuffle: bool = False,
    workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=workers > 0,
    )
