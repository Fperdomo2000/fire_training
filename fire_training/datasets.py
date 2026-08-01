"""Torch Dataset for the `dataset_png/` (3-band) and `dataset_tiff/` (6-band)
fire / no_fire folder layouts:

    <root>/<split>/fire/*
    <root>/<split>/no_fire/*

Returns `{"pixel_values": FloatTensor[C, H, W], "labels": int}`, which is
what `TorchvisionEfficientNetForClassification.forward` and
`transformers.default_data_collator` expect.
"""

import random
from pathlib import Path

import tifffile
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .band_stats import PIXEL_MAX_VALUE

CLASS_TO_LABEL = {"no_fire": 0, "fire": 1}


def discover_samples(root: Path, split: str, extensions: tuple[str, ...]) -> list[tuple[Path, int]]:
    samples = [
        (path, label)
        for class_name, label in CLASS_TO_LABEL.items()
        for path in sorted((Path(root) / split / class_name).iterdir())
        if path.suffix.lower() in extensions
    ]
    if not samples:
        raise FileNotFoundError(f"No samples found under {root}/{split}")
    return samples


class FireDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        image_mean: list[float],
        image_std: list[float],
        image_size: int,
        tiff: bool, # to know whether png or tiff
    ):
        extensions = (".tif", ".tiff") if tiff else (".png",)
        self.samples = discover_samples(root, split, extensions)
        self.tiff = tiff
        self.image_size = image_size
        self.mean = image_mean
        self.std = image_std
        self.augment = split == "train"

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, path: Path) -> torch.Tensor:
        if self.tiff:
            array = tifffile.imread(path)  # (H, W, 6)
            return torch.from_numpy(array).permute(2, 0, 1).float() / PIXEL_MAX_VALUE
        return TF.to_tensor(Image.open(path).convert("RGB"))  # [3, H, W], float in [0, 1]

    def __getitem__(self, idx: int) -> dict:
        path, label = self.samples[idx]
        pixel_values = self._load(path)
        pixel_values = TF.resize(pixel_values, [self.image_size, self.image_size], antialias=True)
        if self.augment and random.random() < 0.5:
            pixel_values = TF.hflip(pixel_values)
        pixel_values = TF.normalize(pixel_values, mean=self.mean, std=self.std)
        return {"pixel_values": pixel_values, "labels": label}
