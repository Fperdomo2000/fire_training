"""Minimal image processor for `TorchvisionEfficientNetForClassification`.

Not a `transformers.BaseImageProcessor` subclass: that class targets PIL RGB
inputs and has no clean story for 6-band TIFF arrays, and reimplementing it
would add more surface than this project needs. This class just knows how
to turn raw pixel arrays into normalized tensors and round-trips its
settings through `preprocessor_config.json`, which is all the training /
inference code here needs -- and is enough metadata for someone else to
reproduce the preprocessing without reading the training script.
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from .band_stats import PIXEL_MAX_VALUE

PREPROCESSOR_CONFIG_FILENAME = "preprocessor_config.json"


class FireImageProcessor:
    def __init__(self, num_channels: int, image_size: int, image_mean: list[float], image_std: list[float]):
        self.num_channels = num_channels
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std

    def preprocess_array(self, array: np.ndarray) -> torch.Tensor:
        """array: (H, W, num_channels) raw pixel values, any numeric dtype."""
        pixel_values = torch.from_numpy(np.asarray(array)).permute(2, 0, 1).float() / PIXEL_MAX_VALUE
        pixel_values = TF.resize(pixel_values, [self.image_size, self.image_size], antialias=True)
        pixel_values = TF.normalize(pixel_values, mean=self.image_mean, std=self.image_std)
        return pixel_values

    def __call__(self, images, return_tensors: str = "pt") -> dict:
        if not isinstance(images, (list, tuple)):
            images = [images]
        batch = []
        for image in images:
            array = np.array(image.convert("RGB")) if isinstance(image, Image.Image) else np.asarray(image)
            batch.append(self.preprocess_array(array))
        return {"pixel_values": torch.stack(batch)}

    def to_dict(self) -> dict:
        return {
            "processor_class": "FireImageProcessor",
            "num_channels": self.num_channels,
            "image_size": self.image_size,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "pixel_max_value": PIXEL_MAX_VALUE,
        }

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        (save_directory / PREPROCESSOR_CONFIG_FILENAME).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_pretrained(cls, load_directory: str | Path) -> "FireImageProcessor":
        data = json.loads((Path(load_directory) / PREPROCESSOR_CONFIG_FILENAME).read_text())
        return cls(
            num_channels=data["num_channels"],
            image_size=data["image_size"],
            image_mean=data["image_mean"],
            image_std=data["image_std"],
        )

    @classmethod
    def from_config(cls, config) -> "FireImageProcessor":
        return cls(
            num_channels=config.num_channels,
            image_size=config.image_size,
            image_mean=config.image_mean,
            image_std=config.image_std,
        )
