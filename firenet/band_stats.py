"""Auxiliary function: per-channel training-set statistics for the IR bands.

Neither the pretrained EfficientNet stem nor ImageNet mean/std say anything
about bands 4-6 (infrared) of the 6-band TIFF dataset, so we standardize
them to zero mean / unit variance using stats measured on the training
split itself, then feed the resulting z-scored channels into the extra stem
input channels built by `firenet.weight_init`.
"""

import json
from pathlib import Path

import numpy as np
import tifffile

# Raw pixel values are divided by this before accumulating stats / feeding
# the model, so everything -- 8-bit RGB and however the IR bands are stored
# -- lands on a comparable float scale. Adjust if your TIFFs use a different
# bit depth.
PIXEL_MAX_VALUE = 255.0
NUM_CHANNELS = 6


def compute_and_save_band_stats(dataset_root: Path, output_path: Path) -> dict:
    """Single-pass per-channel mean/std over dataset_root/train/{fire,no_fire}/*.tif(f)."""
    paths = [
        path
        for class_name in ("fire", "no_fire")
        for pattern in ("*.tif", "*.tiff")
        for path in (Path(dataset_root) / "train" / class_name).glob(pattern)
    ]

    channel_sum = np.zeros(NUM_CHANNELS, dtype=np.float64)
    channel_sqsum = np.zeros(NUM_CHANNELS, dtype=np.float64)
    n_pixels = 0

    for path in paths:
        flat = tifffile.imread(path).astype(np.float64).reshape(-1, NUM_CHANNELS) / PIXEL_MAX_VALUE
        channel_sum += flat.sum(axis=0)
        channel_sqsum += (flat**2).sum(axis=0)
        n_pixels += flat.shape[0]

    mean = channel_sum / n_pixels
    std = np.sqrt(np.maximum(channel_sqsum / n_pixels - mean**2, 1e-8))

    stats = {"mean": mean.tolist(), "std": std.tolist()}
    Path(output_path).write_text(json.dumps(stats, indent=2))
    return stats
