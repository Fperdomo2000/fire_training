"""Run a trained TorchvisionEfficientNetForClassification checkpoint over a
folder of images and write fire/no_fire predictions to a CSV.

    python -m firenet.classify --model-path outputs/efficientnet_v2_s_png/final  --input dataset_png/test  --output predictions.csv
    python -m firenet.classify --model-path outputs/efficientnet_v2_s_tiff/final --input some_folder      --output predictions_png.csv

The image format (.png vs .tif/.tiff) is inferred from the checkpoint's
`num_channels` (3 or 6), so --input just needs to be a file or a folder to
search recursively. Predictions use `config.classification_threshold`, the
F1-optimal threshold picked during training -- not a fixed 0.5.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForImageClassification

from .image_processing import FireImageProcessor
from .metrics import fire_probs


def find_images(root: Path, extensions: tuple[str, ...]) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in extensions)


def load_array(path: Path, tiff: bool) -> np.ndarray:
    if tiff:
        return tifffile.imread(path)  # (H, W, 6), uint16
    return np.array(Image.open(path).convert("RGB"))  # (H, W, 3), uint8


def preprocess_array(array: np.ndarray, image_size: int, image_mean: list[float], image_std: list[float]) -> torch.Tensor:
    """Preprocess image array with proper normalization for uint8 or uint16."""
    from torchvision.transforms import functional as TF

    tensor = torch.from_numpy(array).permute(2, 0, 1).float()

    # Normalize based on dtype: uint16 -> 65535, uint8 -> 255
    if array.dtype == np.uint16:
        tensor = tensor / 65535.0
    elif array.dtype == np.uint8:
        tensor = tensor / 255.0
    else:
        tensor = tensor / tensor.max()

    tensor = TF.resize(tensor, [image_size, image_size], antialias=True)
    tensor = TF.normalize(tensor, mean=image_mean, std=image_std)
    return tensor


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", type=Path, required=True, help="Path to a <output-dir>/final/ checkpoint")
    parser.add_argument("--input", type=Path, required=True, help="Image file, or folder to search recursively")
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV path for the results")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForImageClassification.from_pretrained(args.model_path, trust_remote_code=True).to(device).eval()
    processor = FireImageProcessor.from_pretrained(args.model_path)

    tiff = config.num_channels == 6
    extensions = (".tif", ".tiff") if tiff else (".png",)
    paths = [args.input] if args.input.is_file() else find_images(args.input, extensions)
    if not paths:
        raise FileNotFoundError(f"No {extensions} images found under {args.input}")

    rows = []
    for start in range(0, len(paths), args.batch_size):
        batch_paths = paths[start : start + args.batch_size]
        pixel_values = torch.stack([
            preprocess_array(load_array(path, tiff), processor.image_size, processor.image_mean, processor.image_std)
            for path in batch_paths
        ]).to(device)

        with torch.no_grad():
            logits = model(pixel_values=pixel_values).logits.cpu().numpy()

        probs = fire_probs(logits)
        for path, prob in zip(batch_paths, probs):
            is_fire = prob >= config.classification_threshold
            label = "fire" if is_fire else "no_fire"
            rows.append({"path": str(path), "prob_fire": float(prob), "prediction": label})
            print(f"{path}: {label} (p={prob:.3f})")

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "prob_fire", "prediction"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
