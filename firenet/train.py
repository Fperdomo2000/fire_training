"""Train TorchvisionEfficientNetForClassification on the fire/no_fire folder
datasets (`dataset_png/` -- 3-band, or `dataset_tiff/` -- 6-band).

    python -m firenet.train --dataset-type png  --data-root dataset_png  --output-dir outputs/effnet_v2_s_png
    python -m firenet.train --dataset-type tiff --data-root dataset_tiff --output-dir outputs/effnet_v2_s_tiff

Hyperparameters are defined in firenet/training_config.py. Edit that file to change them.

Writes, under `<output-dir>/final/`: `config.json`, `model.safetensors`,
`preprocessor_config.json`, the copied model/config source files (so the
checkpoint is loadable elsewhere with `trust_remote_code=True`), and, for
the tiff run, `band_stats.json`.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from transformers import EvalPrediction, Trainer, default_data_collator

from . import training_config
from .band_stats import compute_and_save_band_stats
from .configuration_efficientnet import DEFAULT_ID2LABEL, IMAGENET_MEAN, IMAGENET_STD, TorchvisionEfficientNetConfig
from .datasets import FireDataset
from .image_processing import FireImageProcessor
from .metrics import binary_metrics, find_best_threshold, fire_probs
from .modeling_efficientnet import TorchvisionEfficientNetForClassification


def compute_metrics(eval_pred: EvalPrediction) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    probs = fire_probs(logits)
    return binary_metrics(preds, labels, probs=probs)


def resolve_band_stats(data_root: Path, output_dir: Path) -> dict:
    stats_path = output_dir / "band_stats.json"
    if stats_path.exists() and not training_config.recompute_band_stats:
        print(f"Using existing band statistics from {stats_path}")
        return json.loads(stats_path.read_text())
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Computing per-channel training statistics for the infrared bands -> {stats_path}")
    return compute_and_save_band_stats(data_root, stats_path)


def build_datasets(dataset_type: str, data_root: Path, output_dir: Path):
    tiff = dataset_type == "tiff"
    if tiff:
        stats = resolve_band_stats(data_root, output_dir)
        image_mean = IMAGENET_MEAN + stats["mean"][3:]
        image_std = IMAGENET_STD + stats["std"][3:]
    else:
        image_mean, image_std = IMAGENET_MEAN, IMAGENET_STD

    datasets = {
        split: FireDataset(data_root, split, image_mean, image_std, training_config.image_size, tiff)
        for split in ("train", "validation", "test")
    }
    num_channels = 6 if tiff else 3
    return datasets, num_channels, image_mean, image_std


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-type", choices=["png", "tiff"], required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="Path to dataset_png/ or dataset_tiff/")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    datasets, num_channels, image_mean, image_std = build_datasets(args.dataset_type, args.data_root, args.output_dir)
    print(f"train={len(datasets['train'])} validation={len(datasets['validation'])} test={len(datasets['test'])} samples")

    config = TorchvisionEfficientNetConfig(
        num_channels=num_channels,
        image_size=training_config.image_size,
        image_mean=image_mean,
        image_std=image_std,
        id2label=DEFAULT_ID2LABEL,
        label2id={name: idx for idx, name in DEFAULT_ID2LABEL.items()},
    )
    config.register_for_auto_class("AutoConfig")

    model = TorchvisionEfficientNetForClassification(config)
    model.register_for_auto_class("AutoModelForImageClassification")

    trainer = Trainer(
        model=model,
        args=training_config.get_training_args(args.output_dir),
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Pick the fire-probability threshold that maximizes F1 on the validation
    # set, then use it -- not the implicit 0.5 from argmax -- to score the
    # held-out test set. This threshold is what gets saved for inference.
    val_output = trainer.predict(datasets["validation"])
    val_probs = fire_probs(val_output.predictions)
    threshold, val_metrics = find_best_threshold(val_probs, val_output.label_ids)
    print(f"Best threshold from validation sweep: {threshold} -> {val_metrics}")

    test_output = trainer.predict(datasets["test"])
    test_probs = fire_probs(test_output.predictions)
    test_metrics = binary_metrics((test_probs >= threshold).astype(int), test_output.label_ids)
    test_metrics["threshold"] = threshold
    print("Test metrics:", test_metrics)

    config.classification_threshold = threshold

    final_dir = args.output_dir / "final"
    trainer.save_model(str(final_dir))

    FireImageProcessor.from_config(config).save_pretrained(final_dir)

    band_stats_path = args.output_dir / "band_stats.json"
    if args.dataset_type == "tiff" and band_stats_path.exists():
        shutil.copy(band_stats_path, final_dir / "band_stats.json")

    (final_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    (final_dir / "validation_metrics.json").write_text(json.dumps({**val_metrics, "threshold": threshold}, indent=2))
    print(f"Saved final model, image processor, and metadata to {final_dir}")


if __name__ == "__main__":
    main()
