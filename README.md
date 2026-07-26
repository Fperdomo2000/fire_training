# firenet

Hugging Face training pipeline for fire / no_fire classification with
`torchvision.models.efficientnet_v2_s` (`EfficientNet_V2_S_Weights.IMAGENET1K_V1`),
wrapped as a custom `transformers` model so training, checkpoints, and
inference all go through the standard HF APIs (`Trainer`, `save_pretrained`,
`from_pretrained`, safetensors).

Two datasets are supported, expected side by side with this project:

```
dataset_png/           dataset_tiff/            (3-band RGB PNGs / 6-band RGB+IR TIFFs)
  train/fire/*.png        train/fire/*.tif
  train/no_fire/*.png     train/no_fire/*.tif
  validation/...           validation/...
  test/...                 test/...
```

One model is trained per dataset, from the same architecture:

- **PNG (3-band)**: stock pretrained EfficientNetV2-S, untouched stem, ImageNet
  normalization.
- **TIFF (6-band)**: the stem's first conv layer is replaced with a 6-input
  `Conv2d`. Channels 1-3 (RGB) keep the pretrained weights; channels 4-6
  (infrared) are initialized with standardized-normal (N(0, 1)) weights,
  matching the zero-mean/unit-variance standardization applied to those
  bands (see below).

## Layout

- `firenet/training_config.py` — all training hyperparameters (image size,
  epochs, batch size, learning rate, etc.) plus `get_training_args()`, which
  builds the `TrainingArguments` instance from them. **Edit this file to
  change training settings.**
- `firenet/configuration_efficientnet.py` — `TorchvisionEfficientNetConfig`.
- `firenet/modeling_efficientnet.py` — `TorchvisionEfficientNetForClassification`.
- `firenet/weight_init.py` — builds the 6-channel stem conv, kept separate
  from the model class so the extra-channel init scheme can be changed later.
- `firenet/band_stats.py` — auxiliary function that computes per-channel
  mean/std over the TIFF training split, used to standardize the infrared
  bands (`compute_and_save_band_stats`).
- `firenet/datasets.py` — `FireDataset`, a torch `Dataset` over the
  `<root>/<split>/{fire,no_fire}/` layout (PNG or TIFF).
- `firenet/image_processing.py` — `FireImageProcessor`, saved to
  `preprocessor_config.json` so a checkpoint documents its own preprocessing.
- `firenet/metrics.py` — binary classification metrics (accuracy, precision,
  recall, f1) and the validation-set threshold sweep (`find_best_threshold`).
- `firenet/train.py` — training entrypoint (`python -m firenet.train ...`).
- `scripts/train_png.sh`, `scripts/train_tiff.sh` — convenience wrappers.

## Setup

```bash
pip install -r requirements.txt
```

## Train

1. **Edit `firenet/training_config.py`** to set hyperparameters (image size,
   epochs, batch size, learning rate, etc.).

2. Run training:

```bash
python -m firenet.train --dataset-type png  --data-root dataset_png  --output-dir outputs/efficientnet_v2_s_png
python -m firenet.train --dataset-type tiff --data-root dataset_tiff --output-dir outputs/efficientnet_v2_s_tiff
```

or use the convenience wrappers:

```bash
scripts/train_png.sh
scripts/train_tiff.sh
```

For the tiff run, `band_stats.json` (per-channel mean/std over the training
split) is computed automatically on first run and cached under
`<output-dir>/band_stats.json`; set `recompute_band_stats = True` in
`training_config.py` to force a refresh.

After training, the fire-probability threshold is tuned by sweeping the
validation set for the value that maximizes F1 (`firenet.metrics.find_best_threshold`);
that threshold -- not a fixed 0.5 -- is then used to compute the final test
metrics (accuracy, precision, recall, f1), and is saved as
`config.classification_threshold` for inference.

## Output

Each run writes `<output-dir>/final/` with everything needed to reload and
use the model elsewhere:

- `config.json` (architecture, labels, image size/normalization,
  `classification_threshold`, `auto_map` for `trust_remote_code` loading)
- `model.safetensors`
- `preprocessor_config.json`
- `configuration_efficientnet.py`, `modeling_efficientnet.py`,
  `weight_init.py` — copied automatically by `save_pretrained` so the
  checkpoint is self-contained
- `band_stats.json` (tiff only)
- `validation_metrics.json`, `test_metrics.json` — accuracy/precision/recall/f1
  at the tuned threshold

Reload from anywhere, without needing this project on the path:

```python
import torch
from transformers import AutoConfig, AutoModelForImageClassification

path = "outputs/efficientnet_v2_s_tiff/final"
config = AutoConfig.from_pretrained(path, trust_remote_code=True)
model = AutoModelForImageClassification.from_pretrained(path, trust_remote_code=True).eval()

with torch.no_grad():
    probs = torch.softmax(model(pixel_values=pixel_values).logits, dim=-1)[:, 1]  # P(fire)
is_fire = probs >= config.classification_threshold
```
