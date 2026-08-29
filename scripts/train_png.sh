#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Load venv path from config.json
source ./train-venv/bin/activate

python -m firenet.train \
  --dataset-type png \
  --data-root datasets/dataset_png \
  --output-dir outputs/efficientnet_v2_s_png
