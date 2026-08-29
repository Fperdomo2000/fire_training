#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.." # Eso es moverse al padre del directorio donde estás parado

# Load venv
source ./train-venv/bin/activate

python -m firenet.train \
  --dataset-type tiff \
  --data-root datasets/dataset_tiff \
  --output-dir outputs/efficientnet_v2_s_tiff
