#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Load venv path from config.json
VENV_PATH=$(python3 -c "import json; print(json.load(open('config.json'))['venv_path'])")
source "$VENV_PATH/bin/activate"

python -m firenet.train \
  --dataset-type png \
  --data-root dataset_png \
  --output-dir outputs/efficientnet_v2_s_png
