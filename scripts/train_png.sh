#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python -m firenet.train \
  --dataset-type png \
  --data-root dataset_png \
  --output-dir outputs/efficientnet_v2_s_png
