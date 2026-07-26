#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python -m firenet.train \
  --dataset-type tiff \
  --data-root dataset_tiff \
  --output-dir outputs/efficientnet_v2_s_tiff
