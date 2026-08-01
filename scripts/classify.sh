#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Load venv path from config.json
VENV_PATH=$(python3 -c "import json; print(json.load(open('config.json'))['venv_path'])")
source "$VENV_PATH/bin/activate"

# Pass all arguments to the classify script
python -m firenet.classify "$@"
