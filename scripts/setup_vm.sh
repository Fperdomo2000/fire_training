#!/usr/bin/env bash
# One-shot bootstrap for a fresh Ubuntu 22.04 VM with an NVIDIA T4 GPU:
# installs the NVIDIA driver, sets up a pyenv-managed virtualenv, installs
# project dependencies, then runs both training scripts.
#
# Re-runnable: if the driver install needs a reboot, re-run this script
# after rebooting -- it picks up where it left off and won't repeat
# completed steps.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_VERSION="3.11.9"
VENV_NAME="firenet"

log() { printf '\n=== %s ===\n' "$1"; }

# 1. NVIDIA driver
log "Checking NVIDIA driver"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
else
    echo "No working NVIDIA driver detected. Installing..."
    sudo apt-get update
    sudo apt-get install -y ubuntu-drivers-common
    sudo ubuntu-drivers autoinstall
    echo
    echo "Driver installed but not loaded yet. Reboot the VM, then re-run this script:"
    echo "  sudo reboot"
    echo "  ./scripts/setup_vm.sh"
    exit 0
fi

# 2. System build dependencies (needed to build Python via pyenv)
log "Installing system build dependencies"
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    build-essential curl git wget ca-certificates \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
    libffi-dev liblzma-dev

# 3. pyenv (install if missing, reuse if present)
log "Setting up pyenv"
export PYENV_ROOT="$HOME/.pyenv"
if [ ! -d "$PYENV_ROOT" ]; then
    curl -fsSL https://pyenv.run | bash
fi
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

if ! pyenv versions --bare | grep -qx "$PYTHON_VERSION"; then
    pyenv install "$PYTHON_VERSION"
fi

# 4. Virtualenv: reuse if it exists, create otherwise
log "Setting up virtualenv '$VENV_NAME'"
if ! pyenv versions --bare | grep -qx "$VENV_NAME"; then
    pyenv virtualenv "$PYTHON_VERSION" "$VENV_NAME"
fi
pyenv activate "$VENV_NAME"

# 5. Python dependencies
log "Installing Python dependencies"
pip install --upgrade pip
pip install -r requirements.txt

python - <<'EOF'
import torch
print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
EOF

# 6. Datasets are expected to already be in place next to this project
for d in dataset_png dataset_tiff; do
    if [ ! -d "$d" ]; then
        echo "WARNING: $d/ not found next to this project -- copy it in before training."
    fi
done

# 7. Train
log "Training PNG model"
./scripts/train_png.sh

log "Training TIFF model"
./scripts/train_tiff.sh

log "Done"
