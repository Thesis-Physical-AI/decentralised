#!/usr/bin/env bash
# Sets up this repo after cloning: a Python venv with torch + transformers, and the SmolVLM model weights.
#
# Usage:  ./install.sh
# Env:    PYTHON=python3.11 ./install.sh          (interpreter used to create the venv)
#         MODEL_ID=HuggingFaceTB/SmolVLM-500M-Instruct ./install.sh
#         ISAACSIM_PATH=/path/to/isaacsim/_build/linux-x86_64/release ./install.sh
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

PYTHON="${PYTHON:-python3}"
MODEL_ID="${MODEL_ID:-HuggingFaceTB/SmolVLM-Instruct}"
MODEL_DIR="models/$(basename "$MODEL_ID")"
ISAACSIM_PATH="${ISAACSIM_PATH:-$HOME/Projects/isaacsim/_build/linux-x86_64/release}"

echo "==> Creating virtual environment in .venv"
if [ ! -d .venv ]; then
    "$PYTHON" -m venv .venv
fi
.venv/bin/pip install --upgrade pip

echo "==> Installing Python requirements (torch, transformers, ...)"
.venv/bin/pip install -r requirements.txt

echo "==> Downloading $MODEL_ID into $MODEL_DIR (resumes if interrupted)"
# Skip the ONNX exports (~25 GB, unused by torch/transformers). Xet transfers stalled on this
# network, so default to plain HTTP; set HF_HUB_DISABLE_XET=0 to use Xet.
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" .venv/bin/hf download "$MODEL_ID" --local-dir "$MODEL_DIR" --exclude "onnx/*"

echo "==> Checking torch can see the GPU"
.venv/bin/python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

echo "==> Checking for Isaac Sim (needed for hello_puppy.py)"
if [ -x "$ISAACSIM_PATH/python.sh" ]; then
    echo "    found $ISAACSIM_PATH/python.sh"
else
    echo "    WARNING: Isaac Sim not found at $ISAACSIM_PATH"
    echo "    Build Isaac Sim 6.x from source, then rerun with ISAACSIM_PATH=<its release dir> ./install.sh"
fi

cat <<EOF

Done.
  SmolVLM:      source .venv/bin/activate   (model is in $MODEL_DIR)
  Simulation:   $ISAACSIM_PATH/python.sh hello_puppy.py --keep-open
EOF
