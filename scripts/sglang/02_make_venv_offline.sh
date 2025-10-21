#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE_DIR:-/path/to/base}
WHEELHOUSE=${WHEELHOUSE_DIR:-$BASE/wheelhouse/cu126}
VENV=${VENV_PATH:-$BASE/venvs/sglang-oss}

python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install -U pip

# Install from local wheelhouse only:
pip install --no-index --find-links="$WHEELHOUSE" \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

pip install --no-index --find-links="$WHEELHOUSE" \
  sglang==0.5.3 huggingface_hub==0.25.2 transformers==4.44.2 tokenizers==0.19.1 \
  safetensors==0.4.3 accelerate==0.33.0 einops pydantic uvloop

# Install kernel wheel matching your CUDA track:
pip install "$WHEELHOUSE/sgl_kernel-0.3.7+cu126-cp310-abi3-manylinux2014_x86_64.whl"

echo "VENV ready at: $VENV"


