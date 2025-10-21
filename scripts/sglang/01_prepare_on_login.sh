#!/usr/bin/env bash
set -euo pipefail

# ---------- paths ----------
BASE=${BASE_DIR:-/path/to/base}
WHEELHOUSE=$BASE/wheelhouse/cu126          # change to cu128 if needed
HF_MODELS=$BASE/hf_models
HF_HOME=${HF_HOME:-$BASE/.cache/huggingface}
mkdir -p "$WHEELHOUSE" "$HF_MODELS" "$HF_HOME"

# ---------- pick CUDA track (H100 == cu126 by default) ----------
CUDA_INDEX=${CUDA_INDEX:-https://download.pytorch.org/whl/cu126}   # use cu128 if needed

# ---------- clean python user env ----------
python3.12 -m venv /tmp/_tmpvenv && source /tmp/_tmpvenv/bin/activate
pip install -U pip wheel

# ---------- download PyTorch 2.8 & friends into local wheelhouse ----------
pip download -d "$WHEELHOUSE" --index-url "$CUDA_INDEX" \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

# ---------- download SGLang & runtime deps (pure Python wheels from PyPI) ----------
pip download -d "$WHEELHOUSE" \
  sglang==0.4.10 \
  huggingface_hub==0.25.2 transformers==4.44.2 tokenizers==0.19.1 \
  safetensors==0.4.3 accelerate==0.33.0 einops pydantic uvloop

# ---------- download sgl-kernel (CUDA kernels) ----------
# For cu126 (H100 typical):
wget -O "$WHEELHOUSE/sgl_kernel-0.3.7+cu126-cp310-abi3-manylinux2014_x86_64.whl" \
  https://github.com/sgl-project/whl/releases/download/v0.3.7/sgl_kernel-0.3.7+cu126-cp310-abi3-manylinux2014_x86_64.whl

# If you need cu128 instead, use this one (and set CUDA_INDEX to cu128 above):
# wget -O "$WHEELHOUSE/sgl_kernel-0.3.7+cu128-cp310-abi3-manylinux2014_x86_64.whl" \
#  https://github.com/sgl-project/whl/releases/download/v0.3.7/sgl_kernel-0.3.7+cu128-cp310-abi3-manylinux2014_x86_64.whl

# ---------- pre-download the model snapshot to shared storage ----------
# MXFP4 (preferred on H100):
python -m pip install -q "huggingface_hub==0.25.2"
huggingface-cli login --token "${HF_TOKEN:-}" || true  # or ensure your token is already cached
hf download openai/gpt-oss-120b \
  --local-dir "$HF_MODELS/gpt-oss-120b" --local-dir-use-symlinks False

# Optional: also fetch BF16 variant (useful on A100 fallback)
# hf download lmsys/gpt-oss-120b-bf16 \
#   --local-dir "$HF_MODELS/gpt-oss-120b-bf16" --local-dir-use-symlinks False

echo "All wheels in: $WHEELHOUSE"
echo "Model snapshot in: $HF_MODELS/gpt-oss-120b"


