#!/bin/bash
# Cosmos-Transfer2.5 (NVIDIA; depth / segmentation / edge / blurred-RGB control -> photoreal video) for the Head2Wrist
# proxy-render A/B test. uv env inside the repo (its own uv.lock, CUDA 12.8 build); checkpoints download on first run
# into the /data HF hub cache (the token stays in ~/.cache/huggingface/token).
# Needs the NVIDIA Open Model License accepted for nvidia/Cosmos-Guardrail1 on Hugging Face.
# Log: third_party/setup_transfer25.log
set -e
source "$(dirname "$0")/../scripts/env.sh"
cd ${H2W_REPO_ROOT}/third_party
[ -d cosmos-transfer2.5 ] || git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git
cd cosmos-transfer2.5
git checkout --detach 2ff49d0
git log -1 --format='commit %H %cd'
uv python install 3.10
[ ! -d .venv ] || { echo "Existing environment: keep it or choose a new checkout"; exit 1; }   # the repo pins 3.13 (.python-version) but its cu128 flash-attn wheel is cp310 only
uv sync --python 3.10 --extra=cu128
source .venv/bin/activate
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
echo "export HF_HUB_CACHE=${H2W_DATA_ROOT}/checkpoints/hf_hub  # set before running inference"
