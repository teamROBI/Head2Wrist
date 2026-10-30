#!/bin/bash
# Wan2.1 VACE (Alibaba, ICCV 2025; control video + mask + reference image -> video) for the Head2Wrist proxy A/B test,
# via diffusers (WanVACEPipeline). uv env; weights (Wan2.1-VACE-14B-diffusers, 75 GB, not gated) in the /data HF hub
# cache. torch 2.7.1 + CUDA 12.8 (this machine: driver 570 / CUDA 12.8, RTX PRO 6000 Blackwell).
# Log: third_party/setup_wan_vace.log
set -e
source "$(dirname "$0")/../scripts/env.sh"
cd ${H2W_REPO_ROOT}/third_party
mkdir -p wan_vace && cd wan_vace
[ -d .venv ] || uv venv --python 3.11 --seed .venv
source .venv/bin/activate
uv pip install torch==2.7.1 torchvision==0.22.1 --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
uv pip install "diffusers>=0.35" "transformers>=4.51" accelerate ftfy imageio imageio-ffmpeg opencv-python-headless sentencepiece
python -c "import torch, diffusers; from diffusers import WanVACEPipeline; print('torch', torch.__version__, 'diffusers', diffusers.__version__, torch.cuda.get_device_capability(0))"
export HF_HUB_CACHE=${H2W_DATA_ROOT}/checkpoints/hf_hub
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Wan-AI/Wan2.1-VACE-14B-diffusers', max_workers=8))"
