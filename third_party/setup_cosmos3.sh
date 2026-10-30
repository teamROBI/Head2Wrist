#!/bin/bash
# Cosmos 3 (NVIDIA, 2026; Cosmos3-Nano 16B) video transfer (depth / segmentation / edge / blur control -> video) for
# the Head2Wrist proxy-render A/B test, served with vLLM-Omni (recipes/cosmos3/Cosmos3-Nano.md). uv env; weights (~35 GB,
# not gated) in the /data HF hub cache. Run the server with --no-guardrails (the guardrail model is a separate gated repo).
# Log: third_party/setup_cosmos3.log
set -e
source "$(dirname "$0")/../scripts/env.sh"
cd ${H2W_REPO_ROOT}/third_party
mkdir -p cosmos3 && cd cosmos3
[ -d .venv ] || uv venv --python 3.12 --seed .venv
source .venv/bin/activate
# vLLM 0.30.0 default wheel is CUDA 13 (needs driver >= 580; this machine has 570 / CUDA 12.8): use the cu129 release wheel
# (CUDA 12.9 runs on the 12.8 driver via minor-version compatibility; tested with torch 2.13.0+cu129)
uv pip install "https://github.com/vllm-project/vllm/releases/download/v0.30.0/vllm-0.30.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" "vllm-omni==0.30.0" --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match
uv pip install --reinstall torchcodec --index-url https://download.pytorch.org/whl/cu129   # the default torchcodec is a CUDA 13 build (libnvrtc.so.13)
python -c "import torch, vllm; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'vllm', vllm.__version__, torch.cuda.get_device_capability(0))"
python -c "import vllm_omni; print('vllm_omni', getattr(vllm_omni, '__version__', '?'))"
export HF_HUB_CACHE=${H2W_DATA_ROOT}/checkpoints/hf_hub
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('nvidia/Cosmos3-Nano'))"
