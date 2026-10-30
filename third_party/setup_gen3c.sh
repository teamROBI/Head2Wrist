#!/usr/bin/env bash
# GEN3C env for Head2Wrist with uv (no conda): GEN3C/INSTALL.md translated to uv + system CUDA 12.8, adapted for the
# RTX PRO 6000 Blackwell (sm_120) the same way as setup_lyra2.sh: torch 2.7.1 cu128 instead of 2.6.0,
# transformer_engine 2.6.0.post1 instead of 1.12, no apex (GEN3C imports it only for training).
# Source patches marked [Head2Wrist] (grep -rn Head2Wrist cosmos_predict1): TE 2.x moved apply_rotary_pos_emb / SplitAlongDim /
# check_set_window_size; apex imports made optional; prompt-upsampler imports made optional (run with it disabled).
# Env lives at third_party/GEN3C/.venv; download checkpoints separately under ${H2W_DATA_ROOT}/checkpoints/gen3c.
set -e
source "$(dirname "$0")/../scripts/env.sh"
set -o pipefail
[ -d "${H2W_REPO_ROOT}/third_party/GEN3C/.git" ] || git clone https://github.com/nv-tlabs/GEN3C.git "${H2W_REPO_ROOT}/third_party/GEN3C"
git -C "${H2W_REPO_ROOT}/third_party/GEN3C" checkout --detach db2ffe1
cd ${H2W_REPO_ROOT}/third_party/GEN3C
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=16
[ -d .venv ] || uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install pip setuptools wheel packaging ninja cmake pybind11
uv pip install torch==2.7.1 torchvision==0.22.1 --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
SITE=$VIRTUAL_ENV/lib/python3.10/site-packages
export CPATH="$CUDA_HOME/include:$SITE/nvidia/cudnn/include:$SITE/nvidia/nccl/include:$CPATH"
export LD_LIBRARY_PATH="$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
grep -v -E '^(torch|torchvision)==' requirements.txt > .venv/requirements_no_torch.txt
uv pip install -r .venv/requirements_no_torch.txt torch==2.7.1 torchvision==0.22.1 \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
uv pip install "git+https://github.com/microsoft/MoGe.git"
export NVTE_FRAMEWORK=pytorch
uv pip install "transformer_engine_cu12==2.6.0.post1"
uv pip install --no-build-isolation "transformer_engine[pytorch]==2.6.0.post1" "transformer_engine_cu12==2.6.0.post1"
[ -e "$SITE/nvidia/cudart" ] || cp -a "$SITE/nvidia/cuda_runtime" "$SITE/nvidia/cudart"
# MoGe pulls huggingface-hub 1.x (transformers 4.49 needs <1.0); protobuf 6.x for the onnx (gencode 6.31) that TE 2.6 imports
uv pip install "huggingface-hub==0.29.2" "protobuf==6.31.1"
PYTHONPATH=. python -c "
import torch, transformer_engine.pytorch, warp, moge.model.v1
from cosmos_predict1.diffusion.inference.cache_3d import Cache3D_Buffer
print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available(), torch.cuda.get_device_capability())
print('all imports OK')"
echo SETUP_DONE
