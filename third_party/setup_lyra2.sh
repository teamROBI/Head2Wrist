#!/usr/bin/env bash
# Lyra-2 env for Head2Wrist with uv (no conda): Lyra-2/INSTALL.md translated to uv + system CUDA 12.8,
# adapted for RTX PRO 6000 Blackwell (sm_120). Env lives at third_party/lyra/Lyra-2/.venv
set -e
source "$(dirname "$0")/../scripts/env.sh"
set -o pipefail
[ -d "${H2W_REPO_ROOT}/third_party/lyra/.git" ] || git clone https://github.com/nv-tlabs/lyra.git "${H2W_REPO_ROOT}/third_party/lyra"
git -C "${H2W_REPO_ROOT}/third_party/lyra" checkout --detach 9fffc9a
cd ${H2W_REPO_ROOT}/third_party/lyra/Lyra-2
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
uv pip install --no-deps -r requirements.txt
uv pip install "git+https://github.com/microsoft/MoGe.git"
# pin TE to the torch-2.7 era (2.19 needs torch>=2.8 headers and pulls the CUDA-13 core)
export NVTE_FRAMEWORK=pytorch
uv pip install "transformer_engine_cu12==2.6.0.post1"
uv pip install --no-build-isolation "transformer_engine[pytorch]==2.6.0.post1" "transformer_engine_cu12==2.6.0.post1"
[ -e "$SITE/nvidia/cudart" ] || cp -a "$SITE/nvidia/cuda_runtime" "$SITE/nvidia/cudart"
# flash-attn is only needed for its Triton rotary helper (attention falls back to SDPA on sm_120):
# prebuilt wheels first, source build only if both fail
FA=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3
uv pip install "$FA/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl" \
 || uv pip install "$FA/flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" \
 || uv pip install --no-build-isolation flash-attn==2.8.3
USE_SYSTEM_EIGEN=1 uv pip install --no-build-isolation -e 'lyra_2/_src/inference/vipe'
# build helpers needed by DA3's hatchling backend under --no-build-isolation, plus a runtime dep DA3 omits
uv pip install pathspec hatchling hatch-vcs editables trove-classifiers addict "gdown==5.2.0"   # gdown 6 dropped fuzzy=, which ViPE uses
uv pip install --no-build-isolation -e 'lyra_2/_src/inference/depth_anything_3[gs]'
PYTHONPATH=. python -c "
import torch, flash_attn, transformer_engine.pytorch, vipe_ext, depth_anything_3.api, moge.model.v1, gsplat
print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available(), torch.cuda.get_device_capability())
print('all imports OK')"
echo SETUP_DONE
