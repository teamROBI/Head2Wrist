#!/bin/bash
# LeRobot 0.3.3 (as in the robotis_lab image) for the sim policy test: training + the policy server (sim_policy/)
set -e
source "$(dirname "$0")/../../scripts/env.sh"
cd "$(dirname "$0")"
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python "torch==2.7.1" "torchvision==0.22.1" --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python "lerobot==0.3.3"
.venv/bin/python -c "import lerobot, torch; print('lerobot', lerobot.__version__, 'torch', torch.__version__, torch.cuda.is_available())"
