#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
N=$1; shift
W=/workspace/robotis_lab
mkdir -p "$H2W_OUTPUT_ROOT/sim_policy/$N"
docker run --rm --name "${CNAME:-h2w_collect_${N}_$$}" \
  --gpus "${H2W_GPU_DEVICE:-all}" --network host --shm-size 16g --entrypoint /bin/bash \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ALLOW_ROOT=1 \
  -e ISAACSIM_PATH=/isaac-sim -e ISAACSIM_ROOT_PATH=/isaac-sim \
  -e ISAACLAB_PATH="$W/third_party/IsaacLab" -e ROBOTISLAB_PATH="$W" \
  -e "PYTHONPATH=/release:$W/source/robotis_lab" -e PYTHONDONTWRITEBYTECODE=1 \
  -e NVIDIA_DRIVER_CAPABILITIES=all -e HOST_UID="$(id -u)" \
  -v "$H2W_ROBOTIS_ROOT:$W:ro" -v "$H2W_REPO_ROOT:/release:ro" \
  -v "$H2W_OUTPUT_ROOT/sim_twin:/twin:ro" -v "$H2W_OUTPUT_ROOT/sim_policy:/policy" \
  "${H2W_ISAAC_IMAGE:-robotis/robotis-lab:latest}" \
  /isaac-sim/python.sh /release/head2wrist/policy/collect_demos.py \
  --twin "/twin/$N" --plans "/policy/$N/${PLANS:-plans}.json" \
  --out "/policy/$N/${OUTDIR:-demos}" "$@"
