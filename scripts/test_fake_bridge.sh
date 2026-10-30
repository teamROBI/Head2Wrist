#!/usr/bin/env bash
# Dedicated, network-isolated container; no real-robot container or topic is accessed.
set -euo pipefail
source "$(dirname "$0")/env.sh"
FEED=$(realpath "$1"); OUT=${2:-$H2W_OUTPUT_ROOT/fake_bridge}; mkdir -p "$OUT"; OUT=$(realpath "$OUT")
NAME=h2w_fake_$$
docker run --rm --name "$NAME" --network none --entrypoint /bin/bash \
  -e ROS_DOMAIN_ID=77 -e ROS_LOCALHOST_ONLY=1 -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp -e PYTHONPATH=/release \
  -v "$H2W_REPO_ROOT:/release:ro" -v "$FEED:/feed.npz:ro" -v "$OUT:/result" \
  "${H2W_ROS_IMAGE:-robotis/ai-worker:2.0.2}" /release/scripts/fake_bridge_inside.sh
