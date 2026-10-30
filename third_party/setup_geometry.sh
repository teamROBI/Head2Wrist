#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../scripts/env.sh"
# Download upstream source separately; it remains outside this repository's tracked code.
mkdir -p "$H2W_REPO_ROOT/third_party"
DA3="$H2W_REPO_ROOT/third_party/Depth-Anything-3"
[ -d "$DA3/.git" ] || git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git "$DA3"
uv pip install --python "$H2W_PYTHON" -e "$DA3"
# Install pinned URDF/USD assets with setup_robotis.sh.
S3D="$H2W_SAM3D_ROOT"
[ -d "$S3D/.git" ] || git clone https://github.com/facebookresearch/sam-3d-objects.git "$S3D"
git -C "$S3D" checkout --detach a0016abefdb519f98242d90a35c18dc04f9f6af2
printf '%s\n' 'Follow the SAM 3D upstream environment and model setup, then set H2W_SAM3D_PYTHON.'
