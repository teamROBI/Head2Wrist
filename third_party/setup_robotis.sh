#!/usr/bin/env bash
# Download the public robot descriptions and Isaac adapter at the paper revisions.
set -euo pipefail
source "$(dirname "$0")/../scripts/env.sh"
AI_COMMIT=6961f7e7368329d57ebf2adf0fdba08680a43949
LAB_COMMIT=9dd83c5988ea1709b7f232e8922c5c55dcd68218
AI_DIR=${H2W_AI_WORKER_ROOT:-$H2W_REPO_ROOT/third_party/ai_worker}
LAB_DIR=$H2W_ROBOTIS_ROOT
CHECK_ONLY=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --check-only) CHECK_ONLY=1; shift ;;
    --ai-worker-dir) AI_DIR=$2; shift 2 ;;
    --robotis-lab-dir) LAB_DIR=$2; shift 2 ;;
    *) echo "usage: $0 [--check-only] [--ai-worker-dir PATH] [--robotis-lab-dir PATH]" >&2; exit 2 ;;
  esac
done
checkout_pinned() {
  local url=$1 dst=$2 revision=$3
  if [ "$CHECK_ONLY" = 0 ]; then
    if [ ! -d "$dst/.git" ]; then
      [ ! -e "$dst" ] || { echo "Refusing to replace existing directory: $dst" >&2; return 1; }
      GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --no-checkout "$url" "$dst"
    elif [ -n "$(git -C "$dst" status --porcelain --untracked-files=no)" ]; then
      echo "Refusing to change a checkout with local edits: $dst" >&2; return 1
    fi
    git -C "$dst" fetch origin "$revision"
    GIT_LFS_SKIP_SMUDGE=1 git -C "$dst" checkout --detach "$revision"
    if git lfs version >/dev/null 2>&1; then
      git -C "$dst" lfs pull
    fi
  fi
  [ "$(git -C "$dst" rev-parse HEAD)" = "$revision" ] || { echo "Unexpected revision in $dst" >&2; return 1; }
}
checkout_pinned https://github.com/ROBOTIS-GIT/ai_worker.git "$AI_DIR" "$AI_COMMIT"
# The original public robotis_lab URL redirects to its renamed cyclo_lab repository.
checkout_pinned https://github.com/ROBOTIS-GIT/cyclo_lab.git "$LAB_DIR" "$LAB_COMMIT"
if [ "$CHECK_ONLY" = 0 ]; then
  git -C "$LAB_DIR" submodule update --init third_party/IsaacLab
fi
"$H2W_PYTHON" - "$AI_DIR" "$LAB_DIR" <<'PY'
from pathlib import Path
import hashlib,sys
ai,lab=map(Path,sys.argv[1:])
files=[ai/'ffw_description/urdf/ffw_sg2_rev1_follower/ffw_sg2_follower.urdf',
       lab/'source/robotis_lab/robotis_lab/assets/robots/FFW_SG2.py',
       lab/'source/robotis_lab/data/robots/FFW/FFW_SG2.usd']
for path in files:
    if not path.is_file(): raise SystemExit(f'Missing robot asset: {path}')
    data=path.read_bytes()
    if data.startswith(b'version https://git-lfs.github.com/spec/v1'):
        raise SystemExit(f'Unresolved Git LFS pointer: {path}; install git-lfs and run git lfs pull')
    print(f'{path.name}: {len(data)} bytes, sha256 {hashlib.sha256(data).hexdigest()}')
PY
printf 'export H2W_URDF_ROOT=%q\nexport H2W_ROBOTIS_ROOT=%q\n' "$AI_DIR/ffw_description" "$LAB_DIR"
printf '%s\n' 'RobotisLab code/assets: Apache-2.0; Isaac Lab: BSD-3-Clause; Isaac Sim has separate NVIDIA terms.'
