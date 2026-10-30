#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
exec "$H2W_LEROBOT_PYTHON" -m head2wrist.robot.offline_real_check "$@"
