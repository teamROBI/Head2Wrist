#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
exec "$H2W_PYTHON" -m head2wrist.pipeline.run_episode "$@"
