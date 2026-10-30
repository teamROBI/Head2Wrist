#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
for EP in "$@"; do
  N=${EP//\//_}; D="$H2W_OUTPUT_ROOT/sim_twin/$N"; mkdir -p "$D/isaac"
  "$H2W_PYTHON" -m head2wrist.simulation.export_twin --ep "$EP"
  "$(dirname "$0")/run_isaac.sh" "$N" > "$D/isaac/run.log" 2>&1
done
