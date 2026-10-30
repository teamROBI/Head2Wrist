#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
RUN=$1; COUNT=${2:-30}; PORT=${3:-6123}; TAG=${TAG:-_in}; SCENE=${SCENE:-stack_48}
D="$H2W_OUTPUT_ROOT/sim_policy"; RD="$D/train/$RUN"; CK=${CKPT:-$RD/checkpoints/last/pretrained_model}
mkdir -p "$RD"
"$H2W_LEROBOT_PYTHON" -m head2wrist.policy.policy_server --ckpt "$CK" --port "$PORT"   --repr eef_hybrid --twin "$H2W_OUTPUT_ROOT/sim_twin/$SCENE/twin.json" ${SERVER_ARGS:-} > "$RD/server$TAG.log" 2>&1 &
SP=$!
trap 'kill "$SP" 2>/dev/null || true' EXIT
for i in $(seq 120); do
  grep -q listening "$RD/server$TAG.log" && break
  kill -0 "$SP" || { cat "$RD/server$TAG.log"; exit 1; }
  sleep 2
done
PLANS=${EVAL_PLANS:-plans_eval} OUTDIR=eval_$RUN$TAG "$(dirname "$0")/run_collect_named.sh" "$SCENE"   --policy "localhost:$PORT" --first "${FIRST:-0}" --count "$COUNT" --head_pitch 0.69 > "$RD/eval$TAG.log" 2>&1
"$H2W_PYTHON" - "$D/$SCENE/eval_$RUN$TAG" "$RD/eval_summary$TAG.json" "$RUN" <<'ENDPY'
import glob,json,sys
rows=[r for p in glob.glob(sys.argv[1]+'/collect_*.json') for r in json.load(open(p))]
result=dict(run=sys.argv[3],n=len(rows),success=sum(r['success'] for r in rows),episodes=rows)
json.dump(result,open(sys.argv[2],'w'),indent=2)
print(f"{result['success']}/{result['n']} loose successes; run rescore.sh for paper scoring")
ENDPY
