#!/usr/bin/env bash
set -euo pipefail
set +u
source /opt/ros/jazzy/setup.bash
set -u
export PYTHONPATH="/release${PYTHONPATH:+:$PYTHONPATH}"
P=/h2w_release_test
python3 -m head2wrist.robot.fake_sg2 --feed /feed.npz --prefix "$P" --states command --max_s 80 > /result/fake.log 2>&1 &
FP=$!
trap 'kill -TERM "$FP" 2>/dev/null || true; wait "$FP" 2>/dev/null || true' EXIT
sleep 3
TOPICS=(--side right --state_topic "$P/joint_states" --image_topic "$P/camera_right/camera_right/color/image_rect_raw/compressed" --cmd_topic "$P/leader/joint_trajectory_command_broadcaster_right/joint_trajectory" --lift_topic "$P/leader/joystick_controller_right/joint_trajectory" --cmd_topic_fmt "$P/leader/joint_trajectory_command_broadcaster_{side}/joint_trajectory" --head_topic "$P/leader/joystick_controller_left/joint_trajectory")
python3 -m head2wrist.robot.egoh2w_bridge "${TOPICS[@]}" --mode record --record_topic "$P/camera_{side}/camera_{side}/color/image_rect_raw/compressed" --record_s 1 --dump /result/record > /result/record.log 2>&1
python3 -m head2wrist.robot.egoh2w_bridge "${TOPICS[@]}" --mode preset --safe_left 0.3,0,0,0,0,0,0 --safe_right 0.3,0,0,0,0,0,0 --set_lift=-0.03 --max_step 0.03 --dump /result/preset > /result/preset.log 2>&1
set +e
python3 -m head2wrist.robot.egoh2w_bridge "${TOPICS[@]}" --mode goto --safe_left 0,0,0,0,0,0,0 --safe_right 0,0,0,0,0,0,0 --set_lift=-0.08 > /result/refused.log 2>&1
RC=$?
set -e
[ "$RC" = 2 ] || { echo "lift refusal expected exit 2, got $RC"; exit 1; }
python3 -c 'import os,signal,sys; signal.signal(signal.SIGINT,signal.SIG_DFL); os.execvp("python3", ["python3", "-m", "head2wrist.robot.egoh2w_bridge", *sys.argv[1:]])' "${TOPICS[@]}" --mode preset --safe_left 0.8,0,0,0,0,0,0 --safe_right 0.8,0,0,0,0,0,0 --max_step 0.005 --set_lift=-0.02 > /result/interrupt.log 2>&1 &
BP=$!
sleep 2
kill -INT "$BP"
wait "$BP" || true
python3 - <<'PY'
from pathlib import Path
import json, numpy as np
root=Path('/result'); r=np.load(root/'record/record.npz')
assert len(r['q_right']) > 0, 'no fake states recorded'
logs={p.stem:p.read_text() for p in root.glob('*.log')}
assert 'hold' in logs['interrupt'].lower(), logs['interrupt'][-1000:]
json.dump(dict(ros_domain=77,network='none',recorded_states=len(r['q_right']),preset_exit=0,unsafe_lift_exit=2,interrupt_hold=True),open(root/'result.json','w'),indent=2)
print('fake follower smoke passed')
PY
