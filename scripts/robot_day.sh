#!/bin/bash
# The real-robot session, step by step (run in this order). The bridge runs inside the ROBOTIS ai_worker container as a
# zenoh client of the robot (as help_scripts/control/run_keyboard_control.sh); the policy server runs on this host.
#                                         publishing on the arm's command topic (a running leader / teleop would fight the policy)
#                                         (keyboard / leader), keep the other still -> which topic shows which arm (about half of
#                                         the recorded SG2 data had them swapped); shadow / execute use the result
#                                         the robot by hand, and the start joints of demo k
#                                         estimate of the base error where the scene has texture); cross-check with the tape measure
#                                         every pose -> generated-vs-real view metrics + the plan's open-loop outcome (trial log)
#                                         automatic, ~1 rad/s) -> base alignment in the keyboard GUI (live right-wrist view over the
#                                         generated start view; ALIGN=0 skips) -> place the cube -> g -> policy (MAX_S, 120 s) -> back.
#                                         Ctrl+C holds the robot where it is
# Env: ROBOT_IP (ROBOT_HOST), CONTAINER (ai_worker), ROS_DOMAIN_ID (30), MAX_S (40), PORT (6123), BRIDGE_ARGS (extra args),
#      GRIP_CLOSE (0.9): replay / execute raise a 'close' gripper command to this, so the gripper squeezes the object
#      (the plan / training close only to the measured width); GRIP_STATE_MAX (default: the training set's most-closed gripper): the gripper value the policy sees is capped there
#      robot_setup.json's table_h_m (0.74), so the PHYSICAL lift is the plan's + (TABLE_H - table_h_m) - a pure z shift of
#      the arm (lift_joint is prismatic z) that keeps the arm-to-table geometry, joints and wrist views as planned; the
#      policy server keeps the plan's lift (its FK / IK model)
set -u
source "$(dirname "$0")/env.sh"
CMD=${1:-help}; TASK=${2:-}; K=${3:-0}
ROBOT_IP=${ROBOT_IP:-ROBOT_HOST}; C=${CONTAINER:-ai_worker}; DOM=${ROS_DOMAIN_ID:-30}; PORT=${PORT:-6123}; MAX_S=${MAX_S:-120}
REPO=$H2W_REPO_ROOT; X=$REPO/head2wrist/robot; OUT=${OUT_DIR:-$H2W_OUTPUT_ROOT/robot_day/$(date +%Y%m%d)}; mkdir -p $OUT; CT=/tmp/egoh2w
RENV="source /opt/ros/jazzy/setup.bash; source /root/ros2_ws/install/setup.bash 2>/dev/null; export RMW_IMPLEMENTATION=rmw_zenoh_cpp ROS_DOMAIN_ID=$DOM; export ZENOH_CONFIG_OVERRIDE='mode=\"client\";connect/endpoints=[\"tcp/$ROBOT_IP:7447\"]'; cd $CT"
TABLE_H=${TABLE_H:-0.72}
hw_lift() { $H2W_PYTHON -c "import json; s=json.load(open('$H2W_OUTPUT_ROOT/lyra2/task_inputs/${EP/\//_}/robot_setup.json'))['summary']['setup']; print(f'{$LIFT + $TABLE_H - s[\"table_h_m\"]:.4f}')"; }
cx() { docker exec ${TTY:-$([ -t 0 ] && echo -it)} $C bash -c "$RENV; $1"; }   # -it whenever there is a terminal: Ctrl+C must reach the bridge in the container
no_stale_bridge() { local P; P=$(docker exec $C pgrep -f "python3 egoh2w_bridge.py" | tr '\n' ' '); [ -z "${P// /}" ] || { echo "STOP: a bridge is still running in the container (PID $P). Stop it first:  docker exec $C kill $P"; exit 1; }; }
# both arms' init-preset joints (the keyboard GUI preset egoh2w_<task>_k<k>*.json): the bridge lowers the lift only when both arms are there
safe_args() { local PJ; PJ=$(ls ${H2W_AI_WORKER_ROOT:-$H2W_REPO_ROOT/third_party/ai_worker}/help_scripts/control/preset_joint_positions/egoh2w_${TASK}_k${K}*.json 2>/dev/null | head -1)
  [ -n "$PJ" ] || { echo "no init preset egoh2w_${TASK}_k${K}*.json in ai_worker/help_scripts/control/preset_joint_positions" >&2; return 1; }
  python3 -c "import json; P=json.load(open('$PJ'))['positions']; print('--safe_left', ','.join(f'{x:.4f}' for x in P['arm_l'][:7]), '--safe_right', ','.join(f'{x:.4f}' for x in P['arm_r'][:7]), *(['--head', ','.join(f'{x:.4f}' for x in P['head'][:2])] if 'head' in P else []))"; }
need_task() { [ -n "$TASK" ] || { echo "usage: robot_day.sh $CMD <task> [k]"; exit 1; }; B=$H2W_OUTPUT_ROOT/real/$TASK/batch.json; DS=$H2W_OUTPUT_ROOT/real/$TASK/lerobot_v5; [ -f $DS/egoh2w_episodes.json ] || { echo "no dataset for $TASK ($DS): run pipeline/real_batch.py first"; exit 1; }; }
ep_info() { $H2W_PYTHON - "$DS/egoh2w_episodes.json" "$K" <<'PY'
import json, sys
E = json.load(open(sys.argv[1])); k = int(sys.argv[2]); planned = [e for e in E['episodes'] if not any(e['base_offset_fwd_side_yaw'])]
e = planned[k]; print(e['side'], e['lift'], ','.join(f'{x:.4f}' for x in e['start_q']), e['ep'])
PY
}
image_topic() {   # the wrist topic that shows $SIDE's gripper, from today's camera check
  local side=$1 other; other=$([ $side = right ] && echo left || echo right)
  if [ -f $OUT/camcheck/wrist_mapping.json ] && grep -q '"swapped": true' $OUT/camcheck/wrist_mapping.json; then echo "/camera_$other/camera_$other/color/image_rect_raw/compressed"
  else echo "/camera_$side/camera_$side/color/image_rect_raw/compressed"; fi
}
copy_bridge() { docker exec $C mkdir -p $CT; docker cp $X/egoh2w_bridge.py $C:$CT/ >/dev/null; }

case $CMD in
net)
  timeout 5 bash -c "cat < /dev/null > /dev/tcp/$ROBOT_IP/7447" 2>/dev/null && echo "zenoh router $ROBOT_IP:7447: reachable" || { echo "no zenoh router at $ROBOT_IP:7447 (robot on? on the robot: zenohd + the bringup ffw_sg2_follower_ai.launch.py)"; exit 1; }
  copy_bridge; cx "timeout 20 ros2 topic list 2>/dev/null | grep -E 'joint_states|camera_(left|right)/camera_(left|right)/color/image_rect_raw/compressed|leader/joint_trajectory_command_broadcaster|joystick_controller_right' | sort"
  for s in left right; do echo "publishers on the $s arm command topic (should be 0 before execute):"; cx "timeout 15 ros2 topic info /leader/joint_trajectory_command_broadcaster_$s/joint_trajectory 2>/dev/null | grep -i 'publisher count'"; done ;;
camcheck)
  copy_bridge; echo "recording 15 s in 3 s: open and close ONE gripper fully (keyboard / leader), keep the other still"; sleep 3
  cx "timeout 60 python3 egoh2w_bridge.py --side right --mode record --record_s 15 --dump $CT/camcheck" && docker cp $C:$CT/camcheck $OUT/ >/dev/null && \
    $H2W_PYTHON $X/check_wrist_mapping.py $OUT/camcheck/record.npz | tail -4; echo "contact sheet: $OUT/camcheck/wrist_mapping.jpg" ;;
setup)
  need_task; read SIDE LIFT Q0 EP <<< "$(ep_info)"; N=${EP/\//_}
  $H2W_PYTHON -c "
import json; s=json.load(open('$H2W_OUTPUT_ROOT/lyra2/task_inputs/$N/robot_setup.json'))['summary']; u=s['setup']
print(f'demo $EP ($SIDE arm): base {100*u[\"base_forward_m\"]:+.0f} cm forward, {100*u[\"base_right_m\"]:+.0f} cm right of the point on the table under the demonstrator\'s head, heading {u[\"heading_deg\"]:+.1f} deg (+ = turned right); lift {u[\"lift_m\"]:.3f} m')
print(f'  placement tolerance {s[\"search\"].get(\"placement_tolerance\")} (fraction of +-5 cm / +-10 deg placement errors the path survives); start joints $Q0')"
  echo "  top / side view: $H2W_OUTPUT_ROOT/lyra2/task_inputs/$N/robot_setup.png" ;;
place)     # EXPERIMENTAL placement aid: the expected head-camera view at the planned stance over the live one (+ a base-error estimate
           # where the scene has texture). Its geometry is not validated (sim: ~5-10 deg off, cause not found): cross-check with the tape measure
  need_task; read SIDE LIFT Q0 EP <<< "$(ep_info)"; no_stale_bridge; copy_bridge; D=$CT/place_$(date +%H%M%S)
  cx "timeout 30 python3 egoh2w_bridge.py --side $SIDE --mode snap --dump $D" && docker cp $C:$D $OUT/ >/dev/null && \
    $H2W_PYTHON $X/placement_view.py --ep $EP --snap $OUT/$(basename $D)/snap.npz | tail -2 && \
    mkdir -p ${H2W_AI_WORKER_ROOT:-$H2W_REPO_ROOT/third_party/ai_worker}/help_scripts/control/ref_views && cp $OUT/$(basename $D)/expected_head.png ${H2W_AI_WORKER_ROOT:-$H2W_REPO_ROOT/third_party/ai_worker}/help_scripts/control/ref_views/expected_head.png && \
    echo "  live overlay: cd ~/projects/ai_worker/help_scripts/control && bash run_keyboard_control.sh --cams --ref_head ref_views/expected_head.png (keep the head joints and lift as they are now)" ;;
replay)    # the PLANNED joint path itself (no policy): lift, ramp, then each planned pose; the real wrist frame at every pose ->
           # generated-vs-real view metrics (compare_views.py) and the plan's open-loop outcome (trial log)
  need_task; read SIDE LIFT Q0 EP <<< "$(ep_info)"; N=${EP/\//_}; IMG=$(image_topic $SIDE); PF=$OUT/plan_$N.npz
  $H2W_PYTHON -c "
import numpy as np; RP = np.load('$H2W_OUTPUT_ROOT/lyra2/task_inputs/$N/robot_plan.npz'); C = np.load('$H2W_OUTPUT_ROOT/composite/${N}_v5/clip.npz')
assert len(RP['arm_q']) == len(C['gripper_joint']) and (RP['frames'] == C['plan_frame']).all(), 'plan and clip disagree'
np.savez('$PF', q=np.c_[RP['arm_q'], C['gripper_joint']], frames=RP['frames']); print('plan', RP['arm_q'].shape[0], 'poses, lift', float(RP['lift'][0]))" || exit 1
  no_stale_bridge; SA=$(safe_args) || exit 1; copy_bridge; docker cp $PF $C:$CT/ >/dev/null; D=$CT/replay_${N}_$(date +%H%M%S)
  LIFT_HW=$(hw_lift); echo "table $TABLE_H m: physical lift $LIFT_HW (plan $LIFT)"; echo "REPLAY: the lift goes to $LIFT_HW, the arm ramps to the plan's first pose and follows the planned path ($SIDE arm, <= 0.03 rad / tick; the gripper closes as planned). Object placed as in demo $EP. E-stop in hand. Type go:"; read ans; [ "$ans" = go ] || exit 1
  TTY=$([ -t 0 ] && echo -it) cx "python3 egoh2w_bridge.py --side $SIDE --mode replay_plan --plan $CT/$(basename $PF) --image_topic $IMG --set_lift $LIFT_HW --grip_close ${GRIP_CLOSE:-0.9} --grasp_dwell ${GRASP_DWELL:-3.0} --dump $D ${BRIDGE_ARGS:-} $SA"
  docker cp $C:$D $OUT/ >/dev/null 2>&1 && echo "record -> $OUT/$(basename $D)" || { echo "no record"; exit 1; }
  $H2W_PYTHON $X/compare_views.py --ep $EP --record $OUT/$(basename $D)/plan_record.npz | tail -2
  python3 -c "import json; r=json.load(open('$H2W_OUTPUT_ROOT/sim_policy/stack_48/view_metrics_reference.json'))['sim_reference']['all']; print('sim reference (true vs generated, where policies transferred):', {k: round(v, 3) for k, v in r.items()})"
  [ -t 0 ] && $H2W_PYTHON $X/trial_log.py add --task $TASK --k $K --mode replay --dump $OUT/$(basename $D) ;;
server)
  need_task; read SIDE LIFT Q0 EP <<< "$(ep_info)"; CK=${CKPT:-$H2W_OUTPUT_ROOT/sim_policy/train/real_${TASK}_floor/checkpoints/080000/pretrained_model}   # CKPT: downloaded weights or another checkpoint
  [ -d $CK ] || { echo "no checkpoint $CK (training still running?)"; exit 1; }
  # safety floor: the target TCP never goes below the lowest TCP of the training set (minus 4 mm); Z_FLOOR overrides
  ZF=${Z_FLOOR:-$($H2W_PYTHON -c "import glob, pandas as pd, numpy as np; R='$H2W_OUTPUT_ROOT/real/$TASK/${AUG_DS:-lerobot_aug_floor}/data'; z=min(np.stack(pd.read_parquet(f)['observation.state'].values)[:, 2].min() for f in glob.glob(R + '/*/*.parquet')); print(f'{z - 0.004:.4f}')")}
  # and the target TCP stays inside the training set's xy range (+ 1 cm); XY_BOX overrides
  XY=${XY_BOX:-$($H2W_PYTHON -c "import glob, pandas as pd, numpy as np; R='$H2W_OUTPUT_ROOT/real/$TASK/${AUG_DS:-lerobot_aug_floor}/data'; S=np.concatenate([np.stack(pd.read_parquet(f)['observation.state'].values)[:, :2] for f in glob.glob(R + '/*/*.parquet')]); print(f'{S[:,0].min()-0.01:.3f},{S[:,0].max()+0.01:.3f},{S[:,1].min()-0.01:.3f},{S[:,1].max()+0.01:.3f}')")}
  echo "safety floor: target TCP z >= $ZF; xy box $XY (model frame)"
  nohup $H2W_LEROBOT_PYTHON $REPO/head2wrist/policy/policy_server.py --ckpt $CK --port $PORT --repr eef_hybrid --lift $LIFT --side $SIDE --z_floor $ZF --xy_box $XY > $OUT/server_$TASK.log 2>&1 &
  echo $! > $OUT/server.pid; for i in $(seq 60); do grep -q "listening" $OUT/server_$TASK.log && break; sleep 2; done; tail -1 $OUT/server_$TASK.log; echo "server pid $(cat $OUT/server.pid), log $OUT/server_$TASK.log" ;;
shadow|execute)
  need_task; read SIDE LIFT Q0 EP <<< "$(ep_info)"; IMG=$(image_topic $SIDE)
  # the gripper value the policy sees is capped at the training set's most-closed gripper (GRIP_STATE_MAX overrides)
  GSM=${GRIP_STATE_MAX:-$($H2W_PYTHON -c "import glob, pandas as pd, numpy as np; from head2wrist.policy.repr_lib import joint_of; R='$H2W_OUTPUT_ROOT/real/$TASK/${AUG_DS:-lerobot_aug_floor}/data'; w=min(np.stack(pd.read_parquet(f)['observation.state'].values)[:, 9].min() for f in glob.glob(R + '/*/*.parquet')); print(f'{float(joint_of(w)):.3f}')")}
  echo "gripper seen by the policy: capped at $GSM (the training set's most-closed)"
  [ -f $OUT/camcheck/wrist_mapping.json ] || echo "WARNING: no camera check today (robot_day.sh camcheck): assuming the topics are not swapped"
  no_stale_bridge; echo "demo $EP: $SIDE arm, lift $LIFT, image $IMG"; copy_bridge; D=$CT/${CMD}_${TASK}_${K}_$(date +%H%M%S)
  if [ $CMD = shadow ]; then cx "python3 egoh2w_bridge.py --side $SIDE --mode shadow --policy localhost:$PORT --image_topic $IMG --grip_state_max $GSM --max_s 30 --dump $D ${BRIDGE_ARGS:-}"
  else
    ls $OUT/ | grep -q "^shadow_" || { echo "run 'robot_day.sh shadow $TASK $K' first"; exit 1; }
    SA=$(safe_args) || exit 1
    LIFT_HW=$(hw_lift); echo "table $TABLE_H m: physical lift $LIFT_HW (plan $LIFT; the server keeps $LIFT)"
    N=${EP/\//_}; CTRL=${H2W_AI_WORKER_ROOT:-$H2W_REPO_ROOT/third_party/ai_worker}/help_scripts/control; mkdir -p $CTRL/ref_views
    [ -f $CTRL/ref_views/${N}_start.png ] || $H2W_PYTHON -c "import cv2; c = cv2.VideoCapture('$H2W_OUTPUT_ROOT/composite/${N}_v5/wrist_upright.mp4'); ok, f = c.read(); cv2.imwrite('$CTRL/ref_views/${N}_start.png', f)"
    GOTO="python3 egoh2w_bridge.py --side $SIDE --mode goto --image_topic $IMG $SA"
    # 1/4 start pose: lift, then the arm (no policy)
    echo "1/4 PRESET egoh2w_${TASK}_k${K}: FIRST both arms (and the head) move to the preset with the lift where it is, THEN the lift"
    echo "    goes to $LIFT_HW (only once both arms are there). Arms <= 0.06 rad / tick, lift 0.06 m/s. E-stop in hand; Ctrl+C holds the robot."
    echo -n "    Press g: "; read -n 1 -r ans; echo; [ "$ans" = g ] || exit 1
    cx "python3 egoh2w_bridge.py --side $SIDE --mode preset --image_topic $IMG $SA --set_lift $LIFT_HW --max_step 0.06 --lift_speed 0.06" || { echo "preset not reached: check the robot"; exit 1; }
    # 2/4 base alignment: the controller with the generated start view over the live right-wrist camera
    if [ "${ALIGN:-1}" = 1 ]; then                              # base alignment: live right-wrist view over the generated start view (ALIGN=0 skips)
      echo "2/4 BASE: in the window's Base tab drive with the arrows / A / D until the live right-wrist view matches the overlay, then CLOSE THE WINDOW"
      bash $CTRL/run_keyboard_control.sh --cams --ref_right ref_views/${N}_start.png || echo "  (the controller exited with an error)"
    else echo "2/4 base alignment skipped (ALIGN=0): place the cube"; fi
    # 3/4 the policy
    echo -n "3/4 POLICY: press g to run it for $MAX_S s (Ctrl+C stops it): "; read -n 1 -r ans; echo; [ "$ans" = g ] || exit 1
    trap 'echo " (stopped)"' INT
    TTY=$([ -t 0 ] && echo -it) cx "python3 egoh2w_bridge.py --side $SIDE --mode execute --policy localhost:$PORT --image_topic $IMG --set_lift $LIFT_HW --grip_close ${GRIP_CLOSE:-0.9} --grasp_dwell ${GRASP_DWELL:-3.0} --grip_state_max $GSM --start_q $Q0 --max_s $MAX_S --step_gate $SA --dump $D ${BRIDGE_ARGS:-}"
    trap - INT
    # 4/4 back to the start pose: lift up 5 cm first so the gripper clears the table, arm back, lift down
    SKIP=0; trap 'SKIP=1' INT; echo "4/4 RETURN in 3 s: lift +5 cm, arm to the start pose, lift back (Ctrl+C to skip)"; sleep 3; trap - INT
    if [ $SKIP = 0 ]; then
      UP=$(python3 -c "print(f'{min(0.0, $LIFT_HW + 0.05):.4f}')")
      # the return runs at twice the normal limits (no policy in the loop): 0.06 rad / tick (~1 rad/s), lift 0.06 m/s
      cx "$GOTO --max_step 0.06 --lift_speed 0.06 --ramp_max 1.6 --set_lift $UP --start_q $Q0" && cx "$GOTO --lift_speed 0.06 --set_lift $LIFT_HW" && echo "back at the start pose" || echo "return incomplete: check the robot"
    else echo "return skipped"; fi
  fi
  docker cp $C:$D $OUT/ >/dev/null 2>&1 && echo "dump -> $OUT/$(basename $D)"
  [ $CMD = execute ] && [ -t 0 ] && $H2W_PYTHON $X/trial_log.py add --task $TASK --k $K --mode execute --dump $OUT/$(basename $D) ;;
trials)
  $H2W_PYTHON $X/trial_log.py summary ;;
stop)
  docker exec $C bash -c "pkill -INT -f '[e]goh2w_bridge.py'; true"; [ -f $OUT/server.pid ] && kill $(cat $OUT/server.pid) 2>/dev/null; echo stopped ;;
*) sed -n 2,22p $0 ;;
esac
