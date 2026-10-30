# FFW-SG2 deployment

The bridge runs in the ROBOTIS ai_worker **ROS 2 Jazzy** environment. The active wrist camera is an Intel RealSense D405. The policy consumes raw-orientation 424×240 RGB; upright visualization clips rotate the sensor view for display only. Check camera-to-arm mapping before executing a policy.

Supply `ROBOT_IP=ROBOT_HOST` with your robot's configured hostname, `CONTAINER=ai_worker`, `ROS_DOMAIN_ID` (deployment default 30), and `H2W_AI_WORKER_ROOT` for the upstream control GUI and preset files. `PORT` selects the local policy server, `MAX_S` the session duration, and `OUT_DIR` the recording directory. Network hosts/topics can also be given directly to the bridge via `--policy`, `--state_topic`, `--image_topic`, `--cmd_topic`, `--lift_topic` and preset topic options. No lab network settings are included.

Use the author-confirmed paper model [floor 80K ACT](https://huggingface.co/jkim50104/Head2Wrist-policy-cube-bowl), trained on [lerobot_aug_floor](https://huggingface.co/datasets/jkim50104/Head2Wrist-cube-bowl-252ep). Place the downloaded model files in `$H2W_DATA_ROOT/checkpoints/floor/080000/pretrained_model/` and the floor dataset in `$H2W_OUTPUT_ROOT/real/cube_bowl/lerobot_aug_floor/`. Generate `lerobot_v5/egoh2w_episodes.json` with the batch dataset stage for stance/start-pose metadata. `AUG_DS` defaults to `lerobot_aug_floor` for the training-derived safety limits and gripper-state cap. The model's 18/20 result is author-reported; per-trial logs were not saved.

```bash
export ROBOT_IP=ROBOT_HOST
export H2W_AI_WORKER_ROOT="$PWD/third_party/ai_worker"
export CKPT="$H2W_DATA_ROOT/checkpoints/floor/080000/pretrained_model"
bash scripts/robot_day.sh net
bash scripts/robot_day.sh camcheck
bash scripts/robot_day.sh setup cube_bowl 0
bash scripts/robot_day.sh server cube_bowl
bash scripts/robot_day.sh shadow cube_bowl 0
bash scripts/robot_day.sh execute cube_bowl 0
```

`execute` requires explicit terminal confirmations. The stored preset must provide both arms and, when available, the head. Prepare a matching `egoh2w_<task>_k<k>*.json` in the upstream controller's `preset_joint_positions/`. Keep the E-stop accessible. The robot base is placed and aligned once; the policy then moves the arm through IK.

## Preserved safety behavior

- Preset mode moves **both arms and the head first**, with lift unchanged, then moves the lift only when both arms reach the preset. It refuses large arm motion when the lift is too low (`--arms_lift_min`).
- Downward lift commands require both safe arm presets and arm agreement within `--pose_tol`. Start-pose ramps beyond `--ramp_max` are refused.
- Ctrl+C / SIGTERM publishes hold positions for both arms and lift while the ROS context remains active. The runner uses a terminal for container commands when available and refuses to launch over a stale bridge process.
- Policy targets obey the training-derived TCP `z_floor` and `xy_box`. These limits are applied in the server before IK.
- `--step_gate` queries the policy once the previous pose is reached (or the gate times out); `--jump_max` rejects abrupt targets and holds. Per-tick arm/gripper limits, joint limits, stale-frame handling and maximum duration remain enabled.
- Table-height compensation changes physical lift only; policy FK/IK keeps the planned lift. Verify `TABLE_H` for your table.

`GRIP_STATE_MAX`, `GRIP_CLOSE`, `GRASP_DWELL`, `Z_FLOOR`, `XY_BOX`, `AUG_DS`, `ALIGN` and `BRIDGE_ARGS` expose the original operational overrides. Defaults are documented in `scripts/robot_day.sh` and bridge `--help`. The `stop` command controls the selected deployment container and session; do not run it against another session.

Without `CKPT`, the server selects the locally reproduced [floor 80K](https://huggingface.co/jkim50104/Head2Wrist-policy-cube-bowl) directory `$H2W_OUTPUT_ROOT/sim_policy/train/real_cube_bowl_floor/checkpoints/080000/pretrained_model/`. Training uses the floor renderer fixes and native reconstructed object size (`--obj_size` unset); see [REPRODUCE](REPRODUCE.md).

## Offline and fake-follower checks

The offline check never opens a robot connection:

```bash
"$H2W_LEROBOT_PYTHON" -m head2wrist.robot.offline_real_check \
  --ckpt "$H2W_DATA_ROOT/checkpoints/floor/080000/pretrained_model" \
  --record "$H2W_OUTPUT_ROOT/replay/plan_record.npz" --lift -0.25 --gsm 0.559 \
  --device cpu --out "$H2W_OUTPUT_ROOT/predictions.npz"
python -m head2wrist.robot.compare_views --help
```

Generate a feed with `python -m head2wrist.robot.make_fake_feed --help`, then run `bash scripts/test_fake_bridge.sh <feed.npz> <output-dir>`. The smoke test uses a new container with **network disabled**, ROS domain **77**, localhost-only discovery and a `/h2w_release_test` namespace. It records frames, exercises preset ordering, refuses an unsafe lift move, and checks hold on interrupt. It never enters an existing ai_worker container. `H2W_ROS_IMAGE` defaults to `robotis/ai-worker:2.0.2`; the feed should start with both arms at zero and lift at zero for this fixture's expected motion.
