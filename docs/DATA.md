# Data

Follow the [official EgoDex download instructions](https://github.com/apple/ml-egodex) and its data terms. The input root contains task directories with paired `episode.hdf5` and `episode.mp4` files. EgoDex poses use metric ARKit world coordinates; each recording supplies camera transforms, intrinsics and hand joints. Keep the task/episode names used by the paper.

```text
data/
  egodex/test/stack/48.hdf5, 48.mp4
  egodex/test/basic_pick_place/25.hdf5, 25.mp4
  data/lab/cube_bowl/0.hdf5, 0.mp4
  data/quest_raw/<session>/poses.txt, internal.mp4
  assets/gripper_gap_table.npz
  assets/gripper_texture.npz
  assets/ffw_sg2/body_meshes_base_frame.npz
  checkpoints/da3/DA3NESTED-GIANT-LARGE-1.1/
```

`H2W_DATA_ROOT` defaults to `data/`; the nested `data/lab` and output folders preserve the artifact layout expected by the converters. The numeric TCP-to-D405 transform and finger samples are packaged in `head2wrist/assets/`. Download the larger gripper texture, joint-to-jaw-width table and body samples from the [asset bundle](https://huggingface.co/datasets/jkim50104/Head2Wrist-cube-bowl-252ep). The generated dataset and its associated calibration assets are hosted in the dataset repository; follow its file layout when downloading.

Quest recordings: [demonstrations](https://huggingface.co/datasets/jkim50104/Head2Wrist-quest-demos). Paper real model: [ACT floor 80K](https://huggingface.co/jkim50104/Head2Wrist-policy-cube-bowl), trained on [the 252-episode floor dataset](https://huggingface.co/datasets/jkim50104/Head2Wrist-cube-bowl-252ep). Place its configuration and weights in `$H2W_DATA_ROOT/checkpoints/floor/080000/pretrained_model/` for the deployment examples.

## Quest 3 / EgoKit

Collect posed head video and OpenXR hand joints with EgoKit. `quest_ingest` copies recordings with adb. Estimate camera intrinsics, the camera-in-head transform and timing with the self-calibration tool, then convert:

```bash
python -m head2wrist.capture.quest_ingest --list
python -m head2wrist.capture.calib_headcam --help
python -m head2wrist.capture.egokit_to_egodex --session "$H2W_DATA_ROOT/data/quest_raw/session" \
  --calib "$H2W_DATA_ROOT/calib.json" --out "$H2W_DATA_ROOT/data/lab" \
  --task cube_bowl --idx 0 --side right --description 'Put the cube into the bowl'
python -m head2wrist.capture.test_egokit_roundtrip --root "$H2W_EGODEX_ROOT" \
  --ep basic_pick_place/1 --work "$H2W_OUTPUT_ROOT/roundtrip"
```

Calibration JSON contains `fx`, `fy`, `cx`, `cy`, `width`, `height`, `T_head_cam` (4×4) and `dt_ms`. The converter handles Unity handedness, camera/hand transforms, fixed EgoDex image framing and stream alignment. `--start_s` and `--end_s` trim a demo window. All exported videos are H.264/yuv420p.

## Generated artifacts

`$H2W_OUTPUT_ROOT/lyra2/explore_inputs/<task_episode>` stores pre-grasp reconstruction and approach/transport plans; `lyra2/task_inputs/<task_episode>` stores the full plan, stance and proxy; `sim_twin/` stores physics checks; `composite/<task_episode>_v5/clip.npz` stores raw BGR wrist frames and object poses. The `lyra2` artifact directory name is retained for compatibility; the default generator uses the layered renderer.

For real training, `pipeline.real_batch` converts sessions, runs the generator, augments accepted demos, and merges them into `real/<task>/lerobot_aug_floor`. The defaults allocate 252 episodes across the accepted demos; the paper uses `cube_bowl/0`, `/1`, and `/2`, with 84 episodes each and seeds 1000/1001/1002. Demo `/3` failed the physics gate and is excluded. Direct `pipeline.real_augment` calls default to 84 episodes and seed `1000 + demo index`.

The floor recipe includes the fitted table-front edge, table/head-mesh clipping, a floor plane 0.73 m below the tabletop, measured backdrop colour, and corrected Quest focal propagation. It preserves the native estimated object mesh: leave `--obj_size` unset. `policy.gen_episode` retains optional size and pose perturbations for ablations. See [REPRODUCE](REPRODUCE.md) for generation, merging and training commands.

The later size5 variant (`--obj_size 0.05`) is not released.
