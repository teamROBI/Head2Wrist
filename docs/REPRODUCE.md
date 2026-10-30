# Reproducing the workshop experiments

Start from the installed environments in [INSTALL](INSTALL.md) and the data/assets in [DATA](DATA.md). Export the root variables and source `scripts/env.sh`. Each stage saves outputs under `H2W_OUTPUT_ROOT`. Run long jobs detached with logs in that directory. A cached stage is skipped; use `--redo` or `--from` to regenerate it. `--stages` runs an explicit subset for inspecting cached intermediates.

| Paper result | Commands / controls |
|---|---|
| Fig. 1–2, one demonstration through wrist generation | `bash scripts/generate.sh --ep stack/48`; Quest: add `--root "$H2W_DATA_ROOT/data/lab" --ep cube_bowl/0` |
| Physics acceptance and placement-tolerant stance | `python -m head2wrist.robot.setup_pose --help`; `python -m head2wrist.simulation.sim_gate --eps stack/48`; gate variants A–D re-plan at fixed stance |
| Real 18/20 vs 5/20 | [Floor 80K ACT](https://huggingface.co/jkim50104/Head2Wrist-policy-cube-bowl): three accepted Quest demos; 84 augmented episodes per demo with native mesh size (`--obj_size` unset); merge into `lerobot_aug_floor`; EEF-hybrid ACT with strong augmentation for 80,000 steps; `robot_day.sh execute` vs `replay` on the same 20 physical starts (recipe below) |
| Main simulation comparisons, Fig. 3 | `policy.expert_plan --base_pert`, `scripts/run_collect_named.sh`, `policy.gen_scene`, `policy.gen_episode`, `policy.to_lerobot_repr`, `scripts/train_repr.sh`, `scripts/eval_policy.sh`, `scripts/rescore.sh` (sequence below) |
| Table 1, representation / stance / augmentation | Change `REPR=joint_abs,eef_abs,eef_hybrid`; omit `--base_pert` for fixed stance; set `AUG=` for ordinary augmentation, `AUG=strong` for the full recipe |
| Table 1, generator bundles | `policy.gen_episode --body 0 --shadow 0` and `--gripper_tex` / `--renderer` select layers and textures; base/colour/seamless/full are bundled historical renderer revisions, not single independent switches; exact earlier versions require their original cached renders |
| Table 1, object error | `policy.gen_episode --obj_noise_pos`, `--obj_noise_yaw`, `--obj_noise_scale`; inspect `--help` for units and seed; same true test camera |
| Table 1, constant image | `BLANK=1` at training; `--blank_image` in policy server during evaluation; use the early-stop schedule described in the paper |
| Negative Transfer2.5 pilot | Optional `baselines.blender_proxy`, `baselines.run_transfer` and `policy.gen_episode --renderer transfer`; upstream model env required; train/evaluate on the same 20 demonstrations |
| Replay image comparison | `python -m head2wrist.robot.compare_views --help` (generated clip vs real `plan_record.npz`) |

## Real cube-bowl floor recipe

The author-confirmed paper model is [floor 80K](https://huggingface.co/jkim50104/Head2Wrist-policy-cube-bowl), trained on [lerobot_aug_floor (252 episodes, 36,652 frames)](https://huggingface.co/datasets/jkim50104/Head2Wrist-cube-bowl-252ep). The reported 18/20 outcome is an author aggregate; individual trial logs were not saved. Fresh training and reconstruction can vary; policy parity checks with the released weights test code equivalence, not a new physical evaluation.

Use the three released raw Quest sessions `20261017_010420`, `20261017_010444`, and `20261017_010507` as `cube_bowl/0`, `/1`, and `/2`. The fourth capture failed the physics gate. Keep the task's fixed lift at −0.25 m. All generated views use the fitted table-front edge and table/head-mesh clipping, the floor plane 0.73 m below the tabletop, measured backdrop colour, and the corrected Quest head-camera focal length propagated from the source metadata. The grasped mesh keeps its native reconstructed size; leave `--obj_size` unset.

The batch command performs conversion, all generation stages through `compose_v5`, augmentation, export and training. Its defaults are 252 augmented episodes across the accepted demos (84 each here), seeds `1000 + demo index`, `lerobot_aug_floor`, run `real_cube_bowl_floor`, and 80,000 training steps. Supply the calibration JSON distributed with the raw recordings:

```bash
RAW="$H2W_DATA_ROOT/data/quest_raw"
python -m head2wrist.pipeline.real_batch --task cube_bowl \
  --sessions "$RAW/20261017_010420,$RAW/20261017_010444,$RAW/20261017_010507" \
  --calib "$RAW/calib_quest3.json" --description 'Put the cube into the bowl' \
  --side right --lift -0.25 --jobs 1 --aug_workers 2
```

For converted, fully generated demos, run the remaining stages with `--steps augment,dataset,train` and the same arguments. Alternatively, the explicit augmentation/export/training commands in [README](../README.md#usage) produce the same floor dataset layout. `bash scripts/train_repr.sh` with no arguments selects `$H2W_OUTPUT_ROOT/real/cube_bowl/lerobot_aug_floor` and run `real_cube_bowl_floor`; explicit dataset/run arguments support other experiments.

Training uses EEF-hybrid state/actions (10 dimensions), 424×240 RGB at 16 Hz, ACT ResNet-18, chunk size 32, one action per control step, batch size 32, seed 1000, AdamW learning rate `1e-5`, weight decay `1e-4`, eight data workers, and saves every 20,000 steps. Strong image augmentation applies affine ±3° / ±4% translation / 0.95–1.05 scale with probability 0.8, colour jitter with probability 0.8, blur with probability 0.4, and Gaussian noise σ=0.02. The separate LeRobot built-in image transforms remain disabled. The training wrapper defaults to `REPR=eef_hybrid AUG=strong`; set `AUG=` explicitly for the ordinary-augmentation ablation.

The final weights are written to `$H2W_OUTPUT_ROOT/sim_policy/train/real_cube_bowl_floor/checkpoints/080000/pretrained_model/`. Follow [ROBOT](ROBOT.md) for offline checks and supervised deployment of [floor 80K](https://huggingface.co/jkim50104/Head2Wrist-policy-cube-bowl).

## Main simulation sequence

Development scenes: `stack/18`, `stack/48`, `basic_pick_place/25`, `basic_pick_place/1`. Held out: `basic_pick_place/73`, `basic_pick_place/67`. The phone (`basic_pick_place/1`) is the thin-object failure case. Repeat the following per scene, after its generator has passed the gate and exported its twin:

```bash
EP=stack/48; N=stack_48; D="$H2W_OUTPUT_ROOT/sim_policy/$N"
python -m head2wrist.policy.expert_plan --ep "$EP" --n 450 --seed 11 --base_pert --out_name plans_sr
python -m head2wrist.policy.expert_plan --ep "$EP" --n 30 --seed 99 --out_name plans_eval
python -m head2wrist.policy.expert_plan --ep "$EP" --n 20 --seed 777 --range 0.12 --min_range 0.08 --out_name plans_eval_ood
python -m head2wrist.policy.make_pert_plans --scene "$N" --n 24
PLANS=plans_sr OUTDIR=demos_sr bash scripts/run_collect_named.sh "$N" --head_pitch 0.69
PLANS=plans_sr OUTDIR=demos_sr bash scripts/run_collect_named.sh "$N" --clean_hr "/policy/$N/gen/clean_head_1920.png"
python -m head2wrist.policy.gen_scene --scene "$EP"
python -m head2wrist.policy.gen_episode --scene "$EP" --demos "$D/demos_sr" --out "$D/gen/wrist_sr"
for VIEW in true gen; do
  EXTRA=(); [ "$VIEW" = gen ] && EXTRA=(--wrist_from "$D/gen/wrist_sr")
  "$H2W_LEROBOT_PYTHON" -m head2wrist.policy.to_lerobot_repr --demos "$D/demos_sr" \
    "${EXTRA[@]}" --root "$D/lerobot_eef_$VIEW" --repo_id "head2wrist/${N}_$VIEW" --state eef --scene "$N"
  AUG=strong REPR=eef_hybrid bash scripts/train_repr.sh "$D/lerobot_eef_$VIEW" "hyb_${N}_$VIEW"
  SCENE=$N TAG=_in EVAL_PLANS=plans_eval bash scripts/eval_policy.sh "hyb_${N}_$VIEW" 30
  SCENE=$N TAG=_ood EVAL_PLANS=plans_eval_ood bash scripts/eval_policy.sh "hyb_${N}_$VIEW" 20
  SCENE=$N TAG=_pert EVAL_PLANS=plans_eval_pert bash scripts/eval_policy.sh "hyb_${N}_$VIEW" 24
 done
bash scripts/rescore.sh --scene "$N" --runs "true:hyb_${N}_true,gen:hyb_${N}_gen"
```

The paper used 225–630 accepted expert demonstrations per scene. The 450 candidate count is a starting recipe; filter by physics success and preserve paired true/generated trajectory identities. `gen_episode` always takes the grasped object's geometry and initial pose from the twin in this simulation protocol; do not present it as reconstruction from raw EgoDex alone. Training randomizes the robot forward position −5 to +2 cm, sideways ±5 cm, yaw ±10°, and object starts ±8 cm / ±30°.

For stack/18 use 50 in-range, pooled 16+24 out-of-range, and 24 moved-base starts, with the recorded plan seeds and split names. `rescore_evals` accepts explicit `--sets` to score each saved split; pool paired outcomes by plan identity for the main result. A second training seed was used on stack/48 and basic_pick_place/25. Exact historical raw starts, renderer bundles and early-stop schedules are not reconstructed by this generic recipe; distribute the paired plan manifests/cached historical datasets with the data release to reproduce every table cell exactly.

## Strict scoring and statistics

`policy.rescore_evals` reads each saved rollout and requires all three conditions: object lifted ≥3 cm, gripper released, final object within 3 cm XY and 2 cm height of the expert's resting placement. It reports 95% Wilson intervals and an exact two-sided McNemar test on paired starts. A success above the target while still holding the object fails strict scoring. Expert resting height is estimated from accepted demonstration endpoints. Keep plan IDs and dataset pairing fixed; counts from unpaired episodes cannot be used for the reported paired test.

The main comparison is sensitive to reconstruction, grasp and physics versions. Record seeds, root configuration, image/asset versions and cached-stage use with each run. The paper's no-difference results do not establish equivalence, and real trials cover one task near demonstrated starts.
