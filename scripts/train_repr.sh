#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
if [ "$#" -eq 0 ]; then
  DS="$H2W_OUTPUT_ROOT/real/cube_bowl/lerobot_aug_floor"
  RUN=real_cube_bowl_floor
else
  DS=$1; RUN=${2:?usage: train_repr.sh [dataset-dir run-name [args...]]}; shift 2
fi
REPR=${REPR:-eef_hybrid} AUG=${AUG-strong} "$H2W_LEROBOT_PYTHON" -m head2wrist.policy.train_repr \
  --dataset.repo_id=head2wrist/"$(basename "$DS")" --dataset.root="$DS" \
  --dataset.image_transforms.enable=false --seed=1000 \
  --policy.type=act --policy.device=cuda --policy.chunk_size=32 --policy.n_action_steps=1 \
  --policy.push_to_hub=false --wandb.enable=false --output_dir="$H2W_OUTPUT_ROOT/sim_policy/train/$RUN" \
  --job_name="$RUN" --steps="${STEPS:-80000}" --batch_size=32 --save_freq=20000 --log_freq=500 --num_workers=8 "$@"
