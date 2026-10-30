# External dependencies and optional pilots

Only setup recipes are tracked here. Downloaded source, environments, weights and robot assets remain untracked and retain their upstream licenses. Accept model/data terms before downloading them. This project's MIT license does not relicense third-party materials.

| Component | Source / recorded pin | License / use |
|---|---|---|
| Depth Anything 3 | [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3), installed package 0.1.1 | [Apache-2.0 code](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/LICENSE); checkpoint terms apply separately; main metric-depth stage. No source commit was recorded by the installed wheel. |
| SAM 3 | [facebookresearch/sam3](https://github.com/facebookresearch/sam3), `facebook/sam3` model through Transformers 5.17.0 | SAM License and checkpoint terms; segmentation. |
| SAM 3D Objects | [facebookresearch/sam-3d-objects](https://github.com/facebookresearch/sam-3d-objects), `a0016abefdb519f98242d90a35c18dc04f9f6af2` | SAM License; separate upstream environment and weights. |
| LeRobot | [huggingface/lerobot](https://github.com/huggingface/lerobot), PyPI `0.3.3` | Apache-2.0; ACT training and checkpoint loading. |
| ROBOTIS ai_worker | [ROBOTIS-GIT/ai_worker](https://github.com/ROBOTIS-GIT/ai_worker), `6961f7e7368329d57ebf2adf0fdba08680a43949` | Apache-2.0; URDF, meshes, control presets and ROS environment. Check asset terms separately. |
| robotis_lab adapter | [ROBOTIS-GIT/cyclo_lab](https://github.com/ROBOTIS-GIT/cyclo_lab), `9dd83c5988ea1709b7f232e8922c5c55dcd68218` | Public Apache-2.0 code and FFW_SG2 USD assets; the former robotis_lab URL redirects here. Install with `setup_robotis.sh`. |
| Isaac Lab | [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab), v2.3 environment, checkout `3c6e67bb5c7ada942a6d1884ab69338f57596f77` | BSD-3-Clause code; Isaac Sim 5.1 has separate NVIDIA terms. |
| Open3D / Pinocchio / trimesh | pinned versions in `pyproject.toml` | MIT / BSD-2-Clause / MIT respectively; geometry and rendering. |
| LaMa wrapper | [`simple-lama-inpainting==0.1.2`](https://github.com/enesmsahin/simple-lama-inpainting/blob/main/LICENSE) | Apache-2.0 wrapper; upstream model/checkpoint terms also apply. |
| Blender Python | `bpy==5.2.2`, Python 3.13 | GPL-3.0-or-later; optional Blender preparation for the negative pilot. |
| Lyra 2.0 | [nv-tlabs/lyra](https://github.com/nv-tlabs/lyra), **`9fffc9a`** | Apache-2.0 code; checkpoint license separately. Optional setup only; pilot implementation omitted. |
| GEN3C | [nv-tlabs/GEN3C](https://github.com/nv-tlabs/GEN3C), **`db2ffe1`** | Apache-2.0 code; checkpoint license separately. Optional setup only; pilot implementation omitted. |
| Cosmos-Transfer2.5 | [nvidia-cosmos/cosmos-transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5), **`2ff49d0`** | Apache-2.0 code; NVIDIA model and guardrail terms separately. Optional adapter in `baselines/`. |
| Cosmos 3 | `nvidia/Cosmos3-Nano`, vLLM/vLLM-Omni `0.30.0` environment | Upstream model and runtime licenses; setup only, pilot implementation omitted. |
| Wan VACE | `Wan-AI/Wan2.1-VACE-14B-diffusers` | Upstream Wan model terms and Diffusers Apache-2.0 code; setup only, pilot implementation omitted. |
| EgoDex / EgoKit | official dataset and capture project | Their own dataset/software terms; not redistributed by this repository. |

The three recorded video-model git pins are checked out by the matching setup scripts. Optional environment installers can download substantial dependencies and checkpoints. `setup_geometry.sh` prepares DA3 and pinned robot/SAM 3D checkouts, while `setup_robotis.sh` downloads the pinned public robot descriptions/USD and Isaac Lab submodule. SAM 3D's full CUDA environment still follows upstream installation instructions. `lyra2_env.sh` is only for optional Lyra inference and should be sourced after `scripts/env.sh`.
