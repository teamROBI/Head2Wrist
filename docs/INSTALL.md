# Installation

The recorded main environment is Python **3.11.14** (uv). No data-directory symlinks or import-path aliases are needed. Install ffmpeg (with `libx264`), a C++ runtime and OpenGL runtime libraries appropriate for Open3D, and an NVIDIA driver compatible with the chosen CUDA wheel.

```bash
uv venv --python 3.11.14
source .venv/bin/activate
uv pip install -e '.[vision]'
# CUDA 12.8 wheels were used for the main environment:
uv pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
bash third_party/setup_geometry.sh
bash third_party/setup_robotis.sh
bash third_party/lerobot/setup.sh
```

Main environment versions are pinned in `pyproject.toml`; Depth Anything 3 is installed separately by its setup recipe. Download DA3-NESTED-GIANT-LARGE-1.1 into `$H2W_DATA_ROOT/checkpoints/da3/DA3NESTED-GIANT-LARGE-1.1`. SAM 3 is loaded through Transformers (`facebook/sam3`); obtain access to its weights and accept its terms upstream. LaMa downloads its own checkpoint on first use. SAM 3D Objects needs its own upstream CUDA/PyTorch3D environment and accepted model terms. Follow its upstream setup, set `H2W_SAM3D_ROOT` and `H2W_SAM3D_PYTHON`, and place its pipeline checkpoint under `checkpoints/hf/` in that checkout. Its un-packaged `notebook/inference.py` is loaded from that external checkout; no upstream implementation is copied here.

The CPU planning and Open3D rendering stages can run without a GPU. DA3/SAM inference and Isaac physics use a GPU. Select a device with sufficient free memory, and keep concurrent jobs within their agreed memory budget.

## Physics environment

The paper used **Isaac Sim 5.1 and Isaac Lab 2.3** through the `robotis/robotis-lab:latest` image, with Isaac Lab checkout `3c6e67bb5c7ada942a6d1884ab69338f57596f77`. The `latest` tag is mutable: select an image containing those versions with `H2W_ISAAC_IMAGE`. This release does not claim a reproducible immutable container digest.

Both required robot asset sources are public. Run `bash third_party/setup_robotis.sh` to clone [ROBOTIS ai_worker](https://github.com/ROBOTIS-GIT/ai_worker) at `6961f7e7368329d57ebf2adf0fdba08680a43949` and [ROBOTIS cyclo_lab](https://github.com/ROBOTIS-GIT/cyclo_lab) at `9dd83c5988ea1709b7f232e8922c5c55dcd68218`. The old `ROBOTIS-GIT/robotis_lab` URL redirects to `cyclo_lab`; the pinned historical revision still exposes the `robotis_lab` Python package. Install `git-lfs` if a checkout contains LFS pointers. The installer refuses to overwrite local edits and initializes the pinned Isaac Lab submodule.

The installer prints `H2W_URDF_ROOT` and `H2W_ROBOTIS_ROOT` exports. The URDF is `ffw_description/urdf/ffw_sg2_rev1_follower/ffw_sg2_follower.urdf`; the USD is `source/robotis_lab/data/robots/FFW/FFW_SG2.usd` and its Python configuration is `source/robotis_lab/robotis_lab/assets/robots/FFW_SG2.py`. Code and robot assets are Apache-2.0 at these revisions, with Isaac Lab under BSD-3-Clause and Isaac Sim under separate NVIDIA terms. Nothing is vendored into this code release. `--check-only --ai-worker-dir PATH --robotis-lab-dir PATH` verifies existing checkouts without modifying them.

The physics runner mounts the source and assets read-only into a new disposable container; it does not enter an existing robot container. `H2W_GPU_DEVICE` selects the Docker device (for example `device=0`).

## ACT and capture environments

`third_party/lerobot/setup.sh` creates **Python 3.10 / PyTorch 2.7.1 cu128 / torchvision 0.22.1 / LeRobot 0.3.3**. Source `scripts/env.sh` before invoking it or the training tools. The main package metadata targets Python 3.11; the training shell tools expose the same source through `PYTHONPATH` to the separate Python 3.10 interpreter. Do not install the Python 3.11 dependency pins into that training environment.

Quest self-calibration uses MediaPipe hand landmarks and SciPy. Install MediaPipe in a separate capture environment if it conflicts with your vision stack, expose this repository through `PYTHONPATH`, and run `python -m head2wrist.capture.calib_headcam --help` there. ROS 2 deployment uses Jazzy and the ROBOTIS ROS messages; these are supplied by the robot image, not by the main pip environment.

## Configuration

All Python path defaults come from `head2wrist/config.py`. Export variables before starting Python. Paths passed as CLI arguments take precedence. `scripts/env.sh` exports the same defaults for shell entry points; `.env.example` is a template, not an automatically loaded file.

| Variable | Default |
|---|---|
| `H2W_DATA_ROOT` | `<repo>/data` |
| `H2W_OUTPUT_ROOT` | `$H2W_DATA_ROOT/outputs` |
| `H2W_EGODEX_ROOT` | `$H2W_DATA_ROOT/egodex/test` |
| `H2W_URDF_ROOT` | `<repo>/third_party/ai_worker/ffw_description` |
| `H2W_ROBOTIS_ROOT` | `<repo>/third_party/robotis_lab` |
| `H2W_SAM3D_ROOT` | `<repo>/third_party/sam-3d-objects` |
| `H2W_PYTHON` | current Python; shell tools choose `<repo>/.venv/bin/python` |
| `H2W_LEROBOT_PYTHON` | `<repo>/third_party/lerobot/.venv/bin/python` |
| `H2W_SAM3D_PYTHON` | `$H2W_SAM3D_ROOT/.venv/bin/python` |

`H2W_REPO_ROOT` is derived from the installed source location. `H2W_ISAAC_IMAGE`, `H2W_GPU_DEVICE`, `H2W_ROS_IMAGE`, `H2W_AI_WORKER_ROOT`, and the robot network/session variables are shell-runner settings documented in [ROBOT](ROBOT.md). No deployment IP address is supplied.
