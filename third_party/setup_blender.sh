#!/bin/bash
# Blender as a Python module (bpy, same engine incl. Cycles GPU) for the scripted Blender scene proxy of the Head2Wrist
# proxy A/B test: build the proxy scene, set the wrist camera path, render passes, save a .blend to inspect / edit.
# uv env (bpy 5.2 requires Python 3.13). Log: third_party/setup_blender.log
set -e
source "$(dirname "$0")/../scripts/env.sh"
cd ${H2W_REPO_ROOT}/third_party
mkdir -p blender && cd blender
[ -d .venv ] || uv venv --python 3.13 --seed .venv
source .venv/bin/activate
uv pip install "bpy==5.2.2" numpy opencv-python-headless scipy trimesh
python -c "
import bpy; print('blender', bpy.app.version_string)
p = bpy.context.preferences.addons['cycles'].preferences
for t in ('OPTIX', 'CUDA'):
    try:
        p.compute_device_type = t; p.get_devices(); print(t, [d.name for d in p.devices if d.type == t])
    except Exception as e: print(t, 'unavailable', e)
"
