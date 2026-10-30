from head2wrist.config import H2W_SAM3D_PYTHON
"""Generate a physics-checked layered wrist clip from one posed head-camera demo."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np

from head2wrist.config import H2W_EGODEX_ROOT, H2W_OUTPUT_ROOT, H2W_PYTHON, H2W_REPO_ROOT
from head2wrist.pipeline.demo_events import detect_pick_place

STAGES = ['prep', 'objects', 'graspobj', 'overview', 'approach', 'transport', 'task', 'table',
          'stance', 'arm', 'objmesh', 'proxy', 'static3d', 'visibility', 'simgate', 'contact',
          'render', 'bodyshadow', 'compose_v5']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ep', required=True)
    ap.add_argument('--root', default=None)
    ap.add_argument('--side', choices=['right', 'left'])
    ap.add_argument('--from', dest='start', choices=STAGES)
    ap.add_argument('--to', dest='stop', default='compose_v5', choices=STAGES)
    ap.add_argument('--redo', default='')
    ap.add_argument('--stages', help='optional comma-separated subset, e.g. bodyshadow,compose_v5 for a cached render')
    ap.add_argument('--stance_args', default='')
    ap.add_argument('--pre_grasp', type=int, default=10)
    a = ap.parse_args()
    if Path(a.ep).is_absolute() or '..' in Path(a.ep).parts:
        ap.error('--ep must be a relative task/episode name')
    name = a.ep.replace('/', '_')
    I = Path(H2W_OUTPUT_ROOT) / 'lyra2/explore_inputs' / name
    T = Path(H2W_OUTPUT_ROOT) / 'lyra2/task_inputs' / name
    log = Path(H2W_OUTPUT_ROOT) / 'pipeline/logs' / name
    log.mkdir(parents=True, exist_ok=True)
    if a.root is None:
        a.root = (json.loads((I / 'meta.json').read_text()).get('root') if (I / 'meta.json').exists() else None) or H2W_EGODEX_ROOT
    with h5py.File(Path(a.root) / (a.ep + '.hdf5')) as f:
        J = {k: f['transforms/' + k][:, :3, 3].astype(float) for k in f['transforms'] if k.endswith(('ThumbTip', 'IndexFingerTip', 'Hand'))}
    ev = {s: detect_pick_place(J, s) for s in ('right', 'left')}
    side = a.side or next((s for s in ('right', 'left') if ev[s] is not None), None)
    if not side or ev[side] is None:
        ap.error(f'no pick-place detected: {ev}')
    tg, tr = ev[side]
    print(f'{a.ep}: {side} hand, grasp t={tg}, release t={tr}', flush=True)

    def cmd(module, *args, python=H2W_PYTHON):
        return [python, '-m', 'head2wrist.' + module, *map(str, args)]

    def copy_object():
        for suffix in ('', '_raw', '_sam3d'):
            shutil.copy2(I / f'grasp_object{suffix}.npz', T / f'object_mesh{suffix}.npz')
        shutil.copy2(I / 'grasp_object_check.jpg', T / 'object_mesh_check.jpg')

    plan = {
        'prep': ([cmd('planning.make_hover_tour', '--root', a.root, '--ep', a.ep, '--side', side,
                       '--t_max', max(2, tg - a.pre_grasp), '--sam_gpu', '--out', I.parent)], I / 'first_depth.npy'),
        'objects': ([cmd('vision.sam3_objects', '--inputs', I)], I / 'sam3_object_masks.npy'),
        'graspobj': ([cmd('planning.grasp_object', '--inputs', I, '--root', a.root, '--side', side),
                      cmd('rendering.sam3d_object', '--ep', a.ep, '--mask', I / 'grasp_object_mask.npy', '--out', I / 'grasp_object', python=H2W_SAM3D_PYTHON),
                      cmd('rendering.refine_object', '--ep', a.ep, '--mask', I / 'grasp_object_mask.npy', '--prefix', I / 'grasp_object')], I / 'grasp_object.npz'),
        'overview': ([cmd('planning.initial_view', '--inputs', I, '--side', side)], I / 'initial_view.json'),
        'approach': ([cmd('planning.plan_approach', '--inputs', I, '--root', a.root, '--side', side)], I / 'approach_plan.npz'),
        'transport': ([cmd('planning.plan_transport', '--inputs', I, '--root', a.root)], I / 'transport_plan.npz'),
        'task': ([cmd('planning.make_task_inputs', '--src', I, '--out', T, '--multiple', 120)], T / 'task_plan.npz'),
        'table': ([cmd('scene.extend_table', '--inputs', T)], T / 'table_extension.npz'),
        'stance': ([cmd('robot.setup_pose', '--inputs', T, '--egodex', a.root, *shlex.split(a.stance_args))], T / 'robot_setup.json'),
        'arm': ([cmd('robot.whole_body_path', '--inputs', T)], T / 'robot_plan.json'),
        'objmesh': ([copy_object], T / 'object_mesh.npz'),
        'proxy': ([cmd('scene.build_proxy', '--ep', a.ep)], T / 'proxy/proxy.json'),
        'static3d': ([cmd('scene.build_static_objects', '--ep', a.ep)], T / 'proxy/static3d.done'),
        'visibility': ([cmd('scene.object_visibility', '--ep', a.ep)], T / 'proxy/obj0_seen.npz'),
        'simgate': ([cmd('simulation.sim_gate', '--eps', a.ep)], Path(H2W_OUTPUT_ROOT) / 'sim_twin' / name / 'gate_pass.json'),
        'contact': ([cmd('scene.contact_ao', '--ep', a.ep)], T / 'proxy/contact_ao.png'),
        'render': ([cmd('scene.render_proxy', '--ep', a.ep, '--out_name', 'render_v5')], T / 'proxy/render_v5/rgb.mp4'),
        'bodyshadow': ([cmd('rendering.body_shadow_inputs', '--ep', a.ep)], T / 'body_layer.npz'),
        'compose_v5': ([cmd('rendering.compose_wrist', '--ep', a.ep, '--bg', T / 'proxy/render_v5/rgb.mp4', '--tag', '_v5')], Path(H2W_OUTPUT_ROOT) / 'composite' / (name + '_v5') / 'wrist_upright.mp4'),
    }
    redo = set(STAGES[STAGES.index(a.start):] if a.start else []) | set(filter(None, a.redo.split(',')))
    if redo - set(STAGES):
        ap.error(f'unknown stages: {redo - set(STAGES)}')
    env = os.environ.copy()
    env['PYTHONPATH'] = H2W_REPO_ROOT + os.pathsep + env.get('PYTHONPATH', '')
    records = []
    def save_progress():
        (log / 'run.json').write_text(json.dumps(dict(ep=a.ep, side=side,
            t_grasp=int(tg), t_release=int(tr), stages=records), indent=2))
    selected = a.stages.split(',') if a.stages else STAGES[:STAGES.index(a.stop) + 1]
    if set(selected) - set(STAGES):
        ap.error('unknown --stages value')
    for stage in selected:
        commands, output = plan[stage]
        if output.exists() and stage not in redo:
            print(f'  [{stage}] exists, skipped', flush=True)
            records.append(dict(stage=stage, cached=True))
            save_progress()
            continue
        t = time.time()
        lp = log / f'{stage}.log'
        rc = 0
        with lp.open('w') as stream:
            for command in commands:
                if callable(command):
                    command()
                else:
                    stream.write(shlex.join(command) + '\n'); stream.flush()
                    rc = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env).returncode
                    if rc:
                        break
        print(f'  [{stage}] exit {rc} in {time.time() - t:.0f} s (log {lp})', flush=True)
        if rc or not output.exists():
            records.append(dict(stage=stage, cached=False, seconds=time.time() - t, exit=rc, output_exists=output.exists()))
            save_progress()
            print(lp.read_text()[-3000:])
            raise SystemExit(f'stage {stage} failed')
        records.append(dict(stage=stage, cached=False, seconds=time.time() - t, exit=rc))
        save_progress()
    (log / 'run.json').write_text(json.dumps(dict(ep=a.ep, side=side, t_grasp=int(tg), t_release=int(tr), stages=records), indent=2))


if __name__ == '__main__':
    main()
