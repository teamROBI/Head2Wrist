#!/usr/bin/env python3
"""
Sim demos (collect_demos.py ep_*.npz) -> a local LeRobot dataset (LeRobot 0.3.3, third_party/lerobot/.venv) for the
policy test. Successful episodes only. Features: observation.state / action (8: the 7 arm joints + gripper joint1),
observation.images.wrist (424x240, the raw D405 orientation) and/or observation.images.head (the human head view,
480x270). --wrist_from <dir>: the wrist frames come from <dir>/ep_<i>.npy (T, 240, 424, 3) instead of the true sim
wrist camera (the generated-wrist arm of the test), same episodes and actions.
Usage: .../lerobot/.venv/bin/python to_lerobot.py --demos <dir> --root <out> --repo_id egoh2w/<name> [--views wrist,head]
"""
import argparse, glob, os, shutil
import numpy as np





def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    global A, J, W, a, ap, dd, dirs, ds, f, feat, files, fr, i, keep, n_ep, n_fr, t, views, wd, wdirs, x, z
    ap = argparse.ArgumentParser()
    ap.add_argument('--demos', required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--repo_id', required=True)
    ap.add_argument('--views', default='wrist')
    ap.add_argument('--wrist_from', default=None)
    ap.add_argument('--episodes', default=None, help='comma list of episode ids (default all)')
    ap.add_argument('--task', default='Pick up the creamer and place it on the box.')
    a = ap.parse_args()
    views = a.views.split(',')
    J = [f'arm_joint{i}' for i in range(1, 8)] + ['gripper']
    feat = {'observation.state': dict(dtype='float32', shape=(8,), names=J), 'action': dict(dtype='float32', shape=(8,), names=J)}
    if 'wrist' in views: feat['observation.images.wrist'] = dict(dtype='video', shape=(240, 424, 3), names=['height', 'width', 'channels'])
    if 'head' in views: feat['observation.images.head'] = dict(dtype='video', shape=(270, 480, 3), names=['height', 'width', 'channels'])
    if os.path.exists(a.root): shutil.rmtree(a.root)
    ds = LeRobotDataset.create(repo_id=a.repo_id, fps=16, features=feat, robot_type='ffw_sg2', root=a.root, use_videos=True)
    dirs = a.demos.split(',')
    wdirs = a.wrist_from.split(',') if a.wrist_from else [None]*len(dirs)
    assert len(wdirs) == len(dirs)
    files = [(f, wd) for dd, wd in zip(dirs, wdirs) for f in sorted(glob.glob(f'{dd}/ep_*.npz'))]
    if a.episodes: keep = {int(x) for x in a.episodes.split(',')}; files = [(f, wd) for f, wd in files if int(os.path.basename(f)[3:7]) in keep]
    n_ep = n_fr = 0
    for f, wd in files:
        z = np.load(f)
        if not bool(z['success']): continue
        W = np.load(f"{wd}/{os.path.basename(f)[:-4]}.npy") if wd else z['wrist']
        assert len(W) == len(z['state']), (f, len(W), len(z['state']))
        # the collector records each frame's observation AFTER the arm moved to that frame's target, so the recorded action
        # equals the current state: the action that follows observation t is the target recorded at t + 1
        A = np.concatenate([z['action'][1:], z['action'][-1:]], 0)
        for t in range(len(z['state'])):
            fr = {'observation.state': z['state'][t].astype(np.float32), 'action': A[t].astype(np.float32)}
            if 'wrist' in views: fr['observation.images.wrist'] = W[t]
            if 'head' in views: fr['observation.images.head'] = z['human_head'][t]
            ds.add_frame(fr, task=a.task)
        ds.save_episode(); n_ep += 1; n_fr += len(z['state'])
    print(f'{n_ep} episodes, {n_fr} frames -> {a.root}')


if __name__ == '__main__':
    main()
