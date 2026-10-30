from head2wrist.config import H2W_OUTPUT_ROOT
#!/usr/bin/env python3
"""
LeRobot datasets for the EEF representations of the sweep (repr_lib.py), from the sim demos: FK of the recorded joints
-> TCP poses in base_link. --state eef: state p+r6+width (10), action the next absolute TCP pose + width (10).
--state umi: state = UMI-style relative proprio (16), action the next absolute TCP pose + width (10) (train_repr.py makes
it relative to the current pose per sample), aux.eef_abs = the current absolute pose + width (10; not a policy input).
Actions are shifted one frame (the target that follows observation t, as to_lerobot.py). Same episodes / wrist sources
as to_lerobot.py (--wrist_from: generated frames).
"""
import argparse, glob, json, os, shutil
import numpy as np
from head2wrist.policy.repr_lib import Kin, state_vec, p9_of, width_of





def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    global A, S, SD, T, TW, Ta, W, a, act, ap, dd, dirs, ds, f, feat, fr, i, kin, n, n_ep, okk, q, st, t, wd, wdirs, z
    ap = argparse.ArgumentParser()
    ap.add_argument('--demos', required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--repo_id', required=True)
    ap.add_argument('--state', required=True, choices=['eef', 'umi'])
    ap.add_argument('--wrist_from', default=None)
    ap.add_argument('--scene', default='stack_18')
    ap.add_argument('--task', default='Pick up the creamer and place it on the box.')
    a = ap.parse_args()
    TW = json.load(open(f'{H2W_OUTPUT_ROOT}/sim_twin/{a.scene}/twin.json'))
    kin = Kin(TW['robot']['lift'], TW['side'])
    SD = 10 if a.state == 'eef' else 16
    feat = {'observation.state': dict(dtype='float32', shape=(SD,), names=[f's{i}' for i in range(SD)]),
            'action': dict(dtype='float32', shape=(10,), names=['x', 'y', 'z'] + [f'r{i}' for i in range(6)] + ['width']),
            'observation.images.wrist': dict(dtype='video', shape=(240, 424, 3), names=['height', 'width', 'channels'])}
    if a.state == 'umi': feat['aux.eef_abs'] = dict(dtype='float32', shape=(10,), names=['x', 'y', 'z'] + [f'r{i}' for i in range(6)] + ['width'])
    if os.path.exists(a.root): shutil.rmtree(a.root)
    ds = LeRobotDataset.create(repo_id=a.repo_id, fps=16, features=feat, robot_type='ffw_sg2', root=a.root, use_videos=True)
    dirs = a.demos.split(',')
    wdirs = a.wrist_from.split(',') if a.wrist_from else [None]*len(dirs)
    n_ep = 0
    for dd, wd in zip(dirs, wdirs):
        for f in sorted(glob.glob(f'{dd}/ep_*.npz')):
            try: z = np.load(f); okk = bool(z['success']); S = z['state']
            except Exception as e: print(f, 'unreadable (collector stopped mid-write?), skipped:', e); continue
            if not okk: continue
            if wd and not os.path.exists(f"{wd}/{os.path.basename(f)[:-4]}.npy"): print(f, 'no generated view, skipped'); continue
            W = np.load(f"{wd}/{os.path.basename(f)[:-4]}.npy") if wd else z['wrist']
            S = z['state']; A = np.concatenate([z['action'][1:], z['action'][-1:]], 0); n = len(S)
            T = np.stack([kin.fk(q) for q in S[:, :7]]); Ta = np.stack([kin.fk(q) for q in A[:, :7]])
            act = np.concatenate([p9_of(Ta), width_of(A[:, 7])[:, None]], 1).astype(np.float32)
            for t in range(n):
                st = state_vec(a.state, S[t, :7], S[t, 7], T[t], T[max(t - 1, 0)], T[0]).astype(np.float32)
                fr = {'observation.state': st, 'action': act[t], 'observation.images.wrist': W[t]}
                if a.state == 'umi': fr['aux.eef_abs'] = np.r_[p9_of(T[t]), width_of(S[t, 7])].astype(np.float32)
                ds.add_frame(fr, task=a.task)
            ds.save_episode(); n_ep += 1
    print(f'{n_ep} episodes -> {a.root} (state {a.state})')


if __name__ == '__main__':
    main()
