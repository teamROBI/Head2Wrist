from head2wrist.config import H2W_EGODEX_ROOT, H2W_OUTPUT_ROOT
#!/usr/bin/env python3
"""
LeRobot dataset (eef state, the sim recipe's format: to_lerobot_repr.py --state eef) from REAL pipeline episodes, for the
eef_hybrid policy (train_repr.py, REPR=eef_hybrid AUG=strong):
  images   the generated wrist clip (composite/<name><tag>/clip.npz frames_raw_bgr: 424x240, the D405's raw orientation, RGB here)
  state    the planned TCP pose in base_link (FK of robot_plan.npz arm_q at the planned lift) + the jaw width of the gripper
           joint the clip draws (clip gripper_joint)
  action   the next frame's pose + width (the last frame repeats), as the sim demos (actions shifted one frame)
Stance variants (--variants k, the sim's stance-randomized demos): the robot is positioned by hand once, so its base is
off by a few cm / degrees from the plan. A variant moves the base (forward -5..+2 cm, sideways +-5 cm, yaw +-10 deg, the
ranges of expert_plan.py --base_pert) and keeps the TCP's path in the world, so the wrist images stay exactly valid and
only the base-frame state / action change. Each variant is re-solved with the planner's IK (sg2_kin.solve_path: reached
everywhere, joint steps <= --max_dq, arm above the table) and dropped when infeasible; up to 20 draws per wanted variant.
The robot body layer / shadows in the images follow the planned arm, a small mismatch for a variant (body cover ~0.5%).
Writes <root>/egoh2w_episodes.json: per dataset episode the source, the base offset, the lift, side and the first joints
(egoh2w_bridge.py --start_q). Run in the LeRobot venv (third_party/lerobot/.venv):
  real_to_lerobot.py --eps stack/48,... --root <dir> --repo_id egoh2w/<name> [--tag _v5] [--variants 4]
"""
import argparse, json, os, shutil, sys
import numpy as np
X = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); pass; pass
from head2wrist.policy.repr_lib import Kin, state_vec, p9_of, width_of
import head2wrist.robot.sg2_kin as K

OUT = f'{H2W_OUTPUT_ROOT}'


def episode(ep):
    name = ep.replace('/', '_'); T = f'{OUT}/lyra2/task_inputs/{name}'
    c = np.load(f'{OUT}/composite/{name}{a.tag}/clip.npz'); TW = json.load(open(f'{OUT}/sim_twin/{name}/twin.json'))
    RP = np.load(f'{T}/robot_plan.npz'); Q = RP['arm_q'].astype(float); lift = float(RP['lift'][0]); side = TW['side']
    assert len(c['frames_raw_bgr']) == len(Q) and (c['plan_frame'] == RP['frames']).all(), f'{ep}: the clip and the robot plan disagree'
    meta = json.load(open(f'{T}/meta.json')); task = a.task
    if task is None:
        import h5py
        with h5py.File(os.path.join(meta.get('root', f'{H2W_EGODEX_ROOT}'), ep + '.hdf5')) as f:
            task = str(f.attrs.get('llm_description', '') or f.attrs.get('description', '') or 'pick and place')
    rgb = c['frames_raw_bgr'][..., ::-1].copy(); g = c['gripper_joint'].astype(float)
    return dict(name=name, ep=ep, Q=Q, g=g, lift=lift, side=side, Tb=np.array(TW['robot']['base_pose'], float), table_h=float(TW['table_h']), rgb=rgb, task=task)


def variants(E, rng):
    """[(offset (forward m, sideways m, yaw deg), base-frame TCP poses (n,4,4), first joints)] - the plan first"""
    kin = Kin(E['lift'], E['side']); Tl = np.stack([kin.fk(q) for q in E['Q']]); Tw = E['Tb'] @ Tl
    out = [((0.0, 0.0, 0.0), Tl, E['Q'][0])]; tries = 0
    while len(out) < 1 + a.variants and tries < 20*max(1, a.variants):
        tries += 1; df, ds_, dyaw = rng.uniform(-0.05, 0.02), rng.uniform(-0.05, 0.05), rng.uniform(-10, 10)
        cy, sy = np.cos(np.radians(dyaw)), np.sin(np.radians(dyaw)); dT = np.eye(4); dT[:2, :2] = [[cy, -sy], [sy, cy]]; dT[:2, 3] = [df, ds_]
        Tb_e = E['Tb'] @ dT
        rows, _ = K.solve_path(Tb_e, E['lift'], list(Tw), E['table_h'], seeds=4, seed=tries, q_start=E['Q'][0], side=E['side'])
        if not all(r['ok'] for r in rows) or max(r['dq'] for r in rows[1:]) > a.max_dq or min(r['arm_clear'] for r in rows) < 0.01: continue
        out.append(((float(df), float(ds_), float(dyaw)), np.linalg.inv(dT) @ Tl, np.array(rows[0]['q'])))
    return out, tries


def main():
    global E, LeRobotDataset, Ta, Tl, V, a, act, ap, ds, ep, feat, g, ga, i, info, lifts, off, q0, rng, sides, t, tries, x
    ap = argparse.ArgumentParser()
    ap.add_argument('--eps', required=True, help='comma list of task/idx (the pipeline episodes)')
    ap.add_argument('--root', required=True)
    ap.add_argument('--repo_id', required=True)
    ap.add_argument('--tag', default='_v5', help='composite/<name><tag>')
    ap.add_argument('--variants', type=int, default=4, help='stance variants per episode, besides the planned stance')
    ap.add_argument('--max_dq', type=float, default=0.15, help='largest joint step per frame (rad) a variant may need')
    ap.add_argument('--task', default=None, help='task string (default: the episode hdf5 description)')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    feat = {'observation.state': dict(dtype='float32', shape=(10,), names=[f's{i}' for i in range(10)]),
            'action': dict(dtype='float32', shape=(10,), names=['x', 'y', 'z'] + [f'r{i}' for i in range(6)] + ['width']),
            'observation.images.wrist': dict(dtype='video', shape=(240, 424, 3), names=['height', 'width', 'channels'])}
    if os.path.exists(a.root): shutil.rmtree(a.root)
    ds = LeRobotDataset.create(repo_id=a.repo_id, fps=16, features=feat, robot_type='ffw_sg2', root=a.root, use_videos=True)
    rng = np.random.default_rng(a.seed)
    info = []
    for ep in a.eps.split(','):
        E = episode(ep); V, tries = variants(E, rng); g = E['g']; ga = np.r_[g[1:], g[-1:]]
        for off, Tl, q0 in V:
            Ta = np.concatenate([Tl[1:], Tl[-1:]], 0); act = np.concatenate([p9_of(Ta), width_of(ga)[:, None]], 1).astype(np.float32)
            for t in range(len(Tl)):
                ds.add_frame({'observation.state': state_vec('eef', None, g[t], Tl[t]).astype(np.float32), 'action': act[t],
                              'observation.images.wrist': E['rgb'][t]}, task=E['task'])
            ds.save_episode()
            info.append(dict(dataset_episode=len(info), ep=ep, base_offset_fwd_side_yaw=off, lift=E['lift'], side=E['side'], frames=len(Tl),
                             start_q=[float(x) for x in q0] + [float(g[0])], task=E['task']))
        print(f'{ep}: {len(V) - 1}/{a.variants} stance variants feasible ({tries} draws), {len(E["Q"])} frames each, task "{E["task"]}"', flush=True)
    lifts = sorted({round(x['lift'], 3) for x in info})
    sides = sorted({x['side'] for x in info})
    if len(sides) > 1 or (max(lifts) - min(lifts)) > 0.02: print(f'WARNING: one policy serves one arm at one lift; this set has sides {sides}, lifts {lifts}')
    json.dump(dict(eps=a.eps.split(','), tag=a.tag, lifts=lifts, sides=sides, episodes=info), open(f'{a.root}/egoh2w_episodes.json', 'w'), indent=1)
    print(f'{len(info)} episodes ({len(a.eps.split(","))} demos) -> {a.root}')


if __name__ == '__main__':
    main()
