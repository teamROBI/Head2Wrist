#!/usr/bin/env python3
"""
Offline check of a real-robot policy on REAL wrist frames, before robot time. The plan replay (egoh2w_bridge.py --mode
replay_plan -> plan_record.npz) holds a real D405 frame and the measured joints at every planned pose of a trajectory that
grasped the cube. Each row is fed to the policy as if the robot stood there (state + real frame, no ensembling): a policy
that will work on the robot predicts the plan's motion from these frames - down onto the cube during the descent, a
closing gripper at the grasp. Reported per plan phase: the predicted TCP motion over the first H actions vs the plan's
motion over the same rows (z in cm), and the predicted gripper joint.
Run in third_party/lerobot/.venv:
  offline_real_check.py --ckpt <.../pretrained_model> --record <plan_record.npz> [--gsm 0.559] [--lift -0.25] [--h 10]
"""
import argparse, os, sys
import numpy as np, torch
from head2wrist.policy.repr_lib import Kin, state_vec, to_abs, joint_of, width_of, REPRS





def main():
    global ACTPolicy, Q, R, T, TP, W, a, ab, ap, ch, dz, e, g, gclose, h, i, ic, ig, img, k, kin, kind, n, ok_desc, ok_grip, pe, pg, ph, pol, pz, q, r, sv, z
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    ap.add_argument('--out', help='optional prediction archive (.npz)')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--record', required=True)
    ap.add_argument('--repr', default='eef_hybrid')
    ap.add_argument('--lift', type=float, default=-0.25)
    ap.add_argument('--side', default='right')
    ap.add_argument('--gsm', type=float, default=None, help='grip_state_max: the gripper joint the policy sees is capped here (as on the robot)')
    ap.add_argument('--h', type=int, default=10, help='actions of the chunk compared with the plan (one plan row per control tick)')
    ap.add_argument('--grip_thresh', type=float, default=0.35)
    a = ap.parse_args()
    from lerobot.policies.act.modeling_act import ACTPolicy
    pol = ACTPolicy.from_pretrained(a.ckpt).to(a.device).eval()
    kin = Kin(a.lift, a.side)
    kind = REPRS[a.repr]
    R = np.load(a.record)
    Q, W = R['q'], R['wrist']
    n = len(Q)
    T = np.stack([kin.fk(q[:7]) for q in Q])
    z = T[:, 2, 3]
    ig = int(np.argmin(z))
    gclose = np.where(Q[:, 7] > a.grip_thresh)[0]
    ic = int(gclose[0]) if len(gclose) else ig
    pz, pg, pe = np.zeros(n), np.zeros(n), np.zeros((n, 3))
    TP = np.stack([kin.fk(q[:7]) for q in R['target']])[:, :3, 3]
    chunks = []
    for i in range(n):
        g = Q[i, 7] if a.gsm is None else min(Q[i, 7], a.gsm)
        sv = state_vec(kind, Q[i, :7], g, T[i]).astype(np.float32)
        img = torch.from_numpy(np.ascontiguousarray(W[i])).permute(2, 0, 1).float().div(255).unsqueeze(0).to(a.device)
        with torch.no_grad(): ch = pol.predict_action_chunk({'observation.state': torch.from_numpy(sv).unsqueeze(0).to(a.device), 'observation.images.wrist': img})[0].cpu().numpy()
        chunks.append(ch.copy())
        ab = to_abs(a.repr, ch, Q[i, :7], T[i]); h = min(a.h, len(ab)) - 1
        pz[i] = ab[h, 2] - z[i]; pg[i] = joint_of(ab[:a.h, 9]).max(); pe[i] = ab[h, :3] - TP[min(i + h, n - 1)]
    if a.out:
        np.savez(a.out, chunks=np.stack(chunks), pz=pz, pg=pg, pe=pe, tcp=T)
    dz = np.array([z[min(i + a.h, n - 1)] - z[i] for i in range(n)])
    ph = {'approach': range(0, max(1, ig - 12)), 'descent': range(max(0, ig - 12), ig), 'at grasp': range(ig, min(n, ic + 6)),
          'lift / transport': range(min(n, ic + 6), n)}
    print(f'{os.path.basename(os.path.dirname(os.path.dirname(a.ckpt)))}/{os.path.basename(os.path.dirname(a.ckpt))}: {n} real frames, '
          f'plan grasp at row {ig} (TCP z {z[ig]:.3f}), gripper closes at row {ic}')
    print(f'{"phase":18s} {"rows":>7s}  {"plan dz (cm)":>12s}  {"pred dz (cm)":>12s}  {"pred grip max":>13s}  {"pred - plan x, y, z (cm)":>26s}')
    for k, r in ph.items():
        r = list(r)
        if not r: continue
        e = 100*pe[r].mean(0); print(f'{k:18s} {r[0]:3d}-{r[-1]:3d}  {100*dz[r].mean():12.1f}  {100*pz[r].mean():12.1f}  {pg[r].max():13.2f}  {e[0]:8.1f} {e[1]:6.1f} {e[2]:6.1f}')
    ok_desc = pz[list(ph['descent'])].mean() < -0.01 if len(ph['descent']) else False
    ok_grip = pg[list(ph['at grasp'])].max() > a.grip_thresh if len(ph['at grasp']) else False
    print(f'verdict: descends onto the cube {"YES" if ok_desc else "NO"}; closes at the grasp {"YES" if ok_grip else "NO"}')


if __name__ == '__main__':
    main()
