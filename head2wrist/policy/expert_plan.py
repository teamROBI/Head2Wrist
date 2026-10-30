from head2wrist.config import H2W_OUTPUT_ROOT
#!/usr/bin/env python3
"""
Scripted expert for the sim policy test (step 2): pick-and-place episodes in one scene's Isaac twin with the grasped
object's start pose randomized. The grasp is the one the sim gate validated for this scene (twin.json check.tcp_world at
the grasp frame), expressed in the object's frame and carried with the object (offset dx, dy and yaw about its vertical
axis); the release pose is the validated one (the object is held the same way, so it lands at the same place). Path:
start (the scene's overview TCP) -> pre-grasp (10 cm back along the approach) -> grasp (hold: jaws close) -> lift 10 cm
-> above the release (8 cm) -> release (hold: jaws open) -> retreat 10 cm; straight segments at ~12 cm/s, 16 fps; IK by
sg2_kin.solve_path with the scene's base pose and lift (the arm only moves). Episodes not fully reachable, with the arm
or fingers near the table, or with the object overlapping another one are skipped. A path with a joint step > 0.15 rad
(near the shoulder singularity) is slowed down there (refine) instead of skipped.
Writes $H2W_OUTPUT_ROOT/sim_policy/<name>/plans.json.
"""
import argparse, json, os, sys
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
X = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); pass
import head2wrist.robot.sg2_kin as K



def seg(A, B):
    n = max(8, int(np.ceil(np.linalg.norm(B[:3, 3] - A[:3, 3])/(SPEED/FPS))))
    sl = Slerp([0, 1], R.from_matrix([A[:3, :3], B[:3, :3]])); out = []
    for s in np.linspace(0, 1, n + 1)[1:]:
        s_ = s*s*(3 - 2*s); T = np.eye(4); T[:3, :3] = sl(s_).as_matrix(); T[:3, 3] = (1 - s_)*A[:3, 3] + s_*B[:3, 3]; out.append(T)
    return out


def refine(Tb_e, poses, phase, rows, lim=0.10, rounds=4):
    """Slow down where a joint steps > lim per frame (near the shoulder singularity, joint 1 ~ pi/2, where joints 0 and 2
    align): split those steps into finer TCP sub-steps (same phase) and re-solve, until every step is <= lim."""
    for _ in range(rounds):
        dq = [0.0] + [r['dq'] for r in rows[1:]]
        if max(dq) <= lim or not all(r['ok'] for r in rows): break
        P, H = [poses[0]], [phase[0]]
        for k in range(1, len(poses)):
            n = int(np.ceil(dq[k]/lim)) if dq[k] > lim else 1
            sl = Slerp([0, 1], R.from_matrix([poses[k - 1][:3, :3], poses[k][:3, :3]]))
            for s in np.linspace(0, 1, n + 1)[1:]:
                T = np.eye(4); T[:3, :3] = sl(s).as_matrix(); T[:3, 3] = (1 - s)*poses[k - 1][:3, 3] + s*poses[k][:3, 3]; P.append(T); H.append(phase[k])
        poses, phase = P, H
        rows, _ = K.solve_path(Tb_e, lift, poses, th, seeds=4, seed=1, side=side, q_start=np.array(RB['arm_q'][0]))
    return poses, phase, rows


def up(T, dz): T = T.copy(); T[2, 3] += dz; return T


def back(T, d): T = T.copy(); T[:3, 3] -= d*T[:3, 2]; return T


def main():
    global A_, B_, CK, FPS, G_obj, HOLD, OUT, RB, SPEED, T, TCP, TW, T_o, T_o0, T_rel, T_start, T_start_e, Tabove, Tb, Tb_e, Tg, Tlift, Tpre, Tret, _, a, ap, arm_clear, boff, c, c0, cy, dT, df, dq, ds_, dx, dy, dyaw, fing, g, g_close, g_open, i, i_g, i_r, ig, ir, lift, name, o, obj, ok, others, p, ph, phase, plans, poses, r, rng, rows, s_, side, ss, sy, th, tried, yaw
    ap = argparse.ArgumentParser()
    ap.add_argument('--ep', default='stack/18')
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--range', type=float, default=0.08, help='object start offset range (m, each axis)')
    ap.add_argument('--yaw', type=float, default=30.0, help='object start yaw range (deg)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--min_range', type=float, default=0.0, help='out-of-distribution sets: skip offsets with both |dx|, |dy| below this (m)')
    ap.add_argument('--base_pert', action='store_true', help='stance-randomized demos: per episode the base moved forward -5..+2 cm, sideways +-5 cm, yaw +-10 deg (the moved-base test range); the expert plans from there, starting from the same arm joints')
    ap.add_argument('--out_name', default='plans', help='<out_name>.json (e.g. plans_eval: held-out start poses)')
    a = ap.parse_args()
    name = a.ep.replace('/', '_')
    TW = json.load(open(f'{H2W_OUTPUT_ROOT}/sim_twin/{name}/twin.json'))
    RB, CK = TW['robot'], TW['check']
    OUT = f'{H2W_OUTPUT_ROOT}/sim_policy/{name}'
    os.makedirs(OUT, exist_ok=True)
    side = TW['side']
    Tb = np.array(RB['base_pose'])
    lift = RB['lift']
    th = TW['table_h']
    TCP = np.array(CK['tcp_world'])
    ig, ir = CK['grasp_frame'], CK['release_frame']
    obj = next(o for o in TW['objects'] if o['role'] == 'grasped')
    others = [o for o in TW['objects'] if o['role'] != 'grasped']
    c0 = np.array(obj['centre'])
    T_o0 = np.eye(4)
    T_o0[:3, 3] = c0
    G_obj = np.linalg.inv(T_o0) @ TCP[ig]
    T_rel = TCP[ir]
    T_start = TCP[0]
    FPS, SPEED, HOLD = 16, 0.12, 10
    g_open, g_close = RB['gripper_open'], RB['gripper_close']
    rng = np.random.default_rng(a.seed)
    plans = []
    tried = 0
    while len(plans) < a.n and tried < 20*a.n:
        tried += 1
        dx, dy = rng.uniform(-a.range, a.range, 2); yaw = rng.uniform(-a.yaw, a.yaw)
        if max(abs(dx), abs(dy)) < a.min_range: continue
        c = c0 + [dx, dy, 0]
        if any(np.all(np.abs(c[:2] - np.array(o['centre'][:2])) < (np.array(o['size'][:2]) + np.array(obj['size'][:2]))/2 + 0.02) for o in others):
            continue                                                           # would overlap another object
        T_o = np.eye(4); T_o[:3, :3] = R.from_euler('z', yaw, degrees=True).as_matrix(); T_o[:3, 3] = c
        Tb_e, T_start_e, boff = Tb, T_start, [0.0, 0.0, 0.0]
        if a.base_pert:
            df, ds_, dyaw = rng.uniform(-0.05, 0.02), rng.uniform(-0.05, 0.05), rng.uniform(-10, 10)
            cy, sy = np.cos(np.radians(dyaw)), np.sin(np.radians(dyaw)); dT = np.eye(4); dT[:2, :2] = [[cy, -sy], [sy, cy]]; dT[:2, 3] = [df, ds_]
            Tb_e = Tb @ dT; T_start_e = Tb_e @ np.linalg.inv(Tb) @ T_start; boff = [float(df), float(ds_), float(dyaw)]   # same arm joints -> start moves with the base
        Tg = T_o @ G_obj; Tpre = back(Tg, 0.10); Tlift = up(Tg, 0.10); Tabove = up(T_rel, 0.08); Tret = up(T_rel, 0.10)
        poses, phase = [T_start_e], ['start']
        for A_, B_, ph in ((T_start_e, Tpre, 'approach'), (Tpre, Tg, 'descent')):
            s_ = seg(A_, B_); poses += s_; phase += [ph]*len(s_)
        poses += [Tg]*HOLD; phase += ['grasp']*HOLD
        for A_, B_, ph in ((Tg, Tlift, 'lift-off'), (Tlift, Tabove, 'transport'), (Tabove, T_rel, 'set-down')):
            s_ = seg(A_, B_); poses += s_; phase += [ph]*len(s_)
        poses += [T_rel]*HOLD; phase += ['release']*HOLD
        s_ = seg(T_rel, Tret); poses += s_; phase += ['retreat']*len(s_)
        rows, _ = K.solve_path(Tb_e, lift, poses, th, seeds=4, seed=int(tried), side=side, q_start=np.array(RB['arm_q'][0]))
        if all(r['ok'] for r in rows) and max(r['dq'] for r in rows[1:]) > 0.15: poses, phase, rows = refine(Tb_e, poses, phase, rows)
        ok = all(r['ok'] for r in rows); dq = max(r['dq'] for r in rows[1:]); arm_clear = min(r['arm_clear'] for r in rows)
        fing = min(K.grip_clearance(T, th, side)[0] for T in poses)
        if not ok or dq > 0.15 or arm_clear < 0.01 or fing < -0.002:
            continue
        i_g = phase.index('grasp'); i_r = phase.index('release'); g = np.full(len(phase), g_open); ss = lambda x: x*x*(3 - 2*x)
        for i, p in enumerate(phase):
            if p == 'grasp': g[i] = g_open + (g_close - g_open)*ss(min(1.0, (i - i_g)/7))
            elif p in ('lift-off', 'transport', 'set-down'): g[i] = g_close
            elif p == 'release': g[i] = g_close + (g_open - g_close)*ss(min(1.0, (i - i_r)/7))
        plans.append(dict(i=len(plans), object_offset=[float(dx), float(dy), float(yaw)], object_start=c.tolist(),
                          object_quat_wxyz=np.roll(R.from_euler('z', yaw, degrees=True).as_quat(), 1).tolist(),
                          arm_q=[r['q'] for r in rows], gripper_q=g.tolist(), phase=phase, tcp=[T.tolist() for T in poses],
                          place_planned=CK['object_place_planned'], dq_max=dq, arm_clear_min=arm_clear, finger_clear_min=fing,
                          **({'base_pose': Tb_e.tolist(), 'base_offset': boff} if a.base_pert else {})))
        print(f'ep {len(plans) - 1}: offset ({100*dx:+.1f}, {100*dy:+.1f}) cm, yaw {yaw:+.0f} deg, {len(phase)} frames', flush=True)
    json.dump(dict(scene=a.ep, side=side, fps=FPS, base_pose=RB['base_pose'], lift=lift, arm_joints=RB['arm_joints'],
                   gripper_open=g_open, gripper_close=g_close, gripper_max=RB['gripper_max'], plans=plans),
              open(f'{OUT}/{a.out_name}.json', 'w'))
    print(f'{len(plans)} episodes ({tried} sampled) -> {OUT}/{a.out_name}.json')


if __name__ == '__main__':
    main()
