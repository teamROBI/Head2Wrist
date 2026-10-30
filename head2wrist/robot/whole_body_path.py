from head2wrist.config import H2W_OUTPUT_ROOT
#!/usr/bin/env python3
"""
Executable FFW-SG2 plan for one demo: the base + lift key setting (robot_setup.json from setup_pose.py) and the right
arm tracking the synthesized TCP path.
  base / lift  positioned once before the task, then fixed: only the arm moves (default: --pick = --place =
               robot_setup.json). Optional two-stance mode (--pick / --place different): the base and lift ease from one
               to the other over the transport
  arm          IK of the 7 joints per frame from that frame's base pose + lift, one continuous branch
  checks       every pose reached, arm joint step per frame, joint-limit margin, arm above the table, lift column and
               chest never inside the tabletop, base / lift speeds
Writes <inputs>/robot_plan.npz (per wrist frame: base pose, lift, arm joints) and robot_plan.json (summary).
"""
import argparse, json, os, sys
import numpy as np
import head2wrist.robot.sg2_kin as K



def setup_at(i):
    if i <= t0: return kp['pick']
    if i >= t1: return kp['place']
    x = (i - t0)/(t1 - t0); e = x*x*x*(10 - 15*x + 6*x*x); return (1 - e)*kp['pick'] + e*kp['place']


def main():
    global B, Lf, P, S, SIDE, TCP, Tb, _, a, ap, fc, fw, hd, hits, i, k, kp, lf, lk, models, phase, prev, q, r, r_, rows, rt, su, summ, t0, t1, tr, v, v_base, v_lift, w_base, wrist
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default=f'{H2W_OUTPUT_ROOT}/lyra2/task_inputs/basic_pick_place_1')
    ap.add_argument('--side', default=None, choices=['right', 'left'], help='robot arm (default: from the task inputs meta.json)')
    ap.add_argument('--pick', default='robot_setup.json')
    ap.add_argument('--place', default='robot_setup.json')
    ap.add_argument('--fps', type=float, default=16.0)
    ap.add_argument('--max_step', type=float, default=0.15)
    a = ap.parse_args()
    SIDE = a.side or json.load(open(os.path.join(a.inputs, 'meta.json'))).get('side', 'right')
    S = K.Scene(a.inputs)
    P = np.load(os.path.join(a.inputs, 'task_plan.npz'))
    phase = list(P['phase'])
    TCP = P['tcp']
    wrist = [i for i in range(len(TCP)) if np.isfinite(TCP[i]).all()]
    su = {k: json.load(open(os.path.join(a.inputs, getattr(a, k))))['summary']['setup'] for k in ('pick', 'place')}
    kp = {k: np.array([v['base_forward_m'], v['base_right_m'], v['heading_deg'], v['lift_m']]) for k, v in su.items()}
    tr = [i for i in wrist if phase[i] == 'transport']
    t0, t1 = tr[0], tr[-1]
    rows, q, prev = [], None, None
    models = {}
    for i in wrist:
        fw, rt, hd, lf = setup_at(i); Tb = S.base_pose(fw, rt, hd)
        lk = round(float(lf), 4)
        if lk not in models: models[lk] = K.reduced(lk, SIDE)
        # one frame through solve_path (warm start from the previous frame's joints)
        r_, _ = K.solve_path(Tb, lk, [S.cam_to_world(TCP[i])], S.table_h, seeds=8, q_start=None if q is None else np.array(q), side=SIDE)
        r = r_[0]; hits = S.body_hits(Tb, lf)
        r.update(frame=i, phase=phase[i], base=[float(fw), float(rt), float(hd)], lift=float(lf),
                 dq=0.0 if q is None else float(np.abs(np.array(r['q']) - np.array(q)).max()), body_hits=sum(hits.values()))
        rows.append(r); q = r['q']
    fc = [K.grip_clearance(S.cam_to_world(TCP[i]), S.table_h, SIDE)[0] for i in wrist]
    B = np.array([r['base'] for r in rows])
    Lf = np.array([r['lift'] for r in rows])
    v_base = np.linalg.norm(np.diff(B[:, :2], axis=0), axis=1)*a.fps
    w_base = np.abs(np.diff(B[:, 2]))*a.fps
    v_lift = np.abs(np.diff(Lf))*a.fps
    summ = dict(key_positions={k: dict(base_forward_m=float(v[0]), base_right_m=float(v[1]), heading_deg=float(v[2]), lift_m=float(v[3])) for k, v in kp.items()},
                frames=len(rows), reached_frac=float(np.mean([r['ok'] for r in rows])),
                max_arm_step_rad=float(max(r['dq'] for r in rows)), frames_over_max_step=int(sum(r['dq'] > a.max_step for r in rows)),
                joint_margin_min_rad=float(min(r['margin'] for r in rows)), joint_margin_p10_rad=float(np.percentile([r['margin'] for r in rows], 10)),
                arm_clearance_min_cm=100*float(min(r['arm_clear'] for r in rows)), finger_clearance_min_cm=100*float(min(fc)),
                body_table_hits=int(sum(r['body_hits'] for r in rows)),
                base_speed_max_m_s=float(v_base.max()), base_turn_max_deg_s=float(w_base.max()), lift_speed_max_m_s=float(v_lift.max()),
                base_move_m=float(np.linalg.norm(kp['place'][:2] - kp['pick'][:2])), base_turn_deg=float(kp['place'][2] - kp['pick'][2]),
                transition_s=float((t1 - t0)/a.fps))
    np.savez(os.path.join(a.inputs, 'robot_plan.npz'), frames=np.array(wrist), base=B, lift=Lf, arm_q=np.array([r['q'] for r in rows]),
             ok=np.array([r['ok'] for r in rows]))
    json.dump(summ, open(os.path.join(a.inputs, 'robot_plan.json'), 'w'), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
