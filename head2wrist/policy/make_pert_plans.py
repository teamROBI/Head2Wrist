from head2wrist.config import H2W_OUTPUT_ROOT
#!/usr/bin/env python3
"""
Stance-perturbation test set for the sim policy test: the held-out start poses of plans_eval.json, each with the robot
base moved off the stance the demos were recorded at (forward -5..+2 cm - little toward the table -, sideways +-5 cm,
yaw +-10 deg about the base) - the registration error a real deployment has between the retargeted plan and the actual
robot base. An episode is kept only if the expert's pre-grasp / grasp / lift / above-release / release TCP poses stay
reachable from the moved base (planner IK). The robot starts from the same joint configuration (so its gripper starts
displaced with the base). Writes plans_eval_pert.json (plans with 'base_pose' and 'base_offset').
"""
import argparse, json, os, sys
import numpy as np
X = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); pass
import head2wrist.robot.sg2_kin as K





def main():
    global D, P, PL, Q, TW, Tb, Tb0, _, a, ap, c, dT, df, ds, dyaw, i, idx, out, ph, r, rng, rows, s_, th
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='stack_18')
    ap.add_argument('--n', type=int, default=24)
    ap.add_argument('--seed', type=int, default=5)
    a = ap.parse_args()
    D = f'{H2W_OUTPUT_ROOT}/sim_policy/{a.scene}'
    PL = json.load(open(f'{D}/plans_eval.json'))
    TW = json.load(open(f'{H2W_OUTPUT_ROOT}/sim_twin/{a.scene}/twin.json'))
    th = TW['table_h']
    Tb0 = np.array(PL['base_pose'])
    rng = np.random.default_rng(a.seed)
    out = []
    for P in PL['plans']:
        if len(out) >= a.n: break
        for _ in range(12):
            df, ds, dyaw = rng.uniform(-0.05, 0.02), rng.uniform(-0.05, 0.05), np.radians(rng.uniform(-10, 10))
            c, s_ = np.cos(dyaw), np.sin(dyaw); dT = np.eye(4); dT[:2, :2] = [[c, -s_], [s_, c]]; dT[:2, 3] = [df, ds]
            Tb = Tb0 @ dT                                                       # offset in the base's own frame
            ph = P['phase']; idx = sorted({ph.index('descent') - 1, ph.index('grasp'), ph.index('transport'), ph.index('set-down'), ph.index('release')})
            rows, _ = K.solve_path(Tb, PL['lift'], [np.array(P['tcp'][i]) for i in idx], th, seeds=6, seed=1, side=PL['side'], q_start=np.array(P['arm_q'][idx[0]]))
            if all(r['ok'] for r in rows) and min(r['arm_clear'] for r in rows) > 0.01:
                Q = dict(P); Q['base_pose'] = Tb.tolist(); Q['base_offset'] = [float(df), float(ds), float(np.degrees(dyaw))]; Q['i'] = len(out); Q['src_i'] = P['i']
                out.append(Q); print(f"ep {len(out) - 1} (held-out {P['i']}): base {100*df:+.1f} / {100*ds:+.1f} cm, {np.degrees(dyaw):+.1f} deg", flush=True); break
    json.dump(dict(PL, plans=out), open(f'{D}/plans_eval_pert.json', 'w'))
    print(f'{len(out)} perturbed-stance episodes -> {D}/plans_eval_pert.json')


if __name__ == '__main__':
    main()
