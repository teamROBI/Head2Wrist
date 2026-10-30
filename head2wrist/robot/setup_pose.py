from head2wrist.config import H2W_OUTPUT_ROOT
#!/usr/bin/env python3
"""
The robot's key setting for one demo: where the FFW-SG2 stands (base forward / right / heading in the table frame) and
its lift height. The base and lift are positioned once, before the task; after that only the arm moves (user decision
), so one setting must make the whole synthesized wrist path executable. For EgoDex pilots parts of the demo
may be near the edge of reach; in real tests the objects are placed within IK range.
  constraints  base footprint behind the table's near edge, or with --under_table: the base (35 cm) may go under the
               tabletop as long as the lift column and chest (body meshes) do not touch it (--table_edge, m along f from the point under the head;
               EgoDex has no table outline: the demonstrator's forearm rests at -0.06 m, so the edge is at least that
               close); lift within its 0 to -0.5 m travel; every path pose reached (IK), no joint step over --max_step
               rad per frame on one continuous arm configuration, arm >= 2 cm above the table
  objective    joint-limit margin (10th percentile over the path, plus half the minimum)
  search       coarse grid, then a finer grid around the best few
Writes <inputs>/robot_setup.json and robot_setup.png (top and side views: table, objects, base footprint, arm at
the overview / grasp / place poses, the human's torso for reference).
"""
import argparse, json, os, sys
from multiprocessing import Pool
import numpy as np, h5py, pinocchio as pin
import head2wrist.robot.sg2_kin as K



def evaluate(fwd, right, head, lift, frames, seeds=2):
    Tb = S.base_pose(fwd, right, head)
    if (not a.base_behind_edge):                                       # base may go under the tabletop; column and chest must not hit it
        if sum(S.body_hits(Tb, lift).values()) > 0: return None
    elif S.footprint_front(Tb) > a.table_edge: return None
    rows, _ = K.solve_path(Tb, lift, [TW[i] for i in frames], a.table_h, seeds=seeds, side=SIDE)
    okf = np.mean([r['ok'] for r in rows])
    gaps = np.diff(frames); steps = [r['dq']/max(g, 1) for r, g in zip(rows[1:], gaps)]
    mg = np.array([r['margin'] for r in rows])
    feasible = okf == 1 and max(steps) <= a.max_step and min(r['arm_clear'] for r in rows) >= 0.02
    return dict(fwd=fwd, right=right, heading=head, lift=lift, reached=float(okf), max_step=float(max(steps)),
                arm_clear=float(min(r['arm_clear'] for r in rows)), margin_p10=float(np.percentile(mg, 10)),
                margin_min=float(mg.min()), feasible=bool(feasible),
                score=float(okf*10 + (np.percentile(mg, 10) + 0.5*mg.min()) - 2*max(0, max(steps) - a.max_step)))


def tolerance(c):
    n_ok = 0
    for df, ds_, dyaw in PERTURB:
        cy, sy = np.cos(np.radians(dyaw)), np.sin(np.radians(dyaw)); dT = np.eye(4); dT[:2, :2] = [[cy, -sy], [sy, cy]]; dT[:2, 3] = [df, ds_]
        Tb_ = S.base_pose(c['fwd'], c['right'], c['heading']) @ dT
        if (not a.base_behind_edge) and sum(S.body_hits(Tb_, c['lift']).values()) > 0: continue
        rows_, _ = K.solve_path(Tb_, c['lift'], [TW[i] for i in keys], a.table_h, seeds=2, side=SIDE)
        steps_ = [r['dq']/max(g, 1) for r, g in zip(rows_[1:], np.diff(keys))]
        n_ok += all(r['ok'] for r in rows_) and max(steps_) <= a.max_step and min(r['arm_clear'] for r in rows_) >= 0.01
    return n_ok/len(PERTURB)


def _ev(c): return evaluate(*c, keys)


def main():
    global D, FP, FPt, P, PERTURB, Pc, S, SIDE, TCP, TW, Tb, Wp, a, ap, ax, b, best, c, cands, chain, col, dfw, dh, dl, dm, drt, f, fc, fgrid, fig, fine, fwd, grid, head, headframe_cloud, hi_, hum, i, img, k, keys, lab, lift, m, matplotlib, n, ok, onT, p, phase, phase_grasp, plt, pool, pts, q, q2, q3, qq, r, r_, res, right, rows, s, seen_, side, steps, summ, tcp_t, tf, tl, v, wrist, x
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default=f'{H2W_OUTPUT_ROOT}/lyra2/task_inputs/basic_pick_place_1')
    ap.add_argument('--side', default=None, choices=['right', 'left'], help='robot arm (default: from the task inputs meta.json)')
    ap.add_argument('--egodex', default=None, help='dataset root (default: the task inputs meta.json, else EgoDex test)')
    ap.add_argument('--table_h', type=float, default=0.74)
    ap.add_argument('--table_edge', type=float, default=-0.08)
    ap.add_argument('--max_step', type=float, default=0.15, help='max joint change per frame (rad), 2.4 rad/s at 16 fps')
    ap.add_argument('--sub', type=int, default=6)
    ap.add_argument('--base_behind_edge', action='store_true', help='keep the whole base footprint behind the table edge (default: the 35 cm base may slide under the tabletop as long as the lift column and chest do not touch it)')
    ap.add_argument('--phases', default=None, help='comma list: evaluate only these path phases (one key position for a part of the task)')
    ap.add_argument('--tag', default='')
    ap.add_argument('--lift', type=float, default=None, help='fix the lift (m, 0 .. -0.5): all demos of a task at the one lift the robot runs at')
    ap.add_argument('--robust', type=int, default=10, help='re-rank this many best fine setups by how many of the placement errors in '
                    'PERTURB (the base set by hand: forward -5..+2 cm, sideways +-5 cm, yaw +-10 deg, the stance-variant ranges) keep the '
                    'path feasible; 0 = off (the old choice: joint margin at the exact base)')
    a = ap.parse_args()
    SIDE = a.side or json.load(open(os.path.join(a.inputs, 'meta.json'))).get('side', 'right')
    S = K.Scene(a.inputs, a.egodex, a.table_h, a.table_edge)
    P = np.load(os.path.join(a.inputs, 'task_plan.npz'))
    phase = list(P['phase'])
    TCP = P['tcp']
    wrist = [i for i in range(len(TCP)) if np.isfinite(TCP[i]).all() and (a.phases is None or phase[i] in a.phases.split(','))]
    TW = {i: S.cam_to_world(TCP[i]) for i in wrist}
    keys = sorted(set(wrist[::a.sub]) | {i for i in wrist if i == wrist[0] or phase[i] != phase[i-1]} | {wrist[-1]})
    PERTURB = [(-0.05, 0, 0), (0.02, 0, 0), (0, -0.05, 0), (0, 0.05, 0), (0, 0, -10), (0, 0, 10), (-0.03, 0.04, 7), (0.01, -0.04, -7)]
    pool = Pool(int(os.environ.get('SETUP_POSE_PROCS', min(32, os.cpu_count() or 1))))
    grid = [(fwd, right, head, lift) for fwd in np.arange(-0.05 if (not a.base_behind_edge) else -0.35, -1.05, -0.1) for right in np.arange(-0.4, 0.71, 0.1)
            for head in (-45, -30, -15, 0, 15, 30, 45) for lift in ((a.lift,) if a.lift is not None else (-0.5, -0.25, 0.0))]
    res = [r_ for r_ in pool.map(_ev, grid, chunksize=4) if r_]
    res.sort(key=lambda x: -x['score'])
    print('coarse: %d setups behind the table edge; best:' % len(res))
    for r_ in res[:5]: print('  ', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r_.items()})
    fgrid = [(b['fwd'] + dfw, b['right'] + drt, b['heading'] + dh, b['lift'] + dl) for b in res[:4] for dfw in (-0.05, 0, 0.05) for drt in (-0.05, 0, 0.05)
             for dh in (-7.5, 0, 7.5) for dl in ((0,) if a.lift is not None else (-0.1, 0, 0.1)) if -0.5 <= b['lift'] + dl <= 0]
    fine = [r_ for r_ in pool.map(_ev, fgrid, chunksize=2) if r_]
    fine.sort(key=lambda x: -x['score'])
    best = fine[0]
    cands = [c for c in fine if c['feasible']][:a.robust]
    seen_ = {(c['heading'], round(c['lift'], 2)) for c in cands}
    for c in res:                                   # the fine setups cluster at one optimum: add the best of other heading / lift combinations
        if a.robust and c['feasible'] and (c['heading'], round(c['lift'], 2)) not in seen_ and len(cands) < a.robust + 6:
            cands.append(c); seen_.add((c['heading'], round(c['lift'], 2)))
    if cands:
        for c, tl in zip(cands, pool.map(tolerance, cands)): c['tolerance'] = tl
        best = max(cands, key=lambda c: (c['tolerance'], c['score']))
        print('placement tolerance of the best setups:', [(round(c['fwd'], 2), round(c['right'], 2), c['heading'], round(c['lift'], 2), c['tolerance']) for c in cands], flush=True)
    print('chosen:', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in best.items()})
    pool.close()
    print('fine best:', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in best.items()})
    Tb = S.base_pose(best['fwd'], best['right'], best['heading'])
    rows, m = K.solve_path(Tb, best['lift'], [TW[i] for i in wrist], a.table_h, seeds=8, side=SIDE)
    if 'grasp' not in phase or phase.index('grasp') not in wrist: phase_grasp = wrist[len(wrist)//2]
    else: phase_grasp = phase.index('grasp')
    fc = [K.grip_clearance(TW[i], a.table_h, SIDE) for i in wrist]
    steps = [r['dq'] for r in rows[1:]]
    summ = dict(setup=dict(base_forward_m=best['fwd'], base_right_m=best['right'], heading_deg=best['heading'], lift_m=best['lift'],
                           table_h_m=a.table_h, table_edge_m=a.table_edge, base_front_m=S.footprint_front(Tb),
                           base_pose_world=Tb.tolist()),
                path=dict(frames=len(rows), reached_frac=float(np.mean([r['ok'] for r in rows])),
                          max_joint_step_rad=float(max(steps)), frames_over_max_step=int(sum(s > a.max_step for s in steps)),
                          joint_margin_min_rad=float(min(r['margin'] for r in rows)),
                          joint_margin_p10_rad=float(np.percentile([r['margin'] for r in rows], 10)),
                          arm_clearance_min_cm=100*float(min(r['arm_clear'] for r in rows)),
                          finger_clearance_min_cm=100*float(min(x[0] for x in fc)),
                          frames_fingers_below_table=int(sum(x[0] < -0.002 for x in fc))),
                search=dict(coarse=len(res), feasible_coarse=int(sum(r_['feasible'] for r_ in res)), fine=len(fine),
                            placement_tolerance=best.get('tolerance'), perturbations=PERTURB,
                            robust_candidates=[{k: c[k] for k in ('fwd', 'right', 'heading', 'lift', 'score', 'tolerance')} for c in cands]))
    json.dump(dict(summary=summ, q=[r['q'] for r in rows], frames=wrist), open(os.path.join(a.inputs, f'robot_setup{a.tag}.json'), 'w'), indent=1)
    print(json.dumps(summ['setup'] | summ['path'], indent=1))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pass
    from head2wrist.planning.headframe_cloud import headframe_cloud
    img, D, Pc, ok = headframe_cloud(a.inputs)
    Wp = (S.A2R @ (S.cam0[:3, :3] @ Pc[ok][::97].T + S.cam0[:3, 3:4])).T + S.shift
    tf = lambda p: np.stack([(p[..., :2] - S.origin[:2]) @ S.r[:2], (p[..., :2] - S.origin[:2]) @ S.f[:2]], -1)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    onT = np.abs(Wp[:, 2] - a.table_h) < 0.015
    hi_ = Wp[:, 2] > a.table_h + 0.015
    q2 = tf(Wp[onT])
    ax[0].scatter(q2[:, 0], q2[:, 1], s=0.3, c='tan', label='table seen by the head')
    q3 = tf(Wp[hi_ & (Wp[:, 2] < a.table_h + 0.25)])
    ax[0].scatter(q3[:, 0], q3[:, 1], s=0.3, c='tab:green', label='objects')
    ax[0].axhline(a.table_edge, c='k', ls='--', lw=1, label='assumed table edge')
    FP = (Tb[:2, :2] @ K.FOOTPRINT.T).T + Tb[:2, 3]
    FPt = tf(np.c_[FP, np.zeros(4)])
    ax[0].fill(FPt[:, 0], FPt[:, 1], alpha=0.25, c='tab:blue', label='robot base')
    tcp_t = tf(np.array([TW[i][:3, 3] for i in wrist]))
    ax[0].plot(tcp_t[:, 0], tcp_t[:, 1], c='tab:red', lw=1, label='gripper path')
    with h5py.File(os.path.join(S.root, S.meta['ep'] + '.hdf5')) as f:     # converted recordings have no spine / hip joints
        hum = {k: (S.A2R @ f['transforms/'+k][S.meta['t0'], :3, 3] + S.shift) for k in ('rightArm', 'leftArm', 'spine7', 'hip') if 'transforms/' + k in f}
    for k, p in hum.items(): qq = tf(p[None])[0]; ax[0].scatter(*qq, s=25, c='tab:purple', marker='x'); ax[0].annotate('human ' + k, qq, fontsize=7)
    dm = m.createData()
    chain = [f'arm_{SIDE[0]}_link{i}' for i in range(1, 8)] + ['tcp']
    for lab, i, c in (('first pose', wrist[0], 'tab:orange'), ('middle / grasp', phase_grasp, 'tab:red'), ('last pose', wrist[-1], 'tab:brown')):
        q = np.array(rows[wrist.index(i)]['q']); pin.forwardKinematics(m, dm, q); pin.updateFramePlacements(m, dm)
        pts = np.array([(Tb @ np.r_[dm.oMf[m.getFrameId(n)].translation, 1])[:3] for n in ['arm_base_link'] + chain])
        ax[0].plot(*tf(pts).T, '-o', ms=2, c=c, lw=1.5, label=f'arm at {lab}')
        side = np.stack([tf(pts)[:, 1], pts[:, 2]], 1); ax[1].plot(side[:, 0], side[:, 1], '-o', ms=2, c=c, lw=1.5, label=f'arm at {lab}')
    col = np.array([(Tb @ np.r_[0, 0, 0, 1])[:3], (Tb @ np.r_[dm.oMf[m.getFrameId('arm_base_link')].translation, 1])[:3]])
    ax[1].plot(tf(col)[:, 1], col[:, 2], c='tab:blue', lw=6, alpha=0.4, label='robot column (base to torso)')
    ax[1].plot([a.table_edge, 1.0], [a.table_h, a.table_h], c='tan', lw=4, label='table top')
    ax[1].axhline(0, c='gray', lw=1)
    for k, p in hum.items(): ax[1].scatter(tf(p[None])[0, 1], p[2], s=25, c='tab:purple', marker='x')
    ax[0].set_xlabel('right (m)')
    ax[0].set_ylabel('forward, away from the demonstrator (m)')
    ax[0].set_aspect('equal')
    ax[0].legend(fontsize=7, loc='upper left')
    ax[0].set_title('top view')
    ax[1].set_xlabel('forward (m)')
    ax[1].set_ylabel('height above the floor (m)')
    ax[1].set_aspect('equal')
    ax[1].legend(fontsize=7, loc='upper right')
    ax[1].set_title('side view')
    fig.tight_layout()
    fig.savefig(os.path.join(a.inputs, f'robot_setup{a.tag}.png'), dpi=110)
    print('saved', os.path.join(a.inputs, f'robot_setup{a.tag}.png'))


if __name__ == '__main__':
    main()
