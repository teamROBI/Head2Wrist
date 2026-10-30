from head2wrist.config import H2W_OUTPUT_ROOT
#!/usr/bin/env python3
"""
Strict re-scoring of the closed-loop evaluations (the original test counted an episode as a success
when the object was lifted >= 3 cm and its final XY was within 3 cm of the place point, so a policy still HOLDING the
object above the target passed; e.g. bpp/25 true-view moved-base episodes 6 and 18 end 7-9 cm up with the gripper closed).
From each saved rollout (eval_<i>.npz: per-frame object pose and gripper joint) an episode now succeeds only if
  lifted   the object rose >= 3 cm above its start
  released the gripper's last joint value is below the midpoint of the plan's open / close values
  placed   the object's last position is within 3 cm (XY) of the planned place point and within 2 cm of the height at which
           the scene's expert demos leave it after their release (the planned place point is the release height: a released
           object comes to rest 1-3.5 cm below it, on the target or on the other object)
The last recorded frame is used (the 1 s settle after it is not saved); a released object has settled by then within
these tolerances in the rollouts checked. Also: Wilson 95% intervals, and for each TRUE / GENERATED pair on the same plan
set the discordant counts (true-only / generated-only successes) with an exact McNemar p-value.
  rescore_evals.py --scene basic_pick_place_25 --runs true:hyb_basic_pick_place_25_true,gen:hyb_basic_pick_place_25_gen
"""
import argparse, glob, json, math, os
import numpy as np



def wilson(k, n, z=1.96):
    if n == 0: return (float('nan'),) * 2
    p = k/n; c = (p + z*z/(2*n))/(1 + z*z/n); h = z*math.sqrt(p*(1 - p)/n + z*z/(4*n*n))/(1 + z*z/n); return max(0, c - h), min(1, c + h)


def mcnemar(b, c):                      # exact two-sided binomial test on the discordant pairs
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); return min(1.0, 2*sum(math.comb(n, i) for i in range(k + 1))/2**n)


def rest_height():
    zs = []
    for dd in (a.demos or f'{D}/demos_sr').split(','):
        for f in sorted(glob.glob(f'{dd}/ep_*.npz'))[:80]:
            try: z = np.load(f); ok = bool(z['success']); OB = z['object']; S = z['state']
            except Exception: continue
            if ok: zs.append(float(OB[-1, 2]))
    assert zs, 'no expert demos for the rest height'; return float(np.median(zs)), float(np.percentile(zs, 5)), float(np.percentile(zs, 95)), len(zs)


def score(run, tag, plans):
    P = json.load(open(f'{D}/{plans}.json')); pl = {p['i']: p for p in P['plans']}; gmid = 0.5*(P['gripper_open'] + P['gripper_close'])
    ev = f'{D}/eval_{run}_{tag}' if os.path.isdir(f'{D}/eval_{run}_{tag}') else f'{D}/eval_{run}{tag}'
    out = {}
    for f in sorted(glob.glob(f'{ev}/eval_*.npz')):
        z = np.load(f); i = int(os.path.basename(f)[5:9]); p = pl[i]; OB = z['object']; S = z['state']; z0 = p['object_start'][2]
        lifted = OB[:, 2].max() - z0 >= 0.03; released = S[-1, 7] < gmid
        placed = np.linalg.norm(OB[-1, :2] - np.array(p['place_planned'][:2])) < a.xy_tol and abs(OB[-1, 2] - ZR[0]) < a.z_tol
        out[i] = dict(strict=bool(lifted and released and placed), loose=bool(z['success']), lifted=bool(lifted), released=bool(released), placed=bool(placed))
    return out, ev


def main():
    global A, B, D, R, ZR, _, a, ap, b, c, common, ev, held, hi, i, kl, ks, l1, l2, lab, lines, lo, n, o, plans, r, run, runs, s, sets, tag, txt, v
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True)
    ap.add_argument('--runs', required=True, help='label:run,...')
    ap.add_argument('--sets', default='in:plans_eval,ood:plans_eval_ood,pert:plans_eval_pert', help='tag:plans,... (stack_18 uses _in50 / _conf_ood ...)')
    ap.add_argument('--demos', default=None, help='expert demo dir(s) for the rest height (default: <scene>/demos_sr)')
    ap.add_argument('--z_tol', type=float, default=0.02)
    ap.add_argument('--xy_tol', type=float, default=0.03)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    D = f'{H2W_OUTPUT_ROOT}/sim_policy/{a.scene}'
    ZR = rest_height()
    runs = [r.split(':') for r in a.runs.split(',')]
    sets = [s.split(':') for s in a.sets.split(',')]
    lines = [f'{a.scene}: strict success (lifted, released, resting at the place point); expert rest height {ZR[0]:.3f} m (5-95% {ZR[1]:.3f}-{ZR[2]:.3f}, {ZR[3]} demos)']
    R = {}
    for tag, plans in sets:
        for lab, run in runs:
            o, ev = score(run, tag, plans); R[(lab, tag)] = o; n = len(o); ks = sum(v['strict'] for v in o.values()); kl = sum(v['loose'] for v in o.values())
            lo, hi = wilson(ks, n); held = sum(v['loose'] and not v['released'] for v in o.values())
            lines.append(f'  {lab:5s} {tag:5s} strict {ks:2d}/{n:2d} ({100*ks/max(n, 1):3.0f}%, 95% CI {100*lo:3.0f}-{100*hi:3.0f}%)   loose {kl:2d}/{n:2d}   loose-but-still-holding {held}   [{os.path.basename(ev)}]')
        if len(runs) >= 2:
            (l1, _), (l2, _) = runs[:2]; A, B = R[(l1, tag)], R[(l2, tag)]; common = sorted(set(A) & set(B))
            b = sum(A[i]['strict'] and not B[i]['strict'] for i in common); c = sum(B[i]['strict'] and not A[i]['strict'] for i in common)
            lines.append(f'        {tag:5s} paired on {len(common)} plans: {l1}-only {b}, {l2}-only {c}, McNemar p = {mcnemar(b, c):.2f}')
    txt = '\n'.join(lines)
    print(txt)
    open(a.out or f'{D}/results_strict.txt', 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
