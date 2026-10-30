from head2wrist.config import H2W_EGODEX_ROOT, H2W_OUTPUT_ROOT
#!/usr/bin/env python3
"""
Pick EgoDex episodes that suit the FFW-SG2 pilot: one hand (left or right) grasps one object, carries it and releases
it on the table; the other hand stays still. Per episode and hand: first closure (pinch closure), release =
first frame after it where the pinch opens by >= 1.5 cm for 5 frames, carry distance of the pinch point, and the idle
hand's path length. Prints the best episodes per task and writes a json.
"""
import argparse, json, os, sys
import numpy as np, h5py
from head2wrist.pipeline.demo_events import detect_pick_place



def analyse(path):
    with h5py.File(path) as f:
        J = {k: f['transforms/'+k][:, :3, 3].astype(float) for k in f['transforms'].keys()
             if k.endswith(('ThumbTip', 'IndexFingerTip', 'Hand'))}
        conf = {s: (np.minimum(f[f'confidences/{s}IndexFingerTip'][:], f[f'confidences/{s}ThumbTip'][:])
                    if 'confidences' in f else np.ones(len(J[f'{s}Hand']))) for s in ('left', 'right')}
        desc = str(f.attrs.get('llm_description', f.attrs.get('description', '')))
        cam = f['transforms/camera'][:].astype(float)
    T = len(cam); res = {}
    for s, o in (('right', 'left'), ('left', 'right')):
        g = (J[f'{s}ThumbTip'] + J[f'{s}IndexFingerTip'])/2
        ev = detect_pick_place(J, s)
        if ev is None: continue
        tc, tr = ev
        carry = float(np.linalg.norm((g[tr] - g[tc])[[0, 2]]))                 # horizontal (ARKit y up)
        idle = float(np.linalg.norm(np.diff(J[f'{o}Hand'], axis=0), axis=1).sum())
        # a second pick after the release (more than one object moved)
        J2 = {k: v[tr + 10:] for k, v in J.items()}; tc2 = detect_pick_place(J2, s) if T - tr > 40 else None
        reach = float(np.linalg.norm(g[tc] - cam[tc][:3, 3]))
        res[s] = dict(t_grasp=int(tc), t_release=int(tr), carry_m=carry, idle_hand_path_m=idle, second_grasp=tc2 is not None,
                      grasp_dist_from_head_m=reach, frames=T)
    return desc, res


def main():
    global a, ap_, c, cands, d, desc, e, eps, f, ok, out, r, res, s, task
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--root', default=f'{H2W_EGODEX_ROOT}')
    ap_.add_argument('--tasks', default='stack,pick_place_food,basic_pick_place,insert_remove_tennis_ball')
    ap_.add_argument('--max_eps', type=int, default=60)
    ap_.add_argument('--out', default=f'{H2W_OUTPUT_ROOT}/pipeline/episode_candidates.json')
    a = ap_.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out = {}
    for task in a.tasks.split(','):
        d = os.path.join(a.root, task); eps = sorted([f[:-5] for f in os.listdir(d) if f.endswith('.hdf5')], key=lambda x: int(x) if x.isdigit() else x)[:a.max_eps]
        cands = []
        for e in eps:
            try: desc, res = analyse(os.path.join(d, e + '.hdf5'))
            except Exception as ex: continue
            for s, r in res.items():
                ok = 0.10 <= r['carry_m'] <= 0.45 and r['idle_hand_path_m'] < 0.20 and not r['second_grasp'] and r['t_grasp'] >= 20
                if ok: cands.append(dict(ep=f'{task}/{e}', side=s, desc=desc[:100], **r))
        cands.sort(key=lambda c: (c['idle_hand_path_m'], -c['carry_m']))
        out[task] = cands
        print(f'== {task}: {len(cands)} clean single-hand pick-place episodes (of {len(eps)} scanned)')
        for c in cands[:4]:
            print(f"   {c['ep']:28s} {c['side']:5s} grasp t={c['t_grasp']:3d} release t={c['t_release']:3d} carry {100*c['carry_m']:4.0f} cm  idle hand {100*c['idle_hand_path_m']:4.0f} cm  {c['desc']}")
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
