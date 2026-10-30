from head2wrist.config import H2W_EGODEX_ROOT
#!/usr/bin/env python3
"""
The grasped object's SAM 3 mask in the input frame, picked from the human's pinch point at the grasp, before any robot
planning, so its SAM 3D mesh (graspobj stage) can place the robot grasp (plan_approach) and the release height
(plan_transport). Rule as plan_approach's footprint snap: the masks whose table-object points come within 8 cm of the
pinch point, the nearest within 1 cm of the closest, the smallest of those (an olive on a plate -> the olive).
Writes <inputs>/grasp_object_mask.npy (H0, W0 bool) and grasp_object_mask.jpg.
"""
import argparse, json, os, sys
import numpy as np, cv2, h5py
from head2wrist.planning.make_hover_tour import fit_plane
from head2wrist.pipeline.demo_events import detect_pick_place





def main():
    global D, F0, I, J, MAXD, Pc, SIDE, _, _os, _sys, a, a_j, ap, cam, cand, d, dist, ev, f, full, fx, fy, g_c, h, headcam, hgt, img, inv0, j, k, m, meta, n, near, ok, sm, t0, tg, up, uu, vv, w
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', required=True)
    ap.add_argument('--root', default=None, help='dataset root (default: meta.json, else EgoDex test)')
    ap.add_argument('--side', default=None)
    a = ap.parse_args()
    I = a.inputs
    meta = json.load(open(f'{I}/meta.json'))
    t0 = meta['t0']
    SIDE = a.side or meta.get('side', 'right')
    with h5py.File(os.path.join(a.root or meta.get('root', f'{H2W_EGODEX_ROOT}'), meta['ep'] + '.hdf5')) as f:
        cam = f['transforms/camera'][:].astype(float)
        J = {k: f['transforms/' + k][:, :3, 3].astype(float) for k in f['transforms'].keys() if k.startswith(SIDE)}
    ev = detect_pick_place(J, SIDE)
    assert ev is not None, f'no pick-and-place for the {SIDE} hand'
    tg = ev[0]
    inv0 = np.linalg.inv(cam[t0])
    g_c = inv0[:3, :3] @ ((J[f'{SIDE}ThumbTip'][tg] + J[f'{SIDE}IndexFingerTip'][tg])/2) + inv0[:3, 3]
    import os as _os, sys as _sys
    pass
    import head2wrist.headcam as headcam
    F0 = headcam.focal(I)
    D = np.load(f'{I}/first_depth.npy')
    h, w = D.shape
    fx, fy = F0*w/1920, F0*h/1080
    vv, uu = np.mgrid[0:h, 0:w]
    ok = D > 0.05
    Pc = np.stack([(uu - w/2)/fx*D, (vv - h/2)/fy*D, D], -1)
    n, d, _ = fit_plane(Pc[(vv > 0.45*h) & ok][::7])
    n, d = (-n, -d) if n[1] > 0 else (n, d)
    hgt = Pc @ n + d
    full = [m for m in np.load(f'{I}/sam3_object_masks.npy') if 0 < m.mean() < 0.15]
    sm = [cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0 for m in full]
    cand = [m & ok & (hgt > 0.003) & (hgt < 0.25) for m in sm]
    dist = [np.linalg.norm(Pc[m][::5] - g_c, axis=1).min() if m.sum() > 30 else 9.0 for m in cand]
    MAXD = float(os.environ.get('EGOH2W_GRASP_MAX_DIST', '0.08'))
    assert min(dist) < MAXD, f'no object within {100*MAXD:.0f} cm of the pinch point (closest {min(dist):.3f} m; retry with EGOH2W_GRASP_MAX_DIST)'
    near = [j for j in range(len(cand)) if dist[j] < min(dist) + 0.01]
    j = min(near, key=lambda j: cand[j].sum())
    while True:
        a_j = sm[j].sum(); up = [k for k in range(len(sm)) if k != j and a_j < sm[k].sum() < 4*a_j and (sm[k] & sm[j]).sum() > 0.7*a_j]
        if not up: break
        j = max(up, key=lambda k: sm[k].sum())
    np.save(f'{I}/grasp_object_mask.npy', full[j])
    img = cv2.imread(f'{I}/first_frame.png')
    img[full[j]] = (0.5*img[full[j]] + [0, 110, 0]).astype(np.uint8)
    cv2.imwrite(f'{I}/grasp_object_mask.jpg', img[::2, ::2])
    print(f'grasped object: mask {j} of {len(full)} ({100*full[j].mean():.2f}% of the frame), {100*dist[j]:.1f} cm from the pinch '
          f'point; {len(near)} candidate(s) within 1 cm of the closest')


if __name__ == '__main__':
    main()
