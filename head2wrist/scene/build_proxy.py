from head2wrist.config import H2W_OUTPUT_ROOT
#!/usr/bin/env python3
"""
Complete scene proxy for one demo (the input to the proxy-render A/B test): measured geometry where the head camera
saw the scene, complete geometry where it did not.
  static objects  the SAM 3 object masks of the input frame, on the table, duplicates merged (IoU > 0.5), minus the
                  grasped object (it moves; compose_wrist / render_proxy --dynamic place it); one mask file each for
                  sam3d_object.py + refine_object.py (complete SAM 3D meshes, back sides included)
  empty scene     the task input frame (hands and the grasped object already removed) with every static object
                  inpainted too (LaMa), and its depth with those regions set to the table plane: the background with
                  nothing on the table, so the plane / depth-mesh texture never carries an object's pixels
  planes          the table plane (fitted to the depth; extended past the head view by table_extension.npz) and up to
                  two vertical wall planes fitted to the depth above the table
Writes <task_inputs>/proxy/: proxy.json, empty_frame.png, empty_depth.npy, obj<k>_mask.npy, objects.jpg
"""
import argparse, json, os, sys
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
from head2wrist.planning.make_hover_tour import fit_plane





def main():
    global De, Dt, Dx, F0, H0, I, Image, OUT, Pe, Px, SimpleLama, T, W0, _, _os, _sys, a, ap, c, cand, col, d, empty, fr, frac, fx, fy, gm, grasped, h, he, headcam, hgt, hh, img_task, k, keep, m, ms, n, name, pts, rdir, rest, rng, sel, tcol, um, ums, uu, vis, vv, w, walls, wd, wn, xs, ys, zpl
    ap = argparse.ArgumentParser()
    ap.add_argument('--ep', required=True)
    a = ap.parse_args()
    name = a.ep.replace('/', '_')
    I = f'{H2W_OUTPUT_ROOT}/lyra2/explore_inputs/{name}'
    T = f'{H2W_OUTPUT_ROOT}/lyra2/task_inputs/{name}'
    import os as _os, sys as _sys
    pass
    import head2wrist.headcam as headcam
    F0 = headcam.focal(T)
    OUT = f'{T}/proxy'
    os.makedirs(OUT, exist_ok=True)
    img_task = cv2.cvtColor(cv2.imread(f'{T}/first_frame.png'), cv2.COLOR_BGR2RGB)
    H0, W0 = img_task.shape[:2]
    Dt = np.load(f'{T}/first_depth.npy').astype(np.float32)
    h, w = Dt.shape
    fx, fy = F0*w/1920, F0*h/1080
    Dx = np.load(f'{I}/first_depth.npy').astype(np.float32)
    vv, uu = np.mgrid[0:h, 0:w]
    Px = np.stack([(uu - w/2)/fx*Dx, (vv - h/2)/fy*Dx, Dx], -1)
    n, d, _ = fit_plane(Px[(vv > 0.45*h) & (Dx > 0.05)][::7])
    n, d = (-n, -d) if n[1] > 0 else (n, d)
    hgt = Px @ n + d
    grasped = np.load(f'{T}/object_mask.npy').astype(bool)
    cand = []
    for m in np.load(f'{I}/sam3_object_masks.npy'):
        m = m.astype(bool); fr = m.mean()
        if not (0 < fr < 0.15) or m.sum() < 400: continue
        ms = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        hh = hgt[ms & (Dx > 0.05)]
        if len(hh) < 20 or np.median(hh) > 0.25 or np.median(hh) < -0.02: continue           # on the table
        if (m & grasped).sum() > 0.3*min(m.sum(), grasped.sum()): continue                 # the grasped object
        cand.append(m)
    cand.sort(key=lambda m: -m.sum())
    keep = []
    for m in cand:
        if all((m & k).sum()/(m | k).sum() < 0.5 and (m & k).sum() < 0.8*m.sum() for k in keep): keep.append(m)
    for k, m in enumerate(keep): np.save(f'{OUT}/obj{k}_mask.npy', m)
    vis = img_task.copy()
    rng = np.random.default_rng(0)
    for k, m in enumerate(keep):
        c = rng.integers(60, 255, 3); vis[m] = (0.45*vis[m] + 0.55*c).astype(np.uint8)
        ys, xs = np.nonzero(m); cv2.putText(vis, str(k), (int(xs.mean()), int(ys.mean())), 0, 1.4, (255, 255, 255), 3)
    cv2.imwrite(f'{OUT}/objects.jpg', cv2.cvtColor(vis[::2, ::2], cv2.COLOR_RGB2BGR))
    um = np.zeros((H0, W0), np.uint8)
    for m in keep: um |= m.astype(np.uint8)
    um = cv2.dilate(um, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    from simple_lama_inpainting import SimpleLama
    from PIL import Image
    empty = np.array(SimpleLama()(Image.fromarray(img_task), Image.fromarray(um*255)))[:H0, :W0] if um.any() else img_task
    cv2.imwrite(f'{OUT}/empty_frame.png', cv2.cvtColor(empty, cv2.COLOR_RGB2BGR))
    ums = cv2.resize(um, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    rdir = np.stack([(uu - w/2)/fx, (vv - h/2)/fy, np.ones_like(uu, float)], -1)
    zpl = -d/np.minimum(rdir @ n, -1e-6)
    De = np.where(ums & (zpl > 0) & (zpl < 5), zpl, Dt).astype(np.float32)
    gm = cv2.resize(cv2.dilate(grasped.astype(np.uint8), np.ones((31, 31), np.uint8)), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    De = np.where(gm & (zpl > 0) & (zpl < 5), zpl, De).astype(np.float32)
    np.save(f'{OUT}/empty_depth.npy', De)
    Pe = np.stack([(uu - w/2)/fx*De, (vv - h/2)/fy*De, De], -1)
    he = Pe @ n + d
    rest = (De > 0.05) & (he > 0.20)
    walls = []
    for _ in range(2):
        pts = Pe[rest][::5]
        if len(pts) < 500: break
        wn, wd, frac = fit_plane(pts, tol=0.015)
        if abs(wn @ n) > 0.3: break
        if wd < 0: wn, wd = -wn, -wd                                        # camera (origin) on the positive side
        sel = rest & (np.abs(Pe @ wn + wd) < 0.03)
        col = np.median(empty[cv2.resize(sel.astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST) > 0], 0)
        walls.append(dict(n=wn.tolist(), d=float(wd), inliers=int(sel.sum()), colour=col.tolist()))
        rest &= ~sel
    tcol = np.median(empty[cv2.resize(((De > 0.05) & (np.abs(he) < 0.01)).astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST) > 0], 0)
    json.dump(dict(ep=a.ep, table=dict(n=n.tolist(), d=float(d), colour=tcol.tolist()), walls=walls,
                   objects=[dict(k=k, mask=f'obj{k}_mask.npy', area=int(m.sum())) for k, m in enumerate(keep)]),
              open(f'{OUT}/proxy.json', 'w'), indent=1)
    print(f'{a.ep}: {len(keep)} static objects, {len(walls)} walls; wrote {OUT}')


if __name__ == '__main__':
    main()
