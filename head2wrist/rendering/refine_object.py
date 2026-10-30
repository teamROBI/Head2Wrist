from head2wrist.config import H2W_OUTPUT_ROOT
#!/usr/bin/env python3
"""
Refine the SAM 3D object placement (object_mesh.npz, head camera frame) against OUR metric depth and mask, and set it
on the table.
  1. translation-only ICP: the mesh surface the head camera sees (ray-cast) vs the object's depth points (mask eroded
     5 px against edge smear); the SAM 3D layout was ~7 mm off on basic_pick_place/74 (IoU 0.56 with the mask)
  2. drop / raise along the table normal so the lowest vertex touches the table plane (objects rest on the table)
  3. silhouette fit: slide on the table, turn about its normal, scale, to maximise the ray-cast silhouette's IoU
     with the SAM 3 mask (grid, then Nelder-Mead); bpp/74: IoU 0.64 -> 0.88
  4. rest on the LOCAL surface: median height of the depth in a ring around the mask (the global plane was 1-2 cm off
     under the block on bpp/74, and the scene surface cut through it)
  4b. height: flattened along the normal when the mesh is > 40% taller than the depth shows (the phone, bpp/1)
  5. colour: vertex colours scaled per channel so the mesh, shaded like compose_wrist (0.65 + 0.35 |N.up|), matches
     the object's pixels in the input frame (SAM 3D colours were ~25% too bright on bpp/74)
Reports the silhouette IoU with the mask before / after and the object's size in the table frame.
Writes <task_inputs>/object_mesh.npz (refined, the SAM 3D placement kept as object_mesh_sam3d.npz) + object_mesh_check.jpg.
"""
import argparse, os, sys, shutil
import numpy as np, cv2, open3d as o3d
from scipy.spatial import cKDTree
from head2wrist.planning.make_hover_tour import fit_plane



def visible(V):
    """mesh points the head camera sees: ray-cast through the mesh's own projected pixels (every 2nd px)"""
    sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
    uv = (V @ K0.T)[:, :2]/V[:, 2:3]; x0, y0 = np.floor(uv.min(0)).astype(int); x1, y1 = np.ceil(uv.max(0)).astype(int)
    yy, xx = np.mgrid[y0:y1:2, x0:x1:2]; dirs = np.stack([(xx - W0/2)/F0, (yy - H0/2)/F0, np.ones_like(xx, float)], -1).reshape(-1, 3)
    r = sc.cast_rays(o3d.core.Tensor(np.concatenate([np.zeros_like(dirs), dirs], 1).astype(np.float32)))
    t = r['t_hit'].numpy(); ok = np.isfinite(t); return dirs[ok]*t[ok, None]


def silhouette(V):
    uv = (V @ K0.T)[:, :2]/V[:, 2:3]; m = np.zeros((H0, W0), np.uint8)
    ok =(uv[:, 0] >= 0) & (uv[:, 0] < W0) & (uv[:, 1] >= 0) & (uv[:, 1] < H0)
    m[uv[ok, 1].astype(int), uv[ok, 0].astype(int)] = 1
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))


def pose(x):
    dx, dy, yaw, ls = x; c, s_ = np.cos(yaw), np.sin(yaw)
    Rz = c*np.eye(3) + s_*np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]]) + (1 - c)*np.outer(n, n)
    return Rz*np.exp(ls), c0 + dx*e1 + dy*e2 - (Rz*np.exp(ls)) @ c0


def sil_iou(x):
    A, b = pose(x); Ai = np.linalg.inv(A)                              # cast the rays into the object's frame
    o = np.broadcast_to(-Ai @ b, dir_.shape); dd = dir_ @ Ai.T
    r = sc_.cast_rays(o3d.core.Tensor(np.concatenate([o, dd], -1).reshape(-1, 6).astype(np.float32)))
    m = np.isfinite(r['t_hit'].numpy()).reshape(mk_.shape)
    return (m & mk_).sum()/max(1, (m | mk_).sum())


def main():
    global A, C0, D, F, F0, H0, I, K0, O, P, PRE, S, T, V, Vr, W0, _, _gid, _os, _sys, a, ap, b, best, bot, bx0, bx1, by0, by1, c, c0, chk, cnt, col2, d, dd, dir_, dr, dx, dy, e1, e2, f2, flatten, fp, fx, fy, gain, h, headcam, hgt, hit2, img, iou, iou0, iou1, it, j, keep, low, low2, ls, m_, mask, me, me2, me3, minimize, mk_, n, name, r2, rect, res, ring, sc2, sc_, shd, shift, step, sup, top_d, top_m, tree, uu, uv2, v, vv, w, x0, x1, xb, xs, xs2, xs_, xx_, y0, y1, yaw, ys, ys2, ys_, yy_, z
    ap = argparse.ArgumentParser()
    ap.add_argument('--ep', required=True)
    ap.add_argument('--mask', default=None, help='object mask .npy (default: <task_inputs>/object_mask.npy)')
    ap.add_argument('--prefix', default=None, help='mesh prefix: reads <prefix>_sam3d.npz, writes <prefix>.npz (default: <task_inputs>/object_mesh)')
    a = ap.parse_args()
    name = a.ep.replace('/', '_')
    I = f'{H2W_OUTPUT_ROOT}/lyra2/explore_inputs/{name}'
    T = f'{H2W_OUTPUT_ROOT}/lyra2/task_inputs/{name}'
    PRE = a.prefix or f'{T}/object_mesh'
    if not os.path.exists(f'{PRE}_sam3d.npz'): shutil.copy(f'{PRE}.npz', f'{PRE}_sam3d.npz')
    z = dict(np.load(f'{PRE}_sam3d.npz'))
    V, F = z['V'].astype(np.float64), z['F']
    img = cv2.cvtColor(cv2.imread(f'{I}/first_frame.png'), cv2.COLOR_BGR2RGB)
    H0, W0 = img.shape[:2]
    mask = np.load(a.mask or f'{T}/object_mask.npy').astype(np.uint8)
    import os as _os, sys as _sys
    pass
    import head2wrist.headcam as headcam
    F0 = headcam.focal(I)
    D = np.load(f'{I}/first_depth.npy').astype(np.float32)
    h, w = D.shape
    fx, fy = F0*w/1920, F0*h/1080
    vv, uu = np.mgrid[0:h, 0:w]
    P = np.stack([(uu - w/2)/fx*D, (vv - h/2)/fy*D, D], -1)
    n, d, _ = fit_plane(P[(vv > 0.45*h) & (D > 0.05)][::7])
    n, d = (-n, -d) if n[1] > 0 else (n, d)
    me = cv2.erode(cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST), np.ones((5, 5), np.uint8)) > 0
    O = P[me & (D > 0.05)]
    K0 = np.array([[F0, 0, W0/2], [0, F0, H0/2], [0, 0, 1]])
    iou = lambda m: float((m & mask).sum()/max(1, (m | mask).sum()))
    iou0 = iou(silhouette(V))
    shift = np.zeros(3)
    for it in range(8):
        S = visible(V + shift); tree = cKDTree(S); dd, j = tree.query(O)
        keep = dd < np.percentile(dd, 80); step = np.median(O[keep] - S[j[keep]], 0); shift += step
        if np.linalg.norm(step) < 2e-4: break
    Vr = V + shift
    low = float((Vr @ n + d).min())
    Vr = Vr - low*n
    e1 = np.cross(n, [1, 0, 0.])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    ys_, xs_ = np.nonzero(mask)
    bx0, bx1, by0, by1 = max(0, xs_.min() - 60), min(W0, xs_.max() + 60), max(0, ys_.min() - 60), min(H0, ys_.max() + 60)
    sc_ = o3d.t.geometry.RaycastingScene()
    _gid = sc_.add_triangles(o3d.core.Tensor(Vr.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
    yy_, xx_ = np.mgrid[by0:by1:2, bx0:bx1:2]
    dir_ = np.stack([(xx_ - W0/2)/F0, (yy_ - H0/2)/F0, np.ones_like(xx_, float)], -1)
    mk_ = mask[by0:by1:2, bx0:bx1:2] > 0
    c0 = Vr.mean(0)
    c0 = c0 - (c0 @ n + d)*n
    best = (sil_iou([0, 0, 0, 0]), [0, 0, 0, 0])
    for dx in np.arange(-0.02, 0.021, 0.005):
        for dy in np.arange(-0.02, 0.021, 0.005):
            for yaw in np.radians(np.arange(-30, 31, 10)):
                for ls in np.log([0.8, 0.9, 1.0, 1.1]):
                    v = sil_iou([dx, dy, yaw, ls])
                    if v > best[0]: best = (v, [dx, dy, yaw, ls])
    from scipy.optimize import minimize
    res = minimize(lambda x: -sil_iou(x), best[1], method='Nelder-Mead',
                   options=dict(xatol=1e-4, fatol=1e-4, maxiter=300))
    xb = res.x if -res.fun >= best[0] else np.array(best[1])
    A, b = pose(xb)
    Vr = Vr @ A.T + b
    low2 = float((Vr @ n + d).min())
    Vr = Vr - low2*n
    print(f'silhouette fit: slide {100*xb[0]:+.1f}/{100*xb[1]:+.1f} cm, turn {np.degrees(xb[2]):+.1f} deg, scale x{np.exp(xb[3]):.2f}, IoU {sil_iou([0,0,0,0]):.2f} -> {max(-res.fun, best[0]):.2f}')
    iou1 = iou(silhouette(Vr))
    ring = (cv2.dilate(mask, np.ones((61, 61), np.uint8)) > 0) & ~(cv2.dilate(mask, np.ones((21, 21), np.uint8)) > 0)
    ring = cv2.resize(ring.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    sup = float(np.clip(np.median((P[ring & (D > 0.05)] @ n) + d), -0.015, 0.015))
    bot = float(np.percentile(Vr @ n + d, 2))
    Vr = Vr + (sup - bot)*n
    me3 = cv2.erode(cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST), np.ones((3, 3), np.uint8)) > 0
    top_d = float(np.percentile((P[me3 & (D > 0.05)] @ n + d) - sup, 90))
    top_m = float(np.percentile((Vr @ n + d) - sup, 98))
    flatten = 1.0
    if top_m > 1.4*top_d > 0:
        flatten = top_d/top_m; Vr = Vr - (1 - flatten)*np.outer((Vr @ n + d) - sup, n)
    sc2 = o3d.t.geometry.RaycastingScene()
    sc2.add_triangles(o3d.core.Tensor(Vr.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
    me2 = cv2.erode(mask, np.ones((7, 7), np.uint8)) > 0
    ys2, xs2 = np.nonzero(me2)
    dr = np.stack([(xs2 - W0/2)/F0, (ys2 - H0/2)/F0, np.ones(len(xs2))], 1)
    r2 = sc2.cast_rays(o3d.core.Tensor(np.concatenate([np.zeros_like(dr), dr], 1).astype(np.float32)))
    hit2 = np.isfinite(r2['t_hit'].numpy())
    f2 = F[r2['primitive_ids'].numpy()[hit2]]
    uv2 = r2['primitive_uvs'].numpy()[hit2]
    C0 = z['C'].astype(np.float32)
    col2 = (1 - uv2[:, :1] - uv2[:, 1:])*C0[f2[:, 0]] + uv2[:, :1]*C0[f2[:, 1]] + uv2[:, 1:]*C0[f2[:, 2]]
    shd = 0.65 + 0.35*np.abs(r2['primitive_normals'].numpy()[hit2] @ n)
    gain = np.clip(img[me2][hit2].astype(np.float32).mean(0)/(col2*shd[:, None]).mean(0), 0.6, 1.4)
    z['C'] = np.clip(C0*gain, 0, 255).astype(np.uint8)
    hgt = Vr @ n + d
    fp = np.stack([Vr @ e1, Vr @ e2], 1)
    rect = cv2.minAreaRect(fp.astype(np.float32))
    z.update(V=Vr.astype(np.float32), icp_shift=shift, table_drop=low, silhouette_fit=xb, n=n, d=d, support=sup, colour_gain=gain, flatten=flatten)
    np.savez(f'{PRE}.npz', **z)
    chk = img.copy()
    for m_, c in ((mask, (0, 255, 0)), (silhouette(V), (255, 0, 255)), (silhouette(Vr), (0, 128, 255))):
        cnt, _ = cv2.findContours(m_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(chk, cnt, -1, c, 2)
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = max(0, xs.min()-80), xs.max()+80, max(0, ys.min()-80), ys.max()+80
    cv2.imwrite(f'{PRE}_check.jpg', cv2.cvtColor(cv2.resize(chk[y0:y1, x0:x1], None, fx=2, fy=2), cv2.COLOR_RGB2BGR))
    print(f'{a.ep}: ICP shift {np.round(100*shift, 2)} cm, table drop {100*low:+.2f} cm | mask IoU {iou0:.2f} -> {iou1:.2f} | '
          f'footprint {100*max(rect[1]):.1f} x {100*min(rect[1]):.1f} cm, height {100*(hgt.max() - sup):.1f} cm on a surface at {100*sup:+.1f} cm (flatten x{flatten:.2f}); colour gain {np.round(gain, 2)}')


if __name__ == '__main__':
    main()
