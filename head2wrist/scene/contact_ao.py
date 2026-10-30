from head2wrist.config import H2W_OUTPUT_ROOT
#!/usr/bin/env python3
"""
Contact shadows of the static objects on the table, as a top-down map (5 mm/px) over the same extent as the Blender
table plane: for each table point, the fraction of a cosine-weighted hemisphere (64 rays, up to 6 cm) blocked by the
OBJECTS ONLY (Cycles' AO node also counted the table plane crossing the seen mesh: patch shading). ao = 1 - blocked.
Writes <task_inputs>/proxy/contact_ao.png (8-bit, 255 = open) and contact_ao.json (extent lo/hi in the table frame,
res, e1/e2/origin). blender_proxy.py multiplies the seen mesh and the table plane by it.
"""
import argparse, json
import numpy as np, cv2, open3d as o3d




def main():
    global D, De, F0, Hm, P0, PX, Pd, T, TE, Wm, _os, _sys, a, ab, allp, ao, ap, cfg, d, dl, e1, e2, fx, fy, g, gx, gy, h, headcam, hi, img, k, lo, mn, mx, n, name, near, o, oab, ob, objs_ab, occ, ph, r_, rays, rng, sc, seen_tab, u1, u2, uu, vv, w, z
    ap = argparse.ArgumentParser()
    ap.add_argument('--ep', required=True)
    ap.add_argument('--res', type=float, default=0.005)
    ap.add_argument('--dist', type=float, default=0.06)
    ap.add_argument('--rays', type=int, default=64)
    a = ap.parse_args()
    name = a.ep.replace('/', '_')
    T = f'{H2W_OUTPUT_ROOT}/lyra2/task_inputs/{name}'
    PX = f'{T}/proxy'
    import os as _os, sys as _sys
    pass
    import head2wrist.headcam as headcam
    F0 = headcam.focal(T)
    cfg = json.load(open(f'{PX}/proxy.json'))
    n = np.array(cfg['table']['n'])
    d = float(cfg['table']['d'])
    e1 = np.cross(n, [1, 0, 0.])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    o = -d*n
    De = np.load(f'{PX}/empty_depth.npy')
    h, w = De.shape
    fx, fy = F0*w/1920, F0*h/1080
    vv, uu = np.mgrid[0:h, 0:w]
    Pd = np.stack([(uu - w/2)/fx*De, (vv - h/2)/fy*De, De], -1).reshape(-1, 3)
    TE = np.load(f'{T}/table_extension.npz')['points']
    seen_tab = Pd[np.abs(Pd @ n + d) < 0.01]
    allp = np.concatenate([TE, seen_tab])
    ab = np.stack([(allp - o) @ e1, (allp - o) @ e2], 1)
    lo, hi = ab.min(0) - 0.3, ab.max(0) + 0.3
    sc = o3d.t.geometry.RaycastingScene()
    objs_ab = []
    for ob in cfg['objects']:
        z = np.load(f"{PX}/obj{ob['k']}.npz"); sc.add_triangles(o3d.core.Tensor(z['V'].astype(np.float32)), o3d.core.Tensor(z['F'].astype(np.uint32)))
        objs_ab.append(np.stack([(z['V'] - o) @ e1, (z['V'] - o) @ e2], 1))
    Wm, Hm = (np.ceil((hi - lo)/a.res)).astype(int)
    gy, gx = np.mgrid[0:Hm, 0:Wm]
    g = np.stack([lo[0] + (gx.ravel() + 0.5)*a.res, lo[1] + (gy.ravel() + 0.5)*a.res], 1)
    near = np.zeros(len(g), bool)
    for oab in objs_ab:
        mn, mx = oab.min(0) - a.dist, oab.max(0) + a.dist; near |= np.all((g >= mn) & (g <= mx), 1)
    P0 = o + g[near, :1]*e1 + g[near, 1:]*e2 + 0.002*n
    rng = np.random.default_rng(0)
    u1, u2 = rng.random(a.rays), rng.random(a.rays)
    r_, ph = np.sqrt(u1), 2*np.pi*u2
    dl = np.stack([r_*np.cos(ph), r_*np.sin(ph), np.sqrt(1 - u1)], 1)
    D = dl @ np.stack([e1, e2, n])
    occ = np.zeros(len(P0))
    for k in range(a.rays):
        rays = np.concatenate([P0, np.broadcast_to(D[k], P0.shape)], 1).astype(np.float32)
        occ += (sc.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy() < a.dist)
    ao = np.ones(len(g))
    ao[near] = 1 - occ/a.rays
    img = (ao.reshape(Hm, Wm)*255).astype(np.uint8)[::-1]
    cv2.imwrite(f'{PX}/contact_ao.png', img)
    json.dump(dict(lo=lo.tolist(), hi=hi.tolist(), res=a.res, e1=e1.tolist(), e2=e2.tolist(), origin=o.tolist(), dist=a.dist), open(f'{PX}/contact_ao.json', 'w'))
    print(f'contact AO map {Wm}x{Hm}; cells near objects {near.sum()}; min ao {ao.min():.2f}')


if __name__ == '__main__':
    main()
