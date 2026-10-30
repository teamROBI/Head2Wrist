from head2wrist.config import H2W_OUTPUT_ROOT
#!/usr/bin/env python3
"""
Which surface of each static object the head camera actually saw, so the proxy can show the REAL pixels there and fall
back to SAM 3D's colours only on the unseen side (SAM 3D's own texture put an orange blotch inside the bowl).
A mesh vertex is seen if it projects inside the object's SAM 3 mask (eroded 2 px) in the head frame and the ray from
the head camera reaches it before any object mesh (ray casting all static objects; within 3 mm); its colour is then the
head frame's pixel (explore first_frame: hands removed, objects present). The first version compared with DA3's depth
(within 1 cm) and missed most of the bowl: its inside and its front (DA3 depth inside a bowl is too soft).
Writes <task_inputs>/proxy/obj<k>_seen.npz: seen (V,) float 0/1, uv (V,2) head-image coords in [0,1] (u right, v down),
rgb (V,3) uint8 (the head pixel; 0 where unseen), fill (V,3) uint8 (the head pixel where seen, the NEAREST SEEN vertex's
head pixel where unseen, median of the nearest seen vertices (well inside the outline, normals agreeing; else the object's median): SAM 3D's own texture on the unseen side put an orange blotch in the bowl that T2 turned into a
stick; objects the head saw nowhere keep SAM 3D's colours). Prints the seen fraction per object.
"""
import argparse, json
import numpy as np, cv2, open3d as o3d, trimesh
from scipy.spatial import cKDTree





def main():
    global F0, H0, I, Nv, PX, T, V, W0, _, _os, _sys, a, ap, cand, cfg, core, dirs, dist, fill, fv, headcam, img, ins, m, med, name, nn, ob, ok_, rgb, sc, seen, th, u, ui, v, vi, z
    ap = argparse.ArgumentParser()
    ap.add_argument('--ep', required=True)
    a = ap.parse_args()
    name = a.ep.replace('/', '_')
    I = f'{H2W_OUTPUT_ROOT}/lyra2/explore_inputs/{name}'
    T = f'{H2W_OUTPUT_ROOT}/lyra2/task_inputs/{name}'
    PX = f'{T}/proxy'
    import os as _os, sys as _sys
    pass
    import head2wrist.headcam as headcam
    F0 = headcam.focal(T)
    cfg = json.load(open(f'{PX}/proxy.json'))
    img = cv2.cvtColor(cv2.imread(f'{I}/first_frame.png'), cv2.COLOR_BGR2RGB)
    H0, W0 = img.shape[:2]
    sc = o3d.t.geometry.RaycastingScene()
    for ob in cfg['objects']:
        z = np.load(f"{PX}/obj{ob['k']}.npz"); sc.add_triangles(o3d.core.Tensor(z['V'].astype(np.float32)), o3d.core.Tensor(z['F'].astype(np.uint32)))
    for ob in cfg['objects']:
        z = np.load(f"{PX}/obj{ob['k']}.npz"); V = z['V'].astype(np.float64)
        m = cv2.erode(np.load(f"{PX}/{ob['mask']}").astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        u = F0*V[:, 0]/np.maximum(V[:, 2], 1e-4) + W0/2; v = F0*V[:, 1]/np.maximum(V[:, 2], 1e-4) + H0/2
        ins = (V[:, 2] > 0.05) & (u >= 0) & (u < W0 - 1) & (v >= 0) & (v < H0 - 1)
        ui, vi = np.clip(u.astype(int), 0, W0 - 1), np.clip(v.astype(int), 0, H0 - 1)
        dist = np.linalg.norm(V, axis=1); dirs = V/dist[:, None]                               # rays from the head camera (origin)
        th = sc.cast_rays(o3d.core.Tensor(np.concatenate([np.zeros_like(dirs), dirs], 1).astype(np.float32)))['t_hit'].numpy()
        seen = ins & m[vi, ui] & (th > dist - 0.003)
        rgb = np.zeros((len(V), 3), np.uint8); rgb[seen] = img[vi[seen], ui[seen]]
        fill = z['C'][:, :3].astype(np.uint8).copy()
        core = seen & (cv2.erode(m.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)[vi, ui]   # well inside the outline: no table-mixed edge pixels
        if core.sum() > 40:
            # same-facing neighbours only: the bowl's unseen inside is 5 mm from its seen outside (across the wall), whose
            # bottom band caught table colour from slight misregistration (brown fill); fallback = the object's median colour
            Nv = trimesh.Trimesh(V, z['F'], process=False).vertex_normals; med = np.median(rgb[core], 0)
            _, nn = cKDTree(V[core]).query(V[~seen], k=32); fill[seen] = rgb[seen]
            ok_ = (Nv[core][nn]*Nv[~seen][:, None]).sum(-1) > 0.5
            cand = rgb[core][nn].astype(np.float32); cand[~ok_] = np.nan
            fv = np.nanmedian(cand, 1); fv[np.isnan(fv).any(1)] = med
            fill[~seen] = fv.astype(np.uint8)
        np.savez(f"{PX}/obj{ob['k']}_seen.npz", seen=seen.astype(np.float32), uv=np.stack([u/W0, v/H0], 1).astype(np.float32), rgb=rgb, fill=fill)
        print(f"obj{ob['k']}: {100*seen.mean():.0f}% of vertices seen by the head camera")


if __name__ == '__main__':
    main()
