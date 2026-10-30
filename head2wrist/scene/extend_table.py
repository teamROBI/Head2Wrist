from head2wrist.config import H2W_OUTPUT_ROOT
#!/usr/bin/env python3
"""
Extend the known table past the edge of the head camera's view, for GEN3C's dense conditioning.
Where the head image ends, the known area of the wrist-view conditioning ends in a straight line that GEN3C draws as a
crease / table edge (exp-r6-gen3c-task-trajectory, grasp frames). The table plane is known, so the table is continued
in its own metric, top-down coordinates: a texture map of the table (--res m/px) is filled from the head frame where the
head saw it, the rest is outpainted with LaMa, and the map pixels the head never saw (outside its image) become extra
table points at the plane depth, from --near (metres forward of the point under the head; the real near edge of the
table is unknown, so the extension is kept short) to the seen area.
Writes <inputs>/table_extension.npz (points [N,3] first-camera frame, colors [N,3] uint8, res) and a preview map.
"""
import argparse, os, sys
import numpy as np, cv2

from head2wrist.planning.make_hover_tour import fit_plane
from head2wrist.planning.headframe_cloud import headframe_cloud, F0





def main():
    global D, F0, F_, H0, Image, P, R_, S, SimpleLama, W0, X, _, _os, _sys, a, ap, base, d, den, detail, ext, f, filled, fl, gf, gr, headcam, hi_r, hole_lo, img, inimg, k, lo, lo_r, low, n, num, o0, ok, ov, prev, r, seen, sz, tex, u, ui, v, vi, vis, z
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default=f'{H2W_OUTPUT_ROOT}/lyra2/task_inputs/basic_pick_place_1')
    ap.add_argument('--res', type=float, default=0.001)
    ap.add_argument('--near', type=float, default=-0.04, help='how far toward the demonstrator the table is assumed to go (m)')
    ap.add_argument('--far', type=float, default=0.40)
    ap.add_argument('--overlap', type=float, default=0.03, help='metres the extension reaches into the seen area')
    a = ap.parse_args()
    import os as _os, sys as _sys
    pass
    import head2wrist.headcam as headcam
    F0 = headcam.focal(a.inputs)
    img, D, P, ok = headframe_cloud(a.inputs)
    H0, W0 = img.shape[:2]
    low = ok.copy()
    low[:int(0.45*H0)] = False
    n, d, _ = fit_plane(P[low][::40])
    n, d = (-n, -d) if n[1] > 0 else (n, d)
    o0 = -d*n
    f = np.array([0, 0, 1.]) - n[2]*n
    f /= np.linalg.norm(f)
    r = np.cross(f, n)
    r = -r if r[0] < 0 else r
    seen = P[ok & (np.abs(P@n + d) < 0.02)]
    S = np.stack([(seen - o0)@r, (seen - o0)@f], 1)
    lo_r, hi_r = np.percentile(S[:, 0], [1, 99])
    R_ = np.arange(lo_r - 0.05, hi_r + 0.05, a.res)
    F_ = np.arange(a.near, a.far, a.res)
    gr, gf = np.meshgrid(R_, F_[::-1])
    X = o0 + gr[..., None]*r + gf[..., None]*f
    z = X[..., 2]
    u = F0*X[..., 0]/np.maximum(z, 1e-3) + W0/2
    v = F0*X[..., 1]/np.maximum(z, 1e-3) + H0/2
    inimg = (z > 0.05) & (u >= 0) & (u < W0 - 1) & (v >= 0) & (v < H0 - 1)
    ui, vi = np.clip(u.astype(int), 0, W0-1), np.clip(v.astype(int), 0, H0-1)
    vis = inimg & (np.abs(D[vi, ui] - z) < 0.03*z)
    tex = np.zeros(X.shape, np.uint8)
    tex[vis] = img[vi[vis], ui[vis]]
    from simple_lama_inpainting import SimpleLama
    from PIL import Image
    filled = np.array(SimpleLama()(Image.fromarray(tex), Image.fromarray((~vis).astype(np.uint8)*255)))[:tex.shape[0], :tex.shape[1]]
    if vis.sum() > 500 and (~vis).any():
        k = max(1, int(round(0.01/a.res))); sz = (max(1, tex.shape[1]//k), max(1, tex.shape[0]//k))
        num = cv2.resize(tex.astype(np.float32)*vis[..., None], sz, interpolation=cv2.INTER_AREA); den = cv2.resize(vis.astype(np.float32), sz, interpolation=cv2.INTER_AREA)
        lo = np.clip(num/np.maximum(den[..., None], 1e-6), 0, 255).astype(np.uint8); hole_lo = (den < 0.5).astype(np.uint8)
        lo = cv2.inpaint(lo, hole_lo, 5, cv2.INPAINT_TELEA)
        base = cv2.resize(lo, (tex.shape[1], tex.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        fl = filled.astype(np.float32); detail = fl - cv2.GaussianBlur(fl, (0, 0), 0.015/a.res)
        filled = np.where(vis[..., None], filled, np.clip(base + detail, 0, 255)).astype(np.uint8)
    ov = max(1, int(round(a.overlap/a.res)))
    ext = cv2.dilate((~inimg).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*ov+1, 2*ov+1))) > 0
    np.savez(os.path.join(a.inputs, 'table_extension.npz'), points=X[ext].astype(np.float32), colors=filled[ext],
             res=a.res, near=a.near)
    prev = filled.copy()
    prev[ext] = (0.6*prev[ext] + 0.4*np.array([255, 0, 255])).astype(np.uint8)
    cv2.imwrite(os.path.join(a.inputs, 'table_extension_map.jpg'), cv2.cvtColor(np.concatenate([tex, prev], 0), cv2.COLOR_RGB2BGR))
    print('map %dx%d at %.1f mm/px; head-seen %.0f%%; extension points %d (forward %.2f m to the seen edge)'
          % (tex.shape[1], tex.shape[0], 1000*a.res, 100*vis.mean(), int(ext.sum()), a.near))


if __name__ == '__main__':
    main()
