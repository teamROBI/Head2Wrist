from head2wrist.config import H2W_OUTPUT_ROOT
#!/usr/bin/env python3
"""
Render the complete scene proxy (build_proxy.py + one refined SAM 3D mesh per static object) along the task's camera
path, at GEN3C / Cosmos 720p (1280x704, the same per-axis-resized camera as every other video of the task).
  geometry  nearest hit of: the empty-scene depth mesh (what the head saw; triangles across depth jumps cut), the
            static object meshes (complete), and the table / wall planes (fill everything the head never saw)
  colour    background hits (depth mesh or planes): the empty head frame where the head camera saw that point
            (projected, depth-tested), else the outpainted table map (table_extension) on the table, else the
            plane's median colour; objects (--object_tex real): the head frame's pixel where the head saw that surface
            (object_visibility.py; counted as real, source 0), else their calibrated vertex colours x (0.65 + 0.35 |N.up|)
  outputs   in <task_inputs>/proxy/render/: rgb.mp4 (the proxy render = Cosmos-Transfer input / vis source),
            depth.mp4 (Transfer depth control: normalised inverse depth, near = bright), seg.mp4 (flat colour per
            surface), guided_mask.mp4 (white where the colour came from the real head frame), dense_proxy_renders.npz (GEN3C --dense_renders: RGB, mask = 1 everywhere geometry was hit),
            source.npy (per pixel: 0 head frame, 1 table map, 2 object mesh, 3 plane colour, 4 head frame but
            inpainted (a removed object's area), 255 nothing)
"""
import argparse, json, os, subprocess, sys
import numpy as np, cv2, open3d as o3d
from scipy.spatial import cKDTree



def _floor_rgb(px):                                                         # the floor, not the edge tape: dark and unsaturated pixels
    px = px.astype(np.float32); mx, mn = px.max(1), px.min(1); lm = px.mean(1)
    k_ = (lm < 85) & (mx - mn < 30)
    return np.median(px[k_], 0) if k_.sum() > 30 else np.array([58.0, 58.0, 60.0])


def quad(nrm, dd, centre, size):
    nrm = nrm/np.linalg.norm(nrm); c = centre - (centre @ nrm + dd)*nrm
    e1 = np.cross(nrm, [1, 0, 0.]) if abs(nrm[0]) < 0.9 else np.cross(nrm, [0, 1, 0.]); e1 /= np.linalg.norm(e1); e2 = np.cross(nrm, e1)
    Vq = np.array([c + s1*size*e1 + s2*size*e2 for s1, s2 in ((-1, -1), (1, -1), (1, 1), (-1, 1))], np.float32)
    return B.add_triangles(o3d.core.Tensor(Vq), o3d.core.Tensor(np.array([[0, 1, 2], [0, 2, 3]], np.uint32)))


def bilinear(img, u, v):
    """img (H, W, 3) sampled at float pixel coords (cv2.remap caps a map dimension at 32767 points)"""
    x0 = np.clip(np.floor(u).astype(int), 0, img.shape[1] - 2); y0 = np.clip(np.floor(v).astype(int), 0, img.shape[0] - 2)
    fx_, fy_ = np.clip(u - x0, 0, 1)[:, None], np.clip(v - y0, 0, 1)[:, None]
    return ((1 - fx_)*(1 - fy_)*img[y0, x0] + fx_*(1 - fy_)*img[y0, x0 + 1] + (1 - fx_)*fy_*img[y0 + 1, x0] + fx_*fy_*img[y0 + 1, x0 + 1])


def render(c2w, K):
    y, x = np.mgrid[0:a.H, 0:a.W]; dc = np.stack([(x + 0.5 - K[0, 2])/K[0, 0], (y + 0.5 - K[1, 2])/K[1, 1], np.ones_like(x, float)], -1)
    dw = (dc @ c2w[:3, :3].T).reshape(-1, 3); o = np.broadcast_to(c2w[:3, 3], dw.shape)
    ray = o3d.core.Tensor(np.concatenate([o, dw], 1).astype(np.float32))
    ra, rb = A.cast_rays(ray), B.cast_rays(ray)
    ta, tb = ra['t_hit'].numpy(), rb['t_hit'].numpy(); ga, gb = ra['geometry_ids'].numpy(), rb['geometry_ids'].numpy()
    useA = np.isfinite(ta) & ~(np.isfinite(tb) & (tb < ta - 0.03))            # a plane only wins if clearly nearer
    t = np.where(useA, ta, tb); hit = np.isfinite(t)
    X = o + dw*t[:, None]
    rgb = np.zeros((len(t), 3), np.float32); src = np.full(len(t), 255, np.uint8); seg = np.zeros((len(t), 3), np.uint8)
    rgb[~hit] = BGCOL                                                                 # nothing hit: the backdrop colour, not black
    is_obj = np.zeros(len(t), bool)
    # objects
    for (gid, F, C, seenv), sc in zip(objs, ocol):
        m = useA & (ga == gid)
        if not m.any(): continue
        f = F[ra['primitive_ids'].numpy()[m]]; uv = ra['primitive_uvs'].numpy()[m]
        shd = (0.85 + 0.15*np.abs(ra['primitive_normals'].numpy()[m] @ n)) if a.object_tex == 'real' else (0.65 + 0.35*np.abs(ra['primitive_normals'].numpy()[m] @ n))
        bw = np.concatenate([1 - uv[:, :1] - uv[:, 1:], uv], 1)
        rgb[m] = (bw[:, :1]*C[f[:, 0]] + bw[:, 1:2]*C[f[:, 1]] + bw[:, 2:]*C[f[:, 2]])*shd[:, None]
        src[m] = 2; seg[m] = sc; is_obj |= m
        sv = (bw*seenv[f]).sum(1) > 0.5                                                  # the head saw this surface: real pixels
        if sv.any():
            jm = np.nonzero(m)[0][sv]; Xo = X[jm]
            uo = F0*Xo[:, 0]/np.maximum(Xo[:, 2], 1e-4) + W0/2; vo = F0*Xo[:, 1]/np.maximum(Xo[:, 2], 1e-4) + H0/2
            rgb[jm] = bilinear(full, uo, vo); src[jm] = 0
    # background: depth mesh or planes
    bg = hit & ~is_obj                                                               # (not src != 2: seen object pixels are src 0)
    j = np.nonzero(bg)[0]; Xb = X[j]
    uh = F0*Xb[:, 0]/np.maximum(Xb[:, 2], 1e-4) + W0/2; vh = F0*Xb[:, 1]/np.maximum(Xb[:, 2], 1e-4) + H0/2
    inside = (Xb[:, 2] > 0.05) & (uh >= 0) & (uh < W0 - 1) & (vh >= 0) & (vh < H0 - 1)
    dh = np.full(len(j), np.inf); ii = np.nonzero(inside)[0]
    dh[ii] = De[np.clip((vh[ii]*h/H0).astype(int), 0, h - 1), np.clip((uh[ii]*w/W0).astype(int), 0, w - 1)]
    seen = inside & (np.abs(Xb[:, 2] - dh) < np.maximum(0.02, 0.03*Xb[:, 2]))
    rgb[j[seen]] = bilinear(empty, uh[seen], vh[seen]); src[j[seen]] = 0
    js = np.nonzero(seen)[0]; src[j[js[inp[vh[js].astype(int), uh[js].astype(int)]]]] = 4
    rest = j[~seen]
    if len(rest):
        on_table = np.abs(X[rest] @ n + d) < 0.01
        rt = rest[on_table]
        if len(rt):
            dd_, nn = te_tree.query(X[rt], distance_upper_bound=0.006); okk = np.isfinite(dd_)
            rgb[rt[okk]] = te_col[nn[okk]]; src[rt[okk]] = 1
            rgb[rt[~okk]] = cfg['table']['colour']; src[rt[~okk]] = 3
        for nm, gid, pn, pd, pc in planes[1:]:
            m = np.zeros(len(t), bool); m[rest[~on_table]] = True; m &= ~useA & (gb == gid)
            rgb[m] = pc; src[m] = 3
        left = rest[~on_table]; left = left[src[left] == 255]
        rgb[left] = cfg['table']['colour']; src[left] = 3                       # depth-mesh points off the table not seen (rare)
    # segmentation of the background
    # semantic, not by geometry source: depth-mesh pixels on the table plane / a wall plane take that surface's label
    mm = np.nonzero(useA & (ga == gid_mesh))[0]; seg[mm] = SEG['mesh']
    seg[mm[np.abs(X[mm] @ n + d) < 0.015]] = SEG['table']
    for k, (wn, wd, _) in enumerate(walls): seg[mm[np.abs(X[mm] @ wn + wd) < 0.03]] = SEG[f'wall{k}']
    for nm, gid, *_ in planes: seg[~useA & (gb == gid)] = SEG[nm]
    z = (dw @ c2w[:3, 2])*t                                                          # z-depth in the camera
    sh = (a.H, a.W)
    return rgb.reshape(sh + (3,)), np.where(hit, z, np.inf).reshape(sh), seg.reshape(sh + (3,)), src.reshape(sh)


def write(path, fr, rgb=True):
    tmp = path + '.tmp.mp4'; vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), a.fps, (a.W, a.H))
    for f in fr: vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR) if rgb else cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
    vw.release(); subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', tmp, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '14', path], check=True); os.remove(tmp)


def main():
    global A, B, BGCOL, C, C2W, DARK, DIVE, De, Dv, Dz, E2, E_pts, F, F0, FLOOR_RGB, G, H0, Kt, OUT, P, PX, Pd, R, S, SEG, SRC, T, TE, V, Vq, W0, _, _os, _sys, a, a_, a_e, ap, b_, b_e, beyond, c, cfg, col, corner, ctr, d, den, ds, dz, edge, empty, f, f_t, fcol, fl, flo, fr_src, frames, full, fx, fy, gid_mesh, h, headcam, hg, hi, hv_, i, idx, inp, inv, k, keep, lo, lum, n, name, nb, ob, objs, ocol, okd, planes, q, r, r0, r_t, ray, res, rgb, rng, rows, seenv, seg, sf, src, ss, sv_, tab, te_col, te_tree, tr, tri, u, u0, us, uu, uv_, v, vv, w, w_, walls, wc, wd, wn, x, z
    ap = argparse.ArgumentParser()
    ap.add_argument('--ep', required=True)
    ap.add_argument('--W', type=int, default=1280)
    ap.add_argument('--H', type=int, default=704)
    ap.add_argument('--fps', type=float, default=16.0)
    ap.add_argument('--frames', default='all', help="'all' or 'wrist' (from the end of the dive)")
    ap.add_argument('--out_name', default='render', help='output folder under <task_inputs>/proxy/')
    ap.add_argument('--object_tex', choices=['sam3d', 'real'], default='real', help='real: head-frame pixels on the object surface the head saw (object_visibility.py)')
    a = ap.parse_args()
    name = a.ep.replace('/', '_')
    T = f'{H2W_OUTPUT_ROOT}/lyra2/task_inputs/{name}'
    PX = f'{T}/proxy'
    OUT = f'{PX}/{a.out_name}'
    os.makedirs(OUT, exist_ok=True)
    import os as _os, sys as _sys
    pass
    import head2wrist.headcam as headcam
    F0 = headcam.focal(T)
    cfg = json.load(open(f'{PX}/proxy.json'))
    n = np.array(cfg['table']['n'])
    d = float(cfg['table']['d'])
    walls = [(np.array(w_['n']), float(w_['d']), np.array(w_['colour'])) for w_ in cfg['walls']]
    empty = cv2.cvtColor(cv2.imread(f'{PX}/empty_frame.png'), cv2.COLOR_BGR2RGB).astype(np.float32)
    H0, W0 = empty.shape[:2]
    full = cv2.cvtColor(cv2.imread(f"{H2W_OUTPUT_ROOT}/lyra2/explore_inputs/{name}/first_frame.png"), cv2.COLOR_BGR2RGB).astype(np.float32)
    De = np.load(f'{PX}/empty_depth.npy')
    h, w = De.shape
    fx, fy = F0*w/1920, F0*h/1080
    tr = np.load(f'{T}/trajectory.npz')
    Kt = tr['intrinsics'].astype(np.float64).copy()
    Kt[:, 0] *= a.W/float(tr['image_width'])
    Kt[:, 1] *= a.H/float(tr['image_height'])
    P = np.load(f'{T}/task_plan.npz')
    C2W = P['c2w']
    DIVE = int(P['dive_frames'])
    frames = range(len(C2W)) if a.frames == 'all' else range(DIVE, len(C2W))
    vv, uu = np.mgrid[0:h, 0:w]
    Pd = np.stack([(uu - w/2)/fx*De, (vv - h/2)/fy*De, De], -1).reshape(-1, 3)
    idx = np.arange(h*w).reshape(h, w)
    q = [idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]]
    tri = np.concatenate([np.stack([q[0], q[2], q[1]], -1).reshape(-1, 3), np.stack([q[1], q[2], q[3]], -1).reshape(-1, 3)])
    dz = De.reshape(-1)[tri]
    tri = tri[(dz.max(1)/np.maximum(dz.min(1), 1e-3) < 1.03) & (dz.min(1) > 0.05)]
    f_t = np.array([0, 0, 1.0]) - n[2]*n
    f_t /= np.linalg.norm(f_t)
    r_t = np.cross(n, f_t)
    G = Pd.reshape(h, w, 3)
    hg = G @ n + d
    okd = De > 0.05
    tab = okd & (np.abs(hg) < 0.012)
    flo = okd & (hg < -0.25)
    E_pts = []
    for c in range(0, w, 2):
        rows = np.nonzero(tab[:, c])[0]
        if not len(rows): continue
        r0 = rows.max()
        if r0 + 2 < h and flo[r0 + 1:min(h, r0 + 25), c].any(): E_pts.append(G[r0, c])
    if len(E_pts) < 15:
        lum = cv2.resize(cv2.cvtColor(empty.astype(np.uint8), cv2.COLOR_RGB2GRAY), (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
        E2 = []; DARK = []
        for c in range(0, w, 2):
            col = lum[:, c]
            if col[-3:].mean() > 85: continue                                   # no floor at the bottom of this column
            r = h - 1
            while r > h//3 and col[r] < 110: r -= 1                             # up through the dark run
            if r <= h//3 or h - 1 - r < 3 or col[max(0, r - 6):r + 1].mean() < 140: continue   # needs light table right above
            ray = np.array([(c - w/2)/fx, (r - h/2)/fy, 1.0]); den = ray @ n
            if abs(den) > 1e-6 and -d/den > 0: E2.append(ray*(-d/den))
            DARK.append(cv2.resize(empty.astype(np.uint8), (w, h), interpolation=cv2.INTER_AREA)[r + 2:, c].reshape(-1, 3))   # the floor strip's colours
        if len(E2) >= 15: E_pts = E2; FLOOR_RGB = _floor_rgb(np.concatenate(DARK)); print(f'table front edge by colour ({len(E2)} columns): the depth showed no floor', flush=True)
    FLOOR_RGB = globals().get('FLOOR_RGB')
    edge = None
    if len(E_pts) >= 15:
        E_pts = np.array(E_pts); us, ss = E_pts @ r_t, E_pts @ f_t
        keep = np.ones(len(us), bool)
        for _ in range(3):                                                      # robust line s = a + b u
            b_, a_ = np.polyfit(us[keep], ss[keep], 1); res = ss - (a_ + b_*us); keep = np.abs(res) < max(0.01, 2.5*np.std(res[keep]))
        edge = (a_ - 0.003, b_)                                                 # 3 mm toward the table: no sliver of floor on the top
        print(f'table front edge: {keep.sum()} of {len(us)} edge points, s = {a_:.3f} + {b_:.3f} u (m, table frame)', flush=True)
    if edge is not None:
        a_e, b_e = edge; sv_, uv_, hv_ = Pd @ f_t, Pd @ r_t, Pd @ n + d
        beyond = (sv_ < a_e + b_e*uv_ - 0.005) & (np.abs(hv_) < 0.05)
        nb = beyond[tri].any(1); tri = tri[~nb]; print(f'head mesh: {int(nb.sum())} triangles beyond the table edge dropped', flush=True)
    A = o3d.t.geometry.RaycastingScene()
    gid_mesh = A.add_triangles(o3d.core.Tensor(Pd.astype(np.float32)), o3d.core.Tensor(tri.astype(np.uint32)))
    objs = []
    for ob in cfg['objects']:
        f = f"{PX}/obj{ob['k']}.npz"
        if not os.path.exists(f): print('missing', f); continue
        z = np.load(f); V, F, C = z['V'].astype(np.float32), z['F'].astype(np.uint32), z['C'].astype(np.float32)
        sf = f"{PX}/obj{ob['k']}_seen.npz"; seenv = np.load(sf)['seen'] if (a.object_tex == 'real' and os.path.exists(sf)) else np.zeros(len(V), np.float32)
        if a.object_tex == 'real' and os.path.exists(sf) and 'fill' in np.load(sf).files: C = np.load(sf)['fill'].astype(np.float32)   # unseen side: nearest seen colour
        objs.append((A.add_triangles(o3d.core.Tensor(V), o3d.core.Tensor(F)), F, C, seenv))
    B = o3d.t.geometry.RaycastingScene()
    planes = []
    ctr = np.median(Pd[(Pd[:, 2] > 0.05)], 0)
    if edge is None:
        planes.append(('table', quad(n, d, ctr, 3.0), n, d, np.array(cfg['table']['colour'])))
    else:
        a_, b_ = edge; u0 = float(ctr @ r_t)
        corner = lambda u, s_: u*r_t + s_*f_t - d*n
        Vq = np.array([corner(u, a_ + b_*u + ds) for u, ds in ((u0 - 3, 0.0), (u0 + 3, 0.0), (u0 + 3, 6.0), (u0 - 3, 6.0))], np.float32)
        planes.append(('table', B.add_triangles(o3d.core.Tensor(Vq), o3d.core.Tensor(np.array([[0, 1, 2], [0, 2, 3]], np.uint32))), n, d,
                       np.array(cfg['table']['colour'])))
        fl = empty[cv2.resize(flo.astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST) > 0]
        fcol = FLOOR_RGB if FLOOR_RGB is not None else (np.median(fl, 0) if len(fl) > 50 else np.array([58.0, 58.0, 60.0]))   # the colour path's dark strip wins
        planes.append(('floor', quad(n, d + 0.73, ctr - 0.73*n, 4.0), n, d + 0.73, fcol))   # 0.73 m below the table top
        print(f'floor plane 0.73 m below the table, colour {np.round(fcol).astype(int).tolist()} ({len(fl)} floor pixels seen)', flush=True)
    for k, (wn, wd, wc) in enumerate(walls): planes.append((f'wall{k}', quad(wn, wd, ctr, 3.0), wn, wd, wc))
    inp = cv2.dilate(np.load(f'{T}/object_mask.npy').astype(np.uint8), np.ones((31, 31), np.uint8))
    for ob in cfg['objects']: inp |= cv2.dilate(np.load(f"{PX}/{ob['mask']}").astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    inp = inp > 0
    TE = np.load(f'{T}/table_extension.npz')
    te_tree = cKDTree(TE['points'])
    te_col = TE['colors'].astype(np.float32)
    BGCOL = np.median(empty[:max(8, H0//10)].reshape(-1, 3), 0).astype(np.float32)
    SEG = {'mesh': (200, 120, 60), 'table': (230, 150, 70), 'wall0': (90, 90, 200), 'wall1': (90, 200, 200), 'floor': (70, 70, 70)}
    rng = np.random.default_rng(3)
    ocol = [tuple(int(x) for x in rng.integers(40, 255, 3)) for _ in objs]
    R, Dz, S, SRC = [], [], [], []
    for i in frames:
        rgb, z, seg, src = render(C2W[i], Kt[i]); R.append(np.clip(rgb, 0, 255).astype(np.uint8)); Dz.append(z.astype(np.float32)); S.append(seg); SRC.append(src)
    R, Dz, S, SRC = np.stack(R), np.stack(Dz), np.stack(S), np.stack(SRC)
    inv = np.where(np.isfinite(Dz), 1/np.maximum(Dz, 1e-3), 0)
    lo, hi = np.percentile(inv[inv > 0], [1, 99])
    Dv = (np.clip((inv - lo)/(hi - lo), 0, 1)*255).astype(np.uint8)
    write(f'{OUT}/rgb.mp4', R)
    write(f'{OUT}/depth.mp4', Dv, rgb=False)
    write(f'{OUT}/seg.mp4', S)
    write(f'{OUT}/guided_mask.mp4', ((SRC == 0)*255).astype(np.uint8), rgb=False)
    np.savez(f'{OUT}/dense_proxy_renders.npz', img=R, mask=(SRC != 255).astype(np.uint8), known_frac=(SRC != 255).mean((1, 2)))
    np.save(f'{OUT}/source.npy', SRC)
    fr_src = {k: float((SRC == v).mean()) for k, v in (('head_frame', 0), ('table_map', 1), ('object_mesh', 2), ('plane_colour', 3), ('inpainted', 4), ('nothing', 255))}
    json.dump(dict(frames=list(frames)[:1] + [len(frames)], source_fraction=fr_src, inv_depth_range=[float(lo), float(hi)]), open(f'{OUT}/render.json', 'w'), indent=1)
    print(f'{a.ep}: {len(frames)} frames, pixel sources', {k: round(v, 3) for k, v in fr_src.items()})


if __name__ == '__main__':
    main()
