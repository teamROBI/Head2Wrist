from head2wrist.config import H2W_DATA_ROOT
#!/usr/bin/env python3
"""
Real colours on the RH-P12-RN mesh from the FFW-SG2 wrist recordings, so the gripper can be drawn at ANY opening.
  mesh     the five gripper links (URDF), subdivided to <= --edge (0.8 mm ~ 2 px at 10 cm in the 424x240 stream)
  samples  every chosen frame, at its gripper state g: ray-cast the gripper at g; a vertex takes the frame's colour
           where it projects if it is the surface the ray hits there (same link, depth within 1.5 mm), faces the
           camera, and the render has that link on the whole 5x5 neighbourhood (no background at the silhouette);
           coloured samples (HSV S > 70 at V > 50) are dropped: the gripper is black / grey metal, colour = an object
           in front of it. First --k samples per vertex over frames in random order; colour = per-vertex median. A
           held object covers a given vertex only in some frames and openings, so the median drops it (a per-opening
           median turned the inner finger faces red from a Pringles tube held in most frames at 0.6 rad)
  speed    only vertices visible at some opening (0.025 rad grid) are sampled; frames are sampled in the workers
  arms     both arms (the grippers render identically in their own D405, mask IoU 1.000)
  mapping  per episode and wrist video: which state channel (own / other arm) the fingers follow, by the dark-pixel
           IoU of the rendered fingers at 12 frames; the wrist streams are swapped in some datasets (pick_item,
           fold_towel, ...); episodes where neither channel fits are skipped
Writes $H2W_DATA_ROOT/assets/gripper_texture.npz (+ gripper_texture_views.jpg).
"""
import argparse, glob, os, sys
import numpy as np, cv2, pandas as pd, trimesh
from multiprocessing import Pool
import head2wrist.rendering.gripper_render as GR
from head2wrist.wrist_pose import T_TCP_D405

ROOT = f'{H2W_DATA_ROOT}/robot_recordings'
CH = {'right': 15, 'left': 7}; CAM = {'right': 'cam_wrist_right', 'left': 'cam_wrist_left'}
W, H = 424, 240; KC = GR.intrinsics(W, H)
FINGER = (1, 2, 3, 4)                                     # link indices of the moving finger links


def vid(p, cam):
    d, rest = p.rsplit('/data/chunk-', 1); ch, ep = rest.split('/')
    return f'{d}/videos/chunk-{ch}/observation.images.{cam}/{ep.replace(".parquet", ".mp4")}'


def read_frames(path, ts):
    cap = cv2.VideoCapture(path); out = {}; t = 0; need = set(ts)
    while need:
        ok, im = cap.read()
        if not ok: break
        if t in need: out[t] = im; need.discard(t)
        t += 1
    return out


_RC = {}
def rendered(g):
    """cached ray-cast of the gripper at g (rounded to 0.005 rad): depth, link index"""
    k = round(float(g)/0.005)*0.005
    if k not in _RC:
        D, _, L = GR.render('right', k); _RC[k] = (D, L)
    return _RC[k]


def finger_iou(im, g):
    D, L = rendered(g); fm = np.isin(L, FINGER)
    dark = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) < 55; band = np.zeros_like(fm); band[:, :150] = True
    return (fm & dark).sum()/max(1, ((fm | dark) & band).sum())


def mapping(p, cam_side, s):
    """-> state channel (arm name) the video of cam_side follows, or None"""
    var = [side for side in CH if s[:, CH[side]].std() > 0.05]
    if not var: return None
    ts = np.linspace(0, len(s) - 1, 12).astype(int); fr = read_frames(vid(p, CAM[cam_side]), ts)
    sc = {side: np.mean([finger_iou(fr[t], s[t, CH[side]]) for t in fr]) for side in var}
    best = max(sc, key=sc.get); other = [v for k, v in sc.items() if k != best]
    if sc[best] < 0.25 or (other and sc[best] - other[0] < 0.05): return None
    return best


MESHES, OFF, CAND = None, None, None                     # set in main before the pool forks


_ER = {}
def eroded(g):
    k = round(float(g)/0.005)*0.005
    if k not in _ER:
        D, L = rendered(k); E = np.full(L.shape, -1)
        for li in range(len(GR.LINKS)):
            E[cv2.erode((L == li).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = li
        _ER[k] = E
    return _ER[k]


def sample(g, im):
    """(global vertex ids, BGR colours) of the candidate vertices this frame shows cleanly"""
    Tp = GR.link_poses_tcp('right', g); D, _ = rendered(g); E = eroded(g); Tc = np.linalg.inv(T_TCP_D405)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV); ids, cols = [], []
    for li, (vidx, V, Nn0) in enumerate(CAND):
        T = Tc @ Tp[GR.LINKS[li]]; P = V @ T[:3, :3].T + T[:3, 3]; Nn = Nn0 @ T[:3, :3].T
        u = KC[0, 0]*P[:, 0]/P[:, 2] + KC[0, 2] - 0.5; v = KC[1, 1]*P[:, 1]/P[:, 2] + KC[1, 2] - 0.5
        ui, vi = np.round(u).astype(int), np.round(v).astype(int)
        j = np.nonzero((ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & (P[:, 2] > 0.01))[0]
        ok = (E[vi[j], ui[j]] == li) & (np.abs(D[vi[j], ui[j]] - P[j, 2]) < 0.0015) &\
             ((Nn[j]*P[j]).sum(1)/np.linalg.norm(P[j], axis=1) < -0.2)
        j = j[ok]
        if not len(j): continue
        hs = hsv[vi[j], ui[j]]; j = j[~((hs[:, 1] > 70) & (hs[:, 2] > 50))]
        if not len(j): continue
        col = cv2.remap(im, u[j].astype(np.float32)[:, None], v[j].astype(np.float32)[:, None], cv2.INTER_LINEAR)[:, 0]
        ids.append(vidx[j]); cols.append(col)
    return (np.concatenate(ids), np.concatenate(cols)) if ids else (np.zeros(0, np.int64), np.zeros((0, 3), np.uint8))


def job(args):
    p, cam_side, n_frames, seed = args
    s = np.stack(pd.read_parquet(p, columns=['observation.state'])['observation.state'].values)
    ch = mapping(p, cam_side, s)
    if ch is None: return p, cam_side, None, None
    g = s[:, CH[ch]]; still = np.abs(np.gradient(g)) < 0.003
    idx = np.nonzero(still)[0]
    if not len(idx): return p, cam_side, ch, None
    rng = np.random.default_rng(seed); ts = np.sort(rng.choice(idx, min(n_frames, len(idx)), replace=False))
    fr = read_frames(vid(p, CAM[cam_side]), ts.tolist())
    return p, cam_side, ch, [sample(float(g[t]), fr[t]) for t in ts if t in fr]


def finalize(meshes, off, S, cnt, lo=15, hi=55, radius=0.0012):
    """per-vertex colour from its samples: mean of the samples between the lo-th and hi-th brightness percentile
    (dark-biased: background leaking in at 1 px model / state errors is brighter than the black gripper; a plain
    median left brown speckle on the finger linkage), then the median over mesh vertices within `radius`;
    vertices never seen take the nearest seen one on the same link"""
    from scipy.spatial import cKDTree
    NV = len(cnt); col = np.zeros((NV, 3), np.float32); lum = S.astype(np.float32) @ np.array([0.114, 0.587, 0.299])
    for n in range(1, S.shape[1] + 1):
        sel = np.nonzero(cnt == n)[0]
        if not len(sel): continue
        o = np.argsort(lum[sel, :n], 1); a_, b_ = int(np.floor(lo/100*n)), max(int(np.ceil(hi/100*n)), int(np.floor(lo/100*n)) + 1)
        pick = np.take_along_axis(S[sel, :n].astype(np.float32), o[:, a_:b_, None], 1); col[sel] = pick.mean(1)
    out = col.copy()
    for li, m in enumerate(meshes):
        r = np.arange(off[li], off[li+1]); seen = r[cnt[r] >= 3]
        if not len(seen): continue
        P = m.vertices[seen - off[li]]; tree = cKDTree(P)
        nb = tree.query_ball_point(P, radius)
        out[seen] = np.array([np.median(col[seen[j]], 0) for j in nb])
        unseen = r[cnt[r] < 3]
        if len(unseen): _, nn = tree.query(m.vertices[unseen - off[li]]); out[unseen] = out[seen[nn]]
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{H2W_DATA_ROOT}/assets/gripper_texture.npz')
    ap.add_argument('--edge', type=float, default=0.0008)
    ap.add_argument('--k', type=int, default=64)
    ap.add_argument('--frames_per_video', type=int, default=40)
    ap.add_argument('--workers', type=int, default=48)
    ap.add_argument('--from_samples', action='store_true', help='re-finalize from the saved samples (no video pass)')
    ap.add_argument('--radius', type=float, default=0.0012)
    ap.add_argument('--lo', type=float, default=15); ap.add_argument('--hi', type=float, default=55)
    a = ap.parse_args()
    # ---- mesh (right-arm link names; both arms identical in their own camera) ----
    meshes = []
    for l in GR.LINKS:
        m = GR._mesh(f'gripper_r_{l}')
        v, f = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, max_edge=a.edge)
        meshes.append(trimesh.Trimesh(v, f, process=False))
    off = np.cumsum([0] + [len(m.vertices) for m in meshes]); NV = off[-1]
    print('vertices per link', [len(m.vertices) for m in meshes], 'total', NV)
    if a.from_samples:
        z = np.load(a.out.replace('.npz', '_samples.npz')); S = np.zeros((NV, z['S'].shape[1], 3), np.uint8); cnt = np.zeros(NV, np.int32)
        S[z['vid']] = z['S']; cnt[z['vid']] = z['cnt']
        col = finalize(meshes, off, S, cnt, a.lo, a.hi, a.radius); save(a.out, meshes, off, col, cnt); return
    # candidate vertices: visible at some opening on a 0.025 rad grid
    Tc = np.linalg.inv(T_TCP_D405); seen = [np.zeros(len(m.vertices), bool) for m in meshes]
    for g0 in np.arange(0, GR.G_MAX + 1e-6, 0.025):
        Tp = GR.link_poses_tcp('right', g0); D, L = rendered(g0)
        for li, m in enumerate(meshes):
            T = Tc @ Tp[GR.LINKS[li]]; P = m.vertices @ T[:3, :3].T + T[:3, 3]
            u = np.round(KC[0, 0]*P[:, 0]/P[:, 2] + KC[0, 2] - 0.5).astype(int); v = np.round(KC[1, 1]*P[:, 1]/P[:, 2] + KC[1, 2] - 0.5).astype(int)
            j = np.nonzero((u >= 0) & (u < W) & (v >= 0) & (v < H) & (P[:, 2] > 0.01))[0]
            seen[li][j[(L[v[j], u[j]] == li) & (np.abs(D[v[j], u[j]] - P[j, 2]) < 0.0015)]] = True
    global CAND
    CAND = [(off[li] + np.nonzero(seen[li])[0], meshes[li].vertices[seen[li]], meshes[li].vertex_normals[seen[li]]) for li in range(len(meshes))]
    print('candidate (ever visible) vertices', sum(len(c[0]) for c in CAND))
    S = np.zeros((NV, a.k, 3), np.uint8); cnt = np.zeros(NV, np.int32)
    eps = [p for d in sorted(glob.glob(f'{ROOT}/ffw_sg2_rev1_*')) for p in sorted(glob.glob(d + '/data/chunk-*/*.parquet'))]
    rng = np.random.default_rng(0); jobs = [(p, c, a.frames_per_video, int(rng.integers(1 << 30))) for p in eps for c in CH]
    rng.shuffle(jobs); stats = {'own': 0, 'swapped': 0, 'skipped': 0}; used = 0
    with Pool(a.workers) as pool:
        for p, cam_side, ch, frames in pool.imap_unordered(job, jobs, chunksize=2):
            if ch is None or not frames: stats['skipped'] += 1; continue
            stats['own' if ch == cam_side else 'swapped'] += 1
            for gv, col in frames:
                free = cnt[gv] < a.k; gv, col = gv[free], col[free]
                S[gv, cnt[gv]] = col; cnt[gv] += 1
            used += len(frames)
    cv_ = np.concatenate([c[0] for c in CAND])
    print('videos', stats, 'frames used', used, '| candidate vertices with >= 5 samples: %.1f%%, median samples %d' %
          (100*(cnt[cv_] >= 5).mean(), np.median(cnt[cv_])))
    np.savez_compressed(a.out.replace('.npz', '_samples.npz'), vid=cv_, S=S[cv_], cnt=cnt[cv_])      # for re-tuning
    col = finalize(meshes, off, S, cnt, a.lo, a.hi, a.radius); save(a.out, meshes, off, col, cnt)


def save(out, meshes, off, col, cnt):
    np.savez_compressed(out, **{f'v{li}': m.vertices.astype(np.float32) for li, m in enumerate(meshes)},
                        **{f'f{li}': m.faces.astype(np.int32) for li, m in enumerate(meshes)},
                        **{f'c{li}': col[off[li]:off[li+1]] for li in range(len(meshes))},
                        **{f'n{li}': cnt[off[li]:off[li+1]] for li in range(len(meshes))}, links=np.array(GR.LINKS))
    print('wrote', out)


if __name__ == '__main__':
    main()
