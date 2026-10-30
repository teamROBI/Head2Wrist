from head2wrist.config import H2W_DATA_ROOT, H2W_EGODEX_ROOT, H2W_OUTPUT_ROOT
#!/usr/bin/env python3
"""
Simple static-scene wrist-view tour for Lyra-2 (no contact timing, no hand trajectory).

  1. clearest head frame: among frames with above-median sharpness, minimise hand coverage + hand speed
     (slow hands = less motion blur and shadow halo)
  2. remove the hands/arms only: SAM 3 ('hand', 'arm') + rounded margin (default 151 px) for blur/shadows,
     or the EgoDex bone-capsule mask; LaMa inpainting; objects stay
  3. metric depth of that frame: DA3-Nested-1.1 conditioned on EgoDex poses, inpainted region filled with
     single-view DA3 of the inpainted image, scale-matched outside it
  4. hover tour: a virtual RH-P12-RN TCP ~HOVER m above the fitted table plane, approach tilted TILT deg from
     vertical toward the far side, jaw axis horizontal, sweeping the visible table; D405 at the FFW-SG2 URDF
     offset from the TCP (head2wrist/wrist_pose.py), camera side facing the head.
     frames 0..DIVE-1: head pose -> first hover pose (Lyra-2 must start at the input frame); DIVE..N-1: tour
"""
import argparse, json, os, sys
import numpy as np, cv2, h5py, torch

from head2wrist.vision.hand_mask import hand_mask
from head2wrist.video import read_frames
from head2wrist.wrist_pose import cam_from_tcp, tcp_poses
from head2wrist.geometry import interp_pose, K_at, CAPTIONS

N, DIVE, HOVER, TILT, D405_FOV = 161, 41, 0.10, 30.0, 87.0


def fit_plane(P, iters=400, tol=0.008, rng=np.random.default_rng(0)):
    best = (0, None)
    for _ in range(iters):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1]-s[0], s[2]-s[0]); nn = np.linalg.norm(n)
        if nn < 1e-9: continue
        n /= nn; d = -n@s[0]
        inl = np.abs(P@n + d) < tol
        if inl.sum() > best[0]: best = (inl.sum(), (n, d, inl))
    n, d, inl = best[1]
    c = P[inl].mean(0); u, s_, vt = np.linalg.svd(P[inl]-c); n = vt[2]; d = -n@c   # refine
    return n, d, inl.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=f'{H2W_EGODEX_ROOT}')
    ap.add_argument('--ep', default='basic_pick_place/1')
    ap.add_argument('--out', default=f'{H2W_OUTPUT_ROOT}/lyra2/hover_inputs')
    ap.add_argument('--da3', default=f'{H2W_DATA_ROOT}/checkpoints/da3/DA3NESTED-GIANT-LARGE-1.1')
    ap.add_argument('--mask', choices=['capsule', 'sam3', 'sam3fill'], default='sam3fill',
                    help='sam3fill: SAM 3 hand + its shadow, concave gaps filled, objects protected, 31 px (default); '
                         'sam3: SAM 3 hand + --sam_margin dilation (older inputs); capsule: EgoDex bone capsules')
    ap.add_argument('--sam_margin', type=int, default=151)
    ap.add_argument('--sam_gpu', action='store_true')
    ap.add_argument('--tag', default='')
    ap.add_argument('--path', choices=['hover', 'hand'], default='hover')
    ap.add_argument('--side', default=None, choices=['right', 'left'], help='demo hand (default: the episode\'s hand: tag)')
    ap.add_argument('--t_max', type=int, default=None, help='input frame must come before this frame (e.g. the grasp)')
    ap.add_argument('--n_frames', type=int, default=161, help='1 + 80k (Lyra-2 chunk size)')
    a = ap.parse_args()
    global N
    N = a.n_frames; assert (N-1) % 80 == 0
    out = os.path.join(a.out, a.ep.replace('/', '_') + (f'_{a.tag}' if a.tag else '')); os.makedirs(out, exist_ok=True)
    base = os.path.join(a.root, a.ep)
    with h5py.File(base+'.hdf5', 'r') as f:
        cam = f['transforms/camera'][:].astype(np.float64); K = f['camera/intrinsic'][:].astype(np.float64)
        capdev = f.attrs.get('capture_device', None)   # converted recordings (capture/egokit_to_egodex.py): their own focal
        J = {k: f['transforms/'+k][:, :3, 3].astype(np.float64) for k in f['transforms'].keys()
             if k.startswith('left') or k.startswith('right')}
        env = str(f.attrs.get('environment', '')); side = a.side or ('left' if 'hand:left' in env else 'right')
        handT = f[f'transforms/{side}Hand'][:].astype(np.float64)
        tips_y = np.concatenate([f['transforms/'+k][:, 1, 3] for k in f['transforms'].keys() if k.endswith('Tip')]).astype(np.float64)
        tips_demo = np.concatenate([f['transforms/'+k][:, 1, 3] for k in f['transforms'].keys() if k.endswith('Tip') and k.startswith(side)]).astype(np.float64)
    T = len(cam)
    allf = read_frames(base+'.mp4', set(range(T)))
    # ---- 1. clearest frame ----
    Jall = np.stack(list(J.values()))                           # (joints, T, 3)
    sp_ = np.linalg.norm(np.diff(Jall, axis=1), axis=2).mean(0)*30; speed = np.r_[sp_[0], sp_]   # mean hand-joint speed, m/s (frame 0 takes frame 1)
    cand = []
    for t in range(0, min(T, a.t_max) if a.t_max else T, 2):
        if t not in allf: continue
        g = cv2.cvtColor(allf[t][::4, ::4], cv2.COLOR_RGB2GRAY)
        sharp = cv2.Laplacian(g, cv2.CV_64F).var()
        mfrac = hand_mask(J, t, np.linalg.inv(cam[t]), K, (1080, 1920), dil=1.5).mean()
        cand.append((t, mfrac, sharp, speed[t]))
    sh_med = np.median([c[2] for c in cand]); mf_med = np.median([c[1] for c in cand]) + 1e-6
    sp_med = np.median([c[3] for c in cand]) + 1e-6
    # clearest = sharp, little hand in view, hands slow (less motion blur / shadow halo);
    # for --path hand, first require that the frame SEES the action: fraction of the jaw-centre path over the
    # whole episode that projects inside the view and is not under the hand at that frame
    def path_cov(t):
        E = np.linalg.inv(cam[t]); mk = hand_mask(J, t, E, K, (1080, 1920), dil=1.5); ok = 0
        g = (J[f'{side}ThumbTip'] + J[f'{side}IndexFingerTip'])/2
        for q in range(0, T, 2):
            c = E[:3, :3]@g[q] + E[:3, 3]
            if c[2] < 0.05: continue
            u_ = K[0, 0]*c[0]/c[2] + K[0, 2]; v_ = K[1, 1]*c[1]/c[2] + K[1, 2]
            ok += (60 <= u_ < 1860 and 60 <= v_ < 1020 and not mk[int(v_), int(u_)])
        return ok/len(range(0, T, 2))
    sharp_c = [c for c in cand if c[2] >= sh_med]
    if a.path == 'hand':
        cov = {c[0]: path_cov(c[0]) for c in sharp_c}; best = max(cov.values())
        sharp_c = [c for c in sharp_c if cov[c[0]] >= best - 0.02]
    t0, mfrac, sharp, spd = min(sharp_c, key=lambda c: c[1]/mf_med + c[3]/sp_med)
    path_coverage = path_cov(t0)
    img = allf[t0]; H0, W0 = img.shape[:2]
    # ---- 2. remove hands/arms ----
    if a.mask == 'sam3fill':
        pass
        from head2wrist.vision.sam3_hands import load, hand_removal_mask
        dev = 'cuda' if a.sam_gpu else 'cpu'
        proc, sam = load(dev); m, _ = hand_removal_mask(proc, sam, img, device=dev); m = m.astype(np.uint8); del sam
    elif a.mask == 'sam3':
        pass
        from head2wrist.vision.sam3_hands import load, sam3_mask
        dev = 'cuda' if a.sam_gpu else 'cpu'
        proc, sam = load(dev); m0, _ = sam3_mask(proc, sam, img, device=dev); del sam
        k = a.sam_margin
        m = cv2.dilate(m0.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        m = (cv2.GaussianBlur(m.astype(np.float32), (0, 0), k/4) > 0.25).astype(np.uint8)   # rounded outline
    else:
        m = hand_mask(J, t0, np.linalg.inv(cam[t0]), K, (H0, W0), dil=1.5).astype(np.uint8)
        m = cv2.dilate(m, np.ones((25, 25), np.uint8))
    from simple_lama_inpainting import SimpleLama
    from PIL import Image
    inp = np.array(SimpleLama()(Image.fromarray(img), Image.fromarray(m*255)))[:H0, :W0]
    cv2.imwrite(os.path.join(out, 'first_frame.png'), cv2.cvtColor(inp, cv2.COLOR_RGB2BGR))
    np.save(os.path.join(out, 'hand_removal_mask.npy'), m.astype(bool))                  # for seamless depth later
    cv2.imwrite(os.path.join(out, 'first_frame_orig_mask_inpainted.jpg'),
                cv2.cvtColor(np.concatenate([img, np.where(m[..., None] > 0, (255, 0, 255), img).astype(np.uint8), inp], 1)[::3, ::3], cv2.COLOR_RGB2BGR))
    # ---- 3. metric depth ----
    ctx = sorted(set(np.linspace(max(0, t0-45), min(T-1, t0+45), 12).astype(int).tolist()) | {t0})
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(a.da3).to('cuda').eval()
    with torch.no_grad():
        pr = model.inference([allf[i] for i in ctx], extrinsics=np.linalg.inv(cam[ctx]).astype(np.float32),
                             intrinsics=np.tile(K.astype(np.float32)[None], (len(ctx), 1, 1)),
                             align_to_input_ext_scale=True, process_res=756)
        pm = model.inference([inp], process_res=756)
    Dmv = pr.depth[ctx.index(t0)].astype(np.float32); Dmono = pm.depth[0].astype(np.float32)
    Kp = pr.intrinsics[ctx.index(t0)].astype(np.float64)
    h, w = Dmv.shape
    mk = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    ok = ~mk & (Dmv > 0.05) & (Dmono > 0.05)
    s = float(np.median(Dmv[ok]/Dmono[ok])); D = np.where(mk, Dmono*s, Dmv).astype(np.float32)
    # metric depth from the hands: the table is the level plane 6 mm (pad radius) below the lowest fingertips (0.5th
    # percentile over the episode, both hands; ARKit world, y up), so every table pixel has a known true depth. Fit
    # D' = a*D + b to it (robust least squares, 2 cm inlier band). The DA3 extrinsic-scale alignment needs head motion:
    # on basic_pick_place/74 (head moved 7 mm in 43 frames) the table came out 29% too far, a scale alone left the
    # block 5 cm too far; basic_pick_place/1: a = 1.03, b = -0.9 cm
    # The idle hand may leave the table: hanging off its edge or resting in the lap (EgoDex bpp/26: 48 cm below the demo
    # hand's lowest, bpp/7: 16 cm, bpp/6: 10 cm -> the table came out up to 2.4x too far). The demo hand touches the table
    # at the grasp, so when the idle hand goes > 5 cm lower, the demo hand alone sets the table (all development scenes:
    # <= 2.5 cm apart, unchanged)
    y_low = float(np.percentile(tips_y, 0.5)); y_demo = float(np.percentile(tips_demo, 0.5))
    if y_demo - y_low > 0.05:
        print(f'the idle hand goes {100*(y_demo - y_low):.0f} cm below the demo hand: table from the demo hand only', flush=True); y_low = y_demo
    y_tab = y_low - 0.006
    vv_, uu_ = np.mgrid[0:h, 0:w]; F0 = float(K[0, 0]) if capdev else 736.6; fx_, fy_ = F0*w/1920, F0*h/1080
    ry = np.stack([(uu_-w/2)/fx_, (vv_-h/2)/fy_, np.ones_like(D)], -1) @ cam[t0, 1, :3]    # world-y of each pixel ray
    zt = (y_tab - cam[t0, 1, 3])/np.minimum(ry, -1e-3)                                    # true table depth
    cand = (vv_ > 0.45*h) & (D > 0.05) & (ry < -0.05) & (zt < 3.0) & ~mk
    a_d, b_d = float(np.median(zt[cand]/D[cand])), 0.0
    for _ in range(6):
        sel = cand & (np.abs(a_d*D + b_d - zt) < 0.02)
        a_d, b_d = np.linalg.lstsq(np.stack([D[sel], np.ones(sel.sum())], 1), zt[sel], rcond=None)[0]
    a_d, b_d = float(a_d), float(b_d); D = (a_d*D + b_d).astype(np.float32)
    print(f'depth from the hand table plane: D = {a_d:.3f} D_da3 + {100*b_d:.1f} cm ({sel.sum()/cand.sum():.0%} table inliers)')
    np.save(os.path.join(out, 'first_depth.npy'), D)
    del model; torch.cuda.empty_cache()
    # ---- 4. table plane + hover tour (in the first-camera frame, OpenCV axes) ----
    vv, uu = np.mgrid[0:h, 0:w]
    Pc = np.stack([(uu-Kp[0, 2])/Kp[0, 0]*D, (vv-Kp[1, 2])/Kp[1, 1]*D, D], -1)
    low = (vv > 0.45*h) & (D > 0.05)
    n, d, inl_frac = fit_plane(Pc[low][::7])
    if n[1] > 0: n, d = -n, -d                         # normal points up (image-up is -y in OpenCV)
    fwd = np.array([0, 0, 1.]); u = fwd - (fwd@n)*n; u /= np.linalg.norm(u); v = np.cross(n, u)
    onp = Pc[low & (np.abs(Pc@n + d) < 0.01)]
    cu, cv_ = onp@u, onp@v
    u0, u1 = np.quantile(cu, [0.10, 0.85]); v0, v1 = np.quantile(cv_, [0.15, 0.85])
    o = -d*n                                           # a point on the plane
    ts = np.linspace(0, 1, N-DIVE+1)                   # cams[0] = dive endpoint, cams[1:] = N-DIVE tour frames
    pu = u0 + (u1-u0)*(0.15 + 0.7*ts)
    pv = (v0+v1)/2 + min(0.45*(v1-v0), 0.18)*np.sin(2*np.pi*1.0*ts)   # cap the sweep at +/-18 cm
    tilt = np.radians(TILT); head_pos = np.zeros(3)
    cams, prev_y = [], None
    if a.path == 'hand':
        # real hand: D405 at the FFW-SG2 mount relative to the retargeted TCP (thumb-index midpoint,
        # jaw axis thumb->index, continuous roll, camera side facing the head at the first frame),
        # whole episode resampled to the tour length; expressed in the first (head t0) camera frame
        from head2wrist.geometry import pose_at
        Tw = tcp_poses(J, handT, side, range(T), head_pos=cam[t0, :3, 3])
        camw = {t: np.linalg.inv(cam[t0]) @ cam_from_tcp(Tw[t], side) for t in range(T)}
        cams = [pose_at(camw, tf) for tf in np.linspace(0, T-1, len(ts))]
    for i in (range(len(ts)) if a.path == 'hover' else []):
        g = o + pu[i]*u + pv[i]*v + HOVER*n
        z = -np.cos(tilt)*n + np.sin(tilt)*u; z /= np.linalg.norm(z)
        y = v - (v@z)*z; y /= np.linalg.norm(y)
        if prev_y is None:
            if np.cross(y, z) @ (head_pos - g) < 0: y = -y
        elif y @ prev_y < 0: y = -y
        prev_y = y; x = np.cross(y, z)
        Tt = np.eye(4); Tt[:3, 0], Tt[:3, 1], Tt[:3, 2], Tt[:3, 3] = x, y, z, g
        cams.append(cam_from_tcp(Tt, 'right'))
    c2ws, Ks = [], []
    head = np.eye(4); fov_head = np.degrees(2*np.arctan(K[0, 2]/K[0, 0]))
    for i in range(DIVE):
        a_ = i/(DIVE-1); c2ws.append(interp_pose(head, cams[0], a_)); Ks.append(K_at(fov_head*(1-a_) + D405_FOV*a_))
    for c in cams[1:]:
        c2ws.append(c); Ks.append(K_at(D405_FOV))
    assert len(c2ws) == N and len(Ks) == N, (len(c2ws), N)
    w2c = np.stack([np.linalg.inv(c) for c in c2ws]).astype(np.float32)
    np.savez(os.path.join(out, 'trajectory.npz'), w2c=w2c, intrinsics=np.stack(Ks).astype(np.float32),
             image_height=1080, image_width=1920)
    cap = CAPTIONS.get(a.ep, 'A static indoor tabletop scene, soft indoor light.')
    json.dump({'0': cap, '81': 'Close-up view hovering just above the table surface. ' + cap},
              open(os.path.join(out, 'captions.json'), 'w'), indent=1)
    cam_h = [float(c[:3, 3]@n + d) for c in cams]
    meta = dict(**({'head_focal': float(K[0, 0]), 'capture_device': str(capdev)} if capdev else {}), ep=a.ep, root=os.path.abspath(a.root), path=a.path, side=side, t0=int(t0), path_coverage=float(path_coverage), mask=a.mask, inpaint_mask_frac=float(m.mean()), capsule_mask_frac=float(mfrac), hand_speed=float(spd), sharpness=float(sharp), n_frames=N, dive_frames=DIVE,
                hover_tcp_m=HOVER, tilt_deg=TILT, plane_inlier_frac=float(inl_frac),
                table_extent_u=[float(u0), float(u1)], table_extent_v=[float(v0), float(v1)],
                camera_height_above_table_m=[min(cam_h), max(cam_h)], depth_fill_scale=s, depth_hand_affine=[a_d, b_d], table_y_from_hands=y_tab,
                first_depth_median=float(np.median(D[D > 0])))
    json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'), indent=1)
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
