from head2wrist.config import H2W_DATA_ROOT
#!/usr/bin/env python3
"""
EgoKit (Meta Quest 3) recording -> an EgoDex-format episode (<out>/<task>/<idx>.hdf5 + .mp4), so the head2wrist pipeline
runs on it unchanged (pipeline/run_episode.py --root <out> --ep <task>/<idx>).

Input session folder (EgoKit-Quest 1.0.0; format read from the app's own header string):
  internal.mp4   the passthrough (head) camera, H.264
  poses.txt      '# idx t_ms head_pos(x y z) head_rot(x y z w) left[26]:(pos+rot) right[26]:(pos+rot)', ~90 Hz,
                 Unity tracking space (left-handed, x right, y up, z forward, metres), the 26 OpenXR XR_HAND_JOINT joints
                 (0 Palm 1 Wrist 2-5 Thumb M,P,D,T 6-10 Index M,P,I,D,T 11-15 Middle 16-20 Ring 21-25 Little) as pos(3)+rot(4)
  log.txt        session log (kept as an attribute)
Calibration (--calib json, from calib_headcam.py or the device's API): fx, fy, cx, cy at the video's resolution; T_head_cam
(4x4, the camera in the HEAD frame, Unity axes, camera looking +z); dt_ms (video time 0 = pose time t0 + dt_ms).

Output, as EgoDex (verified on EgoDex: transforms/camera is world <- camera with OpenCV camera axes x right, y down,
z forward, in a right-handed y-up world; the <side>Hand rotation is x = +-(wrist -> middle knuckle) (left +, right -),
y = palm normal (d x across), z = -+(little -> index knuckle)):
  video          1920x1080, 30 fps, resampled so the principal point is the image centre and the pixels square, scaled
                 just enough to fill the frame (an off-centre principal point crops a little at the edges, no black
                 border; focal F = the camera's, scaled): only the focal differs from EgoDex (headcam.py, camera/intrinsic)
  transforms     camera, <side>Hand (rotation rebuilt from joint positions in the EgoDex convention), <side>ThumbTip,
                 <side>IndexFingerTip and the other finger joints (positions; identity rotation); not tracked by the
                 headset: <side>Shoulder and <side>Arm (the shoulder joint) at the EgoDex median offsets from the head
                 camera (gravity-aligned), <side>Forearm (the elbow) by two-link geometry from the Arm to the wrist
                 (EgoDex lengths, 45 deg outward of the lowest point) - the prep stage masks the human arm with them
  confidences    1 where the hand was tracked, 0 where the headset lost it
  attrs          capture_device, environment ('hand:<side>'), description, task, llm_description, llm_objects, log
"""
import argparse, json, os, subprocess
import numpy as np, cv2, h5py
from scipy.spatial.transform import Rotation as R, Slerp

OXR = ['Palm', 'Wrist', 'ThumbMetacarpal', 'ThumbProximal', 'ThumbDistal', 'ThumbTip'] +\
      [f'{f}{j}' for f in ('Index', 'Middle', 'Ring', 'Little') for j in ('Metacarpal', 'Proximal', 'Intermediate', 'Distal', 'Tip')]
EGODEX = {'Wrist': 'Hand', 'ThumbMetacarpal': 'ThumbKnuckle', 'ThumbProximal': 'ThumbIntermediateBase', 'ThumbDistal': 'ThumbIntermediateTip', 'ThumbTip': 'ThumbTip'}
for f in ('Index', 'Middle', 'Ring', 'Little'):
    EGODEX.update({f'{f}Metacarpal': f'{f}FingerMetacarpal', f'{f}Proximal': f'{f}FingerKnuckle', f'{f}Intermediate': f'{f}FingerIntermediateBase',
                   f'{f}Distal': f'{f}FingerIntermediateTip', f'{f}Tip': f'{f}FingerTip'})
SHOULDER = {'left': np.array([-0.032, -0.124, -0.161]), 'right': np.array([0.038, -0.123, -0.173])}   # (right, up, forward) of the head camera, EgoDex median
ARM_J = {'left': np.array([-0.180, -0.176, -0.184]), 'right': np.array([0.169, -0.177, -0.244])}     # <side>Arm = the shoulder joint, same frame, EgoDex median
L_UPPER, L_FORE, SWIVEL = 0.305, 0.265, np.radians(45.0)    # EgoDex medians: upper arm, forearm; the elbow 45 deg outward of its lowest point
S = np.diag([1., 1., -1.])                       # Unity (left-handed, z forward) <-> right-handed (z backward), y up both
CV_FROM_GL = np.diag([1., -1., -1.])             # mirrored, a Unity camera's local axes are OpenGL's (x right, y up, z back);
                                                 # OpenCV's (x right, y down, z forward) = OpenGL's x, -y, -z


def read_poses(p):
    """Rows of poses.txt -> t (ms), head pos / quat, hands {side: (n, 26, 7)}, tracked {side: (n,) bool}, t0 (the first
    timestamp of the file, video time 0 = t0 + dt_ms, also when the first rows have no head pose). A real recording
    may hold what the synthetic sessions do not: non-numeric tokens, rows before head tracking starts (quaternion all zero
    or not finite), repeated or out-of-order timestamps, and an untracked hand written as NaN or zeros. Such rows are dropped;
    an untracked hand's samples are filled from its tracked neighbours (so interpolation never mixes them in) and the
    tracked mask gives the confidences."""
    rows, n_txt = [], 0
    for l in open(p):
        if not l.strip() or l.lstrip().startswith('#'): continue
        r = l.replace(',', ' ').split()
        if len(r) < 373: n_txt += 1; continue
        try: rows.append([float(x) for x in r[:373]])
        except ValueError: n_txt += 1
    A = np.array(rows, np.float64); assert len(A), f'no 373-column numeric rows in {p}'
    t0 = float(np.nanmin(A[:, 1]))
    qn = np.linalg.norm(A[:, 5:9], axis=1); head_ok = np.isfinite(A[:, 1:9]).all(1) & (qn > 0.5)
    A = A[head_ok]; A[:, 5:9] /= qn[head_ok][:, None]; assert len(A), f'no row of {p} has a tracked head pose'
    A = A[np.argsort(A[:, 1], kind='stable')]; uniq = np.r_[True, np.diff(A[:, 1]) > 0]; A = A[uniq]
    t = A[:, 1]; head_p, head_q = A[:, 2:5], A[:, 5:9]; hands, tracked = {}, {}
    for k, s in enumerate(('left', 'right')):
        J = A[:, 9 + k*182: 9 + (k + 1)*182].reshape(-1, 26, 7).copy()
        ok = np.isfinite(J).all((1, 2)) & (np.abs(np.nan_to_num(J[:, :, :3])).sum((1, 2)) > 1e-6)
        if not ok.any(): J[:] = 0.0
        elif not ok.all():
            F_ = J.reshape(len(J), -1)
            for c in range(F_.shape[1]): F_[~ok, c] = np.interp(t[~ok], t[ok], F_[ok, c])
            J = F_.reshape(-1, 26, 7)
        hands[s], tracked[s] = J, ok
    notes = [f'{n_txt} unreadable rows' if n_txt else '', f'{(~head_ok).sum()} rows without a head pose' if (~head_ok).any() else '',
             f'{(~uniq).sum()} repeated timestamps' if (~uniq).any() else ''] +\
            [(f'{s} hand never tracked: written at the world origin with confidence 0' if not tracked[s].any() else
              f'{s} hand untracked in {(~tracked[s]).sum()} of {len(t)} samples (filled from its tracked neighbours)') for s in tracked if (~tracked[s]).any()]
    if any(notes): print('poses.txt: ' + '; '.join(x for x in notes if x), flush=True)
    return t, head_p, head_q, hands, tracked, t0


def unity_pose(p, q):
    T = np.tile(np.eye(4), (len(p), 1, 1)); T[:, :3, :3] = R.from_quat(q).as_matrix(); T[:, :3, 3] = p; return T


def to_world(T):                                 # a Unity pose -> the right-handed world (same object, mirrored z)
    out = T.copy(); out[..., :3, :3] = S @ T[..., :3, :3] @ S; out[..., :3, 3] = T[..., :3, 3] @ S; return out


def interp(t_src, X, t):                         # linear, clamped
    return np.stack([np.interp(t, t_src, X[:, k]) for k in range(X.shape[1])], 1)


def hand_rot(P, side):                           # EgoDex <side>Hand rotation from joint positions (world, right-handed)
    n_ = lambda v: v/np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)
    w = P[:, OXR.index('Wrist')]; d = n_(P[:, OXR.index('MiddleProximal')] - w)
    l0 = n_(P[:, OXR.index('IndexProximal')] - P[:, OXR.index('LittleProximal')]); nr = n_(np.cross(d, l0)); l = np.cross(nr, d)
    return np.stack([d, nr, -l], -1) if side == 'left' else np.stack([-d, nr, l], -1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--session', required=True); ap.add_argument('--calib', required=True)
    ap.add_argument('--out', default=f'{H2W_DATA_ROOT}/data/lab'); ap.add_argument('--task', required=True); ap.add_argument('--idx', type=int, required=True)
    ap.add_argument('--side', choices=['left', 'right'], required=True, help='the hand that does the task')
    ap.add_argument('--description', default=''); ap.add_argument('--objects', default='', help='comma list, e.g. "cup,plate"')
    ap.add_argument('--fps', type=float, default=30.0); ap.add_argument('--max_s', type=float, default=None)
    ap.add_argument('--start_s', type=float, default=0.0, help='video time window: one demo out of a longer session'); ap.add_argument('--end_s', type=float, default=None)
    a = ap.parse_args()
    C = json.load(open(a.calib)); Thc = np.array(C['T_head_cam'], float); dt = float(C.get('dt_ms', 0.0))
    t, hp, hq, hands, tracked, t0 = read_poses(os.path.join(a.session, 'poses.txt'))
    cap = cv2.VideoCapture(os.path.join(a.session, 'internal.mp4')); fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    Wv, Hv = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); n_in = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    try:                                                                           # OpenCV's count is an estimate; the container's is exact
        n_in = int(subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=nb_frames', '-of', 'csv=p=0',
                                   os.path.join(a.session, 'internal.mp4')], capture_output=True, text=True, check=True).stdout.strip())
    except (ValueError, subprocess.CalledProcessError): pass
    sx, sy = Wv/float(C.get('width', Wv)), Hv/float(C.get('height', Hv))       # calibration at another resolution -> scale it
    fx, fy, cx, cy = C['fx']*sx, C['fy']*sy, C['cx']*sx, C['cy']*sy
    end = min([n_in/fps_in] + [x for x in (a.end_s, None if a.max_s is None else a.start_s + a.max_s) if x is not None])
    tv = a.start_s + np.arange(int(np.floor((end - a.start_s)*a.fps + 1e-6)))/a.fps; src_idx = np.minimum(np.round(tv*fps_in).astype(int), n_in - 1)
    tp = t0 + dt + 1000*src_idx/fps_in                                            # pose time of each output frame (ms)
    tol = 500.0/fps_in; inr = (tp >= t[0] - tol) & (tp <= t[-1] + tol); src_idx, tp = src_idx[inr], tp[inr]       # only frames the pose stream covers
    if (~inr).any(): print(f'{(~inr).sum()} video frames outside the pose stream dropped', flush=True)
    # head + camera poses at the frame times (slerp), Unity -> world
    sl = Slerp(t, R.from_quat(hq)); tpc = np.clip(tp, t[0], t[-1])
    Th = unity_pose(interp(t, hp, tpc), sl(tpc).as_quat()); Tc_u = Th @ Thc
    cam = to_world(Tc_u); cam[:, :3, :3] = cam[:, :3, :3] @ CV_FROM_GL            # OpenCV camera axes
    # the video: square pixels, principal point at the centre, 1920x1080
    k_ = fx/fy    # the smallest scale whose image covers the whole 1920x1080 frame around the centred principal point (no black
    s = 1.003*max(960.0/cx, 960.0/(Wv - cx), 540.0/(k_*cy), 540.0/(k_*(Hv - cy)))   # border; an off-centre point crops a little FOV)
    F = fx*s; M = np.array([[s, 0, 960 - s*cx], [0, s*k_, 540 - s*k_*cy]], np.float32)
    print(f'video scale {s:.4f} (vs {1920.0/Wv:.4f} for the width alone): horizontal FOV kept {100*1920/(s*Wv):.1f}%', flush=True)
    os.makedirs(os.path.join(a.out, a.task), exist_ok=True); base = os.path.join(a.out, a.task, str(a.idx))
    tmp = base + '.tmp.mp4'; vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), a.fps, (1920, 1080)); cur = -1; fr = None
    n_bad = 0
    for k in src_idx:
        while cur < k:
            ok, f = cap.read(); cur += 1
            if ok: fr = f
            else: n_bad += 1
        vw.write(cv2.warpAffine(fr, M, (1920, 1080), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
    if n_bad: print(f'WARNING: {n_bad} video frames could not be decoded (the previous frame was repeated)', flush=True)
    vw.release(); subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', tmp, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '16', base + '.mp4'], check=True); os.remove(tmp)
    with h5py.File(base + '.hdf5', 'w') as h:
        h['camera/intrinsic'] = np.array([[F, 0, 960], [0, F, 540], [0, 0, 1.]], np.float32)
        h['transforms/camera'] = cam.astype(np.float32)
        for side in ('left', 'right'):
            J = hands[side]; Pw = interp(t, J[:, :, :3].reshape(len(t), -1), tpc).reshape(len(tp), 26, 3) @ S      # world positions
            ok = tracked[side].astype(float); conf = np.interp(tpc, t, ok) > 0.5
            for k, nm in enumerate(OXR):
                if nm not in EGODEX: continue
                T = np.tile(np.eye(4), (len(tp), 1, 1)); T[:, :3, 3] = Pw[:, k]
                if nm == 'Wrist': T[:, :3, :3] = hand_rot(Pw, side)
                h[f'transforms/{side}{EGODEX[nm]}'] = T.astype(np.float32); h[f'confidences/{side}{EGODEX[nm]}'] = conf.astype(np.float32)
            fw = cam[:, :3, 2].copy(); fw[:, 1] = 0; fw /= np.linalg.norm(fw, axis=1, keepdims=True)
            rt = cam[:, :3, 0].copy(); rt[:, 1] = 0; rt -= fw*np.sum(rt*fw, 1, keepdims=True); rt /= np.linalg.norm(rt, axis=1, keepdims=True)
            o = SHOULDER[side]; T = np.tile(np.eye(4), (len(tp), 1, 1)); T[:, :3, 3] = cam[:, :3, 3] + o[0]*rt + o[1]*np.array([0, 1., 0]) + o[2]*fw
            h[f'transforms/{side}Shoulder'] = T.astype(np.float32); h[f'confidences/{side}Shoulder'] = np.ones(len(tp), np.float32)
            o = ARM_J[side]; A_ = cam[:, :3, 3] + o[0]*rt + o[1]*np.array([0, 1., 0]) + o[2]*fw; W_ = Pw[:, OXR.index('Wrist')]
            v = W_ - A_; dd = np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6); ax = v/dd; dist = np.minimum(dd[:, 0], L_UPPER + L_FORE - 1e-4)
            x = (L_UPPER**2 - L_FORE**2 + dist**2)/(2*dist); rr = np.sqrt(np.maximum(L_UPPER**2 - x**2, 0))           # elbow circle: centre along the axis, radius
            dn = np.array([0, -1., 0]) - ax*(-ax[:, 1:2]); dn /= np.maximum(np.linalg.norm(dn, axis=1, keepdims=True), 1e-6); sd = np.cross(ax, dn)
            out_ = rt if side == 'right' else -rt; sd *= np.sign(np.sum(sd*out_, 1, keepdims=True) + 1e-9)
            E_ = A_ + ax*x[:, None] + rr[:, None]*(np.cos(SWIVEL)*dn + np.sin(SWIVEL)*sd)
            for nm_, X_ in (('Arm', A_), ('Forearm', E_)):
                T = np.tile(np.eye(4), (len(tp), 1, 1)); T[:, :3, 3] = X_; h[f'transforms/{side}{nm_}'] = T.astype(np.float32); h[f'confidences/{side}{nm_}'] = np.ones(len(tp), np.float32)
        h.attrs.update(capture_device='quest3_egokit', environment=f'hand:{a.side}', description=a.description, llm_description=a.description,
                       task=a.task, llm_objects=np.array([o for o in a.objects.split(',') if o], dtype=h5py.string_dtype()),
                       session_name=os.path.basename(os.path.normpath(a.session)), calib=json.dumps(C),
                       log=open(os.path.join(a.session, 'log.txt')).read()[:20000] if os.path.exists(os.path.join(a.session, 'log.txt')) else '')
    print(f'{base}: {len(tp)} frames at {a.fps} fps (video {Wv}x{Hv} {fps_in:.1f} fps -> 1920x1080, F = {F:.1f}), poses {len(t)} at '
          f'{1000*(len(t) - 1)/max(t[-1] - t[0], 1):.0f} Hz, {a.side} hand tracked {100*np.mean(np.interp(tpc, t, tracked[a.side].astype(float)) > 0.5):.0f}% of frames')


if __name__ == '__main__':
    main()
