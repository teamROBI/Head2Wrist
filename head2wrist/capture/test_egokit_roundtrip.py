from head2wrist.config import H2W_EGODEX_ROOT
#!/usr/bin/env python3
"""
Round-trip test of egokit_to_egodex.py: an EgoDex episode -> a synthetic EgoKit session (Unity left-handed poses.txt at
the EgoDex rate, the head video re-imaged into a 1280x720 camera with an off-centre principal point, a known camera-in-head
offset) -> egokit_to_egodex.py -> compare with the original: camera poses, joint positions, the <side>Hand rotation (rebuilt
from positions; EgoDex's own differs by a few degrees), the intrinsics and the video.
"""
import argparse, json, os, subprocess, sys
import numpy as np, cv2, h5py
from scipy.spatial.transform import Rotation as R
import head2wrist.capture.egokit_to_egodex as E





def main():
    global A_, B_, C2, K, Mi, S, Tcu, Th, Thc, a, ang, ap, cam, cap, cap2, cxq, cyq, d, f, fq, fr, g, hq, i, inv, k, m, msg, n, nm, ok, p, r, rows, s_, ses, side, src, t_ms, v, vw, x, z_
    ap = argparse.ArgumentParser()
    ap.add_argument('--ep', default='basic_pick_place/1')
    ap.add_argument('--root', default=f'{H2W_EGODEX_ROOT}')
    ap.add_argument('--work', default='/tmp/egokit_roundtrip')
    ap.add_argument('--n', type=int, default=150)
    ap.add_argument('--task', default='rt')
    ap.add_argument('--idx', type=int, default=0)
    ap.add_argument('--zoom', type=float, default=1.0, help='the synthetic camera is this much narrower than EgoDex (converted focal = zoom x 736.6)')
    a = ap.parse_args()
    S = E.S
    ses = f'{a.work}/session'
    os.makedirs(ses, exist_ok=True)
    f = h5py.File(f'{a.root}/{a.ep}.hdf5')
    n = min(a.n, len(f['transforms/camera']))
    cam = f['transforms/camera'][:n].astype(float)
    K = f['camera/intrinsic'][:]
    fq, cxq, cyq = a.zoom*float(K[0, 0])/1.5, 652.0, 351.0
    Thc = np.eye(4)
    Thc[:3, :3] = R.from_euler('x', 12, degrees=True).as_matrix()
    Thc[:3, 3] = [0.03, -0.02, 0.06]
    Tcu = cam.copy()
    Tcu[:, :3, :3] = cam[:, :3, :3] @ E.CV_FROM_GL
    Tcu[:, :3, :3] = S @ Tcu[:, :3, :3] @ S
    Tcu[:, :3, 3] = cam[:, :3, 3] @ S
    Th = Tcu @ np.linalg.inv(Thc)
    hq = R.from_matrix(Th[:, :3, :3]).as_quat()
    inv = {v: k for k, v in E.EGODEX.items()}
    t_ms = 1000*np.arange(n)/30.0
    rows = []
    for i in range(n):
        r = [i, t_ms[i]] + list(Th[i, :3, 3]) + list(hq[i])
        for side in ('left', 'right'):
            for nm in E.OXR:
                if nm == 'Palm': p = 0.5*(f[f'transforms/{side}Hand'][i, :3, 3] + f[f'transforms/{side}MiddleFingerKnuckle'][i, :3, 3])
                else: p = f[f'transforms/{side}{E.EGODEX[nm]}'][i, :3, 3]
                r += list(np.asarray(p, float) @ S) + [0, 0, 0, 1]
        rows.append(' '.join(f'{x:.6f}' for x in r))
    open(f'{ses}/poses.txt', 'w').write('# idx t_ms head_pos(x y z) head_rot(x y z w) left[26]:(pos+rot) right[26]:(pos+rot)\n' + '\n'.join(rows) + '\n')
    open(f'{ses}/log.txt', 'w').write('synthetic session from ' + a.ep + '\n')
    cap = cv2.VideoCapture(f'{a.root}/{a.ep}.mp4')
    vw = cv2.VideoWriter(f'{ses}/tmp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (1280, 720))
    z_ = a.zoom/1.5
    Mi = np.array([[z_, 0, cxq - 960*z_], [0, z_, cyq - 540*z_]], np.float32)
    src = []
    for i in range(n):
        ok, fr = cap.read(); src.append(fr); vw.write(cv2.warpAffine(fr, Mi, (1280, 720), flags=cv2.INTER_AREA))
    vw.release()
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', f'{ses}/tmp.mp4', '-c:v', 'libx264', '-crf', '12', '-pix_fmt', 'yuv420p', f'{ses}/internal.mp4'], check=True)
    os.remove(f'{ses}/tmp.mp4')
    json.dump(dict(fx=fq, fy=fq, cx=cxq, cy=cyq, width=1280, height=720, T_head_cam=Thc.tolist(), dt_ms=0.0), open(f'{a.work}/calib.json', 'w'))
    side = 'left' if 'hand:left' in str(f.attrs.get('environment', '')) else 'right'
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'egokit_to_egodex.py'), '--session', ses, '--calib', f'{a.work}/calib.json',
                    '--out', f'{a.work}/out', '--task', a.task, '--idx', str(a.idx), '--side', side, '--description', str(f.attrs.get('description', ''))], check=True)
    g = h5py.File(f'{a.work}/out/{a.task}/{a.idx}.hdf5')
    m = min(n, len(g['transforms/camera']))
    C2 = g['transforms/camera'][:m].astype(float)
    ang = lambda A, B: np.degrees(np.arccos(np.clip((np.trace(np.swapaxes(A, -1, -2) @ B, axis1=-2, axis2=-1) - 1)/2, -1, 1)))
    print('frames', m, '| camera position error (mm) max', round(1000*np.abs(C2[:, :3, 3] - cam[:m, :3, 3]).max(), 3), '| rotation error (deg) max', round(float(ang(C2[:, :3, :3], cam[:m, :3, :3]).max()), 4))
    for k in ('ThumbTip', 'IndexFingerTip', 'Hand', 'Shoulder', 'Arm', 'Forearm'):
        for s_ in ('left', 'right'):
            A_, B_ = g[f'transforms/{s_}{k}'][:m].astype(float), f[f'transforms/{s_}{k}'][:m].astype(float)
            msg = f'{s_}{k}: position error (mm) max {1000*np.abs(A_[:, :3, 3] - B_[:, :3, 3]).max():.2f}'
            if k == 'Hand': msg += f', rotation vs EgoDex median {np.median(ang(A_[:, :3, :3], B_[:, :3, :3])):.1f} deg'
            print(' ', msg)
    print('intrinsic', np.round(g['camera/intrinsic'][:], 2).tolist(), 'vs EgoDex', np.round(K, 2).tolist())
    cap2 = cv2.VideoCapture(f'{a.work}/out/{a.task}/{a.idx}.mp4')
    d = []
    for i in range(0, m, 25):
        cap2.set(cv2.CAP_PROP_POS_FRAMES, i); ok, fr = cap2.read()
        if ok: d.append(np.abs(fr[40:-40, 40:-40].astype(float) - src[i][40:-40, 40:-40].astype(float)).mean())
    print('video |diff| inside a 40 px margin (8-bit)', np.round(d, 2), '| attrs', dict(capture_device=g.attrs['capture_device'], environment=g.attrs['environment']))


if __name__ == '__main__':
    main()
