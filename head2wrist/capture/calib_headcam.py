from head2wrist.config import H2W_DATA_ROOT
#!/usr/bin/env python3
"""
Self-calibration of a head-mounted capture device's camera against its own hand tracking (EgoKit / Meta Quest 3): the
headset tracks the hand joints in 3D (poses.txt); MediaPipe finds the same joints in the camera's images; the camera's
focal length, principal point, pose in the HEAD frame and the video-to-pose time offset are the ones that project the
tracked joints onto the detected ones (robust least squares; the detections are assigned to the left / right tracked hand
by the better fit, MediaPipe's handedness assumes a mirrored selfie view). Output: the --calib json egokit_to_egodex.py
reads (fx, fy, cx, cy, width, height, T_head_cam (Unity axes, camera looking +z, y up), dt_ms) + rms_px, n.
Run in $H2W_DATA_ROOT/envs/capture (MediaPipe stays out of the main .venv):
  calib_headcam.py --session <session folder> --out <calib.json> [--every 3] [--max_frames 900]
"""
import argparse, json, os, sys
import numpy as np, cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R, Slerp
from head2wrist.capture.egokit_to_egodex import read_poses, unity_pose, interp

MP2OXR = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25]   # MediaPipe's 21 landmarks -> OpenXR's 26 joints
USE = [3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]   # thumb IP / tip and the PIP, DIP, tip of each finger: defined alike in both (the wrist and MCPs are not)
MODEL = f'{H2W_DATA_ROOT}/assets/mediapipe/hand_landmarker.task'


def detect(video, every, max_frames):
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL),
                                                   running_mode=vision.RunningMode.VIDEO, num_hands=2, min_hand_detection_confidence=0.5))
    cap = cv2.VideoCapture(video); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0; W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out, k = [], -1
    while len(out) < 4*max_frames:
        ok, fr = cap.read(); k += 1
        if not ok or k//every >= max_frames: break
        if k % every: continue
        res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)), int(1000*k/fps))
        for h in res.hand_landmarks:
            out.append((k, np.array([[p.x*W, p.y*H] for p in h], np.float64)))
    lm.close(); return out, fps, W, H


def project(par, Pw, Twh):
    f, cx, cy = np.exp(par[0]), par[1], par[2]
    Thc = np.eye(4); Thc[:3, :3] = R.from_rotvec(par[3:6]).as_matrix(); Thc[:3, 3] = par[6:9]
    Tcw = np.linalg.inv(Twh @ Thc); Pc = np.einsum('nij,nkj->nki', Tcw[:, :3, :3], Pw) + Tcw[:, None, :3, 3]      # per detection
    z = np.maximum(Pc[..., 2], 1e-3); return np.stack([cx + f*Pc[..., 0]/z, cy - f*Pc[..., 1]/z], -1), Pc[..., 2]   # Unity camera: y up -> image v down


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--session', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--every', type=int, default=3); ap.add_argument('--max_frames', type=int, default=900)
    ap.add_argument('--dt_range', type=float, default=200.0, help='+- ms searched for the video-to-pose offset')
    ap.add_argument('--t_prior', default='-0.032,-0.02,0.06;0.0,-0.02,0.06;0.032,-0.02,0.06',
                    help='candidate camera positions in the head frame from the device geometry (m, Unity x right y up z forward; ";"-separated: '
                         'Quest 3 has an RGB camera ~3.2 cm left and right of the centre): each is a prior (sigma --t_sigma) for one '
                         'search; the one that reprojects best wins')
    ap.add_argument('--t_sigma', type=float, default=0.015)
    a = ap.parse_args()
    t, hp, hq, hands, tracked, t0 = read_poses(os.path.join(a.session, 'poses.txt'))
    dets, fps, W, H = detect(os.path.join(a.session, 'internal.mp4'), a.every, a.max_frames)
    print(f'{len(dets)} hand detections in {W}x{H} at {fps:.1f} fps', flush=True); assert len(dets) > 20, 'too few hand detections'
    fr = np.array([k for k, _ in dets]); UV = np.stack([u for _, u in dets])[:, USE]; sl = Slerp(t, R.from_quat(hq))
    TPR = [np.array([float(x) for x in c.split(',')]) for c in a.t_prior.split(';')]; tpr = TPR[0]

    def at(dt):                                    # tracked joints (both hands) + head pose at each detection's time
        tp = np.clip(t0 + dt + 1000*fr/fps, t[0], t[-1]); Twh = unity_pose(interp(t, hp, tp), sl(tp).as_quat())
        J = {s: interp(t, hands[s][:, :, :3].reshape(len(t), -1), tp).reshape(len(tp), 26, 3)[:, np.array(MP2OXR)[USE]] for s in ('left', 'right')}
        ok = {s: np.interp(tp, t, tracked[s].astype(float)) > 0.99 for s in J}
        return Twh, J, ok

    def fit(dt, par0, iters=3):
        Twh, J, ok = at(dt); par = par0.copy()
        for _ in range(iters):                     # assign each detection to the hand it fits, then solve
            err = {}
            for s in J:
                uv, z = project(par, J[s], Twh); e = np.median(np.linalg.norm(uv - UV, axis=-1), 1); e[~ok[s] | (z.min(1) <= 0.02)] = np.inf; err[s] = e
            side = np.where(err['left'] <= err['right'], 'left', 'right'); best = np.minimum(err['left'], err['right']); use = np.isfinite(best) & (best < max(60.0, np.percentile(best[np.isfinite(best)], 80)) if np.isfinite(best).any() else False)
            if use.sum() < 10: return np.inf, par, use
            P = np.where((side[use] == 'left')[:, None, None], J['left'][use], J['right'][use]); Tw = Twh[use]; U = UV[use]
            k_ = max(1, U.size//6)                                      # prior: 1 sigma = 6 px (huber's quadratic zone), repeated to weigh ~1/2 of the data
            r = least_squares(lambda p: np.concatenate([(project(p, P, Tw)[0] - U).ravel(), np.repeat((p[6:9] - tpr)/a.t_sigma*6.0, k_)]), par, loss='huber', f_scale=6.0, max_nfev=200); par = r.x
        uv, _ = project(par, P, Tw); res = np.linalg.norm(uv - U, axis=-1)
        return float(np.sqrt(np.mean(np.minimum(res, 50)**2))), par, use

    f0 = np.log(W/2/np.tan(np.radians(45))); cand = []
    for tpr in TPR:                                # one search per candidate camera position (fit() reads tpr)
        best = (np.inf, None, None, None)
        for pitch in (-25, -10, 0, 10):            # restarts over the camera's tilt in the head frame
            par0 = np.array([f0, W/2, H/2] + list(R.from_euler('x', pitch, degrees=True).as_rotvec()) + list(tpr))
            for dt in np.arange(-a.dt_range, a.dt_range + 1, 50.0):
                c, par, use = fit(dt, par0, iters=2)
                if c < best[0]: best = (c, par, dt, use)
        for dt in np.arange(best[2] - 50, best[2] + 51, 10.0):    # refine the offset around the best
            c, par, use = fit(dt, best[1], iters=3)
            if c < best[0]: best = (c, par, dt, use)
        print(f'  prior t {tpr}: rms {best[0]:.2f} px, f {np.exp(best[1][0]):.1f}, t {np.round(best[1][6:9], 3)}, dt {best[2]:.0f} ms', flush=True); cand.append(best + (tpr,))
    c, par, dt, use, tpr = min(cand, key=lambda x: x[0]); f = float(np.exp(par[0])); Thc = np.eye(4); Thc[:3, :3] = R.from_rotvec(par[3:6]).as_matrix(); Thc[:3, 3] = par[6:9]
    out = dict(fx=f, fy=f, cx=float(par[1]), cy=float(par[2]), width=W, height=H, T_head_cam=Thc.tolist(), dt_ms=float(dt), rms_px=c, n=int(use.sum()),
               hfov_deg=float(np.degrees(2*np.arctan(W/2/f))), t_prior=tpr.tolist(), session=os.path.abspath(a.session))
    json.dump(out, open(a.out, 'w'), indent=1)
    print(f"f {f:.1f} px (hfov {out['hfov_deg']:.1f} deg), principal point ({par[1]:.1f}, {par[2]:.1f}), camera in head t {np.round(par[6:9], 3)} m, "
          f"rot {np.round(R.from_rotvec(par[3:6]).as_euler('xyz', degrees=True), 1)} deg, dt {dt:.0f} ms, rms {c:.1f} px over {int(use.sum())} hands -> {a.out}")


if __name__ == '__main__':
    main()
