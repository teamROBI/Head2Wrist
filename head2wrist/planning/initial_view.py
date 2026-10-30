from head2wrist.config import H2W_EGODEX_ROOT, H2W_OUTPUT_ROOT
#!/usr/bin/env python3
"""
Initial robot-wrist pose = an overview D405 view that sees ALL objects on the table.

  1. objects: SAM 3 masks from generic prompts (masks/sam3_objects.py) on the input frame, lifted with its metric
     depth (height-only clustering confused the flat phone with depth seams). Default: only the task objects, i.e.
     masks within 10 cm of the demo's grasp or release pinch point (on basic_pick_place/74, seven objects spread
     over the table, no view within arm reach saw them all)
  2. search camera poses on the ROBOT's side of the table (it stands where the human sat; right arm: near side,
     slightly right, looking forward-left and down; left arm: mirrored; oblique 45-65 deg from vertical, level
     horizon), within arm reach of that side's shoulder (EgoDex <side>Shoulder as proxy), >=10 cm in front of it,
     15-50 cm above the table; for each (azimuth, elevation) take the
     smallest distance at which every object point projects inside the image (margins; the left band is kept free
     because the D405 sees the gripper there)
  3. score: close + oblique ~55 deg + azimuth ~25 deg toward the arm's side
Outputs initial_view.json (camera c2w in the first-camera frame, OpenCV axes) + a preview splat image.
"""
import argparse, json, os, sys
import numpy as np, cv2, h5py
from scipy.spatial.transform import Rotation as R
from scipy import ndimage

from head2wrist.planning.make_hover_tour import fit_plane
from head2wrist.wrist_pose import T_TCP_D405
from head2wrist.robot.finger_clearance import lowest_finger_height

W, H = 848, 480; FX = (W/2)/np.tan(np.radians(87/2))
MARGIN_L, MARGIN = 0.15, 0.03          # left band (gripper) and other margins, fraction of width/height


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default=f'{H2W_OUTPUT_ROOT}/lyra2/explore_inputs/basic_pick_place_1')
    ap.add_argument('--ep', default='basic_pick_place/1')
    ap.add_argument('--side', default=None, choices=['right', 'left'], help='robot arm = demo hand (default: meta.json)')
    ap.add_argument('--cover', choices=['task', 'all'], default='task',
                    help='task: objects within --task_radius of the demo grasp / release points (the object, the target and '
                         'their neighbours; default); all: every SAM 3 object (fails on cluttered tables: no view within reach)')
    ap.add_argument('--task_radius', type=float, default=0.10)
    ap.add_argument('--min_cam_h', type=float, default=0.28, help='preferred lowest camera height above the table (m); '
                    'falls back to 0.15. A low overview put the TCP 4 cm above the table on real cube_box/1 and the transit '
                    'looped 27 cm up to reach the pre-grasp (plan_approach.py); from >= 0.28 m the TCP starts above it')
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.inputs, 'meta.json'))); t0 = meta['t0']; side = a.side or meta.get('side', 'right')
    a.ep = meta.get('ep', a.ep); ROOT = meta.get('root', f'{H2W_EGODEX_ROOT}'); sgn = 1.0 if side == 'right' else -1.0
    D = np.load(os.path.join(a.inputs, 'first_depth.npy')); h, w = D.shape
    img = cv2.cvtColor(cv2.imread(os.path.join(a.inputs, 'first_frame.png')), cv2.COLOR_BGR2RGB)
    import os as _os, sys as _sys; import head2wrist.headcam as headcam; F0 = headcam.focal(a.inputs)
    fx, fy = F0*w/1920, F0*h/1080
    vv, uu = np.mgrid[0:h, 0:w]; ok = D > 0.05
    P = np.stack([(uu-w/2)/fx*D, (vv-h/2)/fy*D, D], -1)
    col = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    # ---- 1. table plane, walls, objects ----
    low = (vv > 0.45*h) & ok
    n, d, _ = fit_plane(P[low][::7])
    if n[1] > 0: n, d = -n, -d                                  # normal points up
    hgt = P@n + d
    walls = []
    rest = ok & (hgt > 0.20)
    for _ in range(2):
        pts = P[rest][::5]
        if len(pts) < 500: break
        wn, wd, frac = fit_plane(pts, tol=0.01)
        if abs(wn@n) > 0.5: break                                # not vertical -> not a wall
        walls.append((wn, wd)); rest &= np.abs(P@wn + wd) > 0.01
    near_wall = np.zeros_like(ok)
    for wn, wd in walls: near_wall |= np.abs(P@wn + wd) < 0.025
    # objects = SAM 3 masks (generic prompts, see sam3_objects), on the table; geometry alone can't separate a flat
    # phone (1.2 cm) from depth seams / the inpainted region (0.8-1.1 cm)
    sm = np.load(os.path.join(a.inputs, 'sam3_object_masks.npy'))
    with h5py.File(os.path.join(ROOT, a.ep)+'.hdf5') as f:
        cam_ = f['transforms/camera'][:].astype(float)
        Jd = {k: f['transforms/'+k][:, :3, 3].astype(float) for k in f['transforms'].keys() if k.endswith(('ThumbTip', 'IndexFingerTip', 'Hand'))}
    pass
    from head2wrist.pipeline.demo_events import detect_pick_place
    tg, tr = detect_pick_place(Jd, side); inv_ = np.linalg.inv(cam_[t0])
    task_pts = [inv_[:3, :3]@((Jd[f'{side}ThumbTip'][t] + Jd[f'{side}IndexFingerTip'][t])/2) + inv_[:3, 3] for t in (tg, tr)]
    keep = np.zeros_like(ok); boxes = []
    for mk in sm:
        mk = cv2.resize(mk.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        mk &= ok & (hgt > -0.02) & (hgt < 0.25)
        if mk.sum() < 50 or mk.mean() > 0.15: continue
        if a.cover == 'task' and min(np.linalg.norm(P[mk][::7] - q, axis=1).min() for q in task_pts) > a.task_radius: continue
        keep |= mk; ys, xs = np.nonzero(mk); boxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
    if a.cover == 'task' and not boxes:                          # a smaller radius dropped every object: keep the one nearest
        best = None                                              # the grasp point (the grasped object), the minimum to frame
        for mk in sm:
            mk = cv2.resize(mk.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            mk &= ok & (hgt > -0.02) & (hgt < 0.25)
            if mk.sum() < 50 or mk.mean() > 0.15: continue
            dg = np.linalg.norm(P[mk][::7] - task_pts[0], axis=1).min()
            if best is None or dg < best[0]: best = (dg, mk)
        assert best is not None, 'no SAM 3 object on the table in the head frame (object out of view or blurred at t0?)'
        keep |= best[1]; ys, xs = np.nonzero(best[1]); boxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
        print(f'no object within {100*a.task_radius:.0f} cm: framing the object nearest the grasp ({100*best[0]:.1f} cm)', flush=True)
    print('walls', len(walls), f'| objects kept ({a.cover})', len(boxes))
    O = P[keep]; C = O.mean(0)
    vis = col.copy(); vis[keep] = (0.5*vis[keep] + 0.5*np.array([0, 255, 0])).astype(np.uint8)
    for b in boxes: cv2.rectangle(vis, b[:2], b[2:], (255, 0, 0), 2)
    cv2.imwrite(os.path.join(a.inputs, 'initial_view_objects.jpg'), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    # ---- 2. candidate views ----
    fwd = np.array([0, 0, 1.]); f_t = fwd - (fwd@n)*n; f_t /= np.linalg.norm(f_t); r_t = np.cross(f_t, n)  # table axes: away / right (OpenCV x ~ right)
    if r_t @ np.array([1, 0, 0.]) < 0: r_t = -r_t
    with h5py.File(os.path.join(ROOT, a.ep)+'.hdf5') as f:
        cam = f['transforms/camera'][:].astype(float)
    Osub = O[::max(1, len(O)//4000)]
    # robot body proxy: EgoDex right shoulder and head at t0, in the first-camera frame
    inv0 = np.linalg.inv(cam[t0])
    with h5py.File(os.path.join(ROOT, a.ep)+'.hdf5') as f:
        sh_w = f[f'transforms/{side}Shoulder'][t0, :3, 3].astype(float)
    shoulder = inv0[:3, :3]@sh_w + inv0[:3, 3]; head = np.zeros(3)
    head_h = head@n + d
    cands = []
    for az in sgn*np.radians(np.arange(0, 61, 5)):               # robot side: 0 = straight from the viewer, + = to the right
        for el in np.radians(np.arange(45, 66, 5)):              # oblique: optical axis 45-65 deg from straight down
            dir_out = np.cos(el)*n + np.sin(el)*(-np.cos(az)*f_t + np.sin(az)*r_t)
            z = -dir_out
            # level horizon in the UPRIGHT image (raw D405 rotated 90 deg CCW: upright-right = raw +y, upright-up = raw +x):
            # raw y horizontal, raw x pointing up
            y = np.cross(z, n); y /= np.linalg.norm(y); x = np.cross(y, z)
            if x @ n < 0: x, y = -x, -y
            Rc = np.stack([x, y, z], 1)
            for dist in np.arange(0.15, 0.90, 0.01):
                pos = C + dist*dir_out
                c = (Osub - pos)@Rc
                if (c[:, 2] < 0.05).any(): continue
                u = FX*c[:, 0]/c[:, 2] + W/2; v = FX*c[:, 1]/c[:, 2] + H/2
                if u.min() > MARGIN_L*W and u.max() < (1-MARGIN)*W and v.min() > MARGIN*H and v.max() < (1-MARGIN)*H:
                    hc = pos@n + d
                    reach = np.linalg.norm(pos - shoulder)
                    fwd_of_shoulder = (pos - shoulder)@f_t
                    if hc < a.min_cam_h:   # a small object framed from below the lowest camera height: back off
                        continue           # (one 6 cm cube on the real table: every view broke at 15 cm)
                    if hc > 0.50 or reach > 0.641 + 0.14 or fwd_of_shoulder < 0.10:
                        break
                    # the gripper hangs ~15 cm in front of / below the camera: a low oblique view put the fingertips
                    # 2 cm into the table (stack/48; 1.4 cm over 36 frames on basic_pick_place/1) -> farther out
                    Tc_ = np.eye(4); Tc_[:3, :3] = Rc; Tc_[:3, 3] = pos
                    if lowest_finger_height(Tc_ @ np.linalg.inv(T_TCP_D405), n, d) < 0.02:
                        continue
                    score = dist + 0.2*abs(np.degrees(el)-55)/10 + 0.1*abs(abs(np.degrees(az))-25)/10
                    T = np.eye(4); T[:3, :3] = Rc; T[:3, 3] = pos
                    cands.append((score, np.degrees(az), np.degrees(el), dist, T))
                    break
    print('candidates', len(cands))
    if not cands and (a.cover != 'task' or a.task_radius > 0.04):   # a cluttered table: no reachable view frames them all
        nr = a.task_radius/2 if a.cover == 'task' else a.task_radius
        print(f'no view frames the kept objects: retrying with --cover task --task_radius {nr:.3f}', flush=True)
        os.execv(sys.executable, [sys.executable] + sys.argv + ['--cover', 'task', '--task_radius', f'{nr:.3f}'])
    if not cands and a.min_cam_h > 0.15:
        print(f'no view from >= {a.min_cam_h:.2f} m: retrying with --min_cam_h 0.15', flush=True)
        os.execv(sys.executable, [sys.executable] + sys.argv + ['--min_cam_h', '0.15'])
    assert cands, 'no reachable overview pose frames the task objects'
    cands.sort(key=lambda c: c[0]); score, az, el, dist, T = cands[0]
    # ---- 3. preview: splat the input frame's points into the chosen view ----
    Pall = P[ok]; call = col[ok]; c = (Pall - T[:3, 3])@T[:3, :3]; m = c[:, 2] > 0.02
    u = (FX*c[m, 0]/c[m, 2] + W/2).astype(int); v = (FX*c[m, 1]/c[m, 2] + H/2).astype(int); zz = c[m, 2]
    ib = (u >= 0) & (u < W) & (v >= 0) & (v < H); order = np.argsort(-zz[ib])
    pv = np.zeros((H, W, 3), np.uint8); pv[v[ib][order], u[ib][order]] = call[m][ib][order]
    pv = cv2.dilate(pv, np.ones((3, 3), np.uint8))
    cv2.rectangle(pv, (int(MARGIN_L*W), int(MARGIN*H)), (int((1-MARGIN)*W), int((1-MARGIN)*H)), (0, 255, 255), 1)
    cv2.putText(pv, 'gripper band', (8, H//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite(os.path.join(a.inputs, 'initial_view_preview.jpg'), cv2.cvtColor(pv, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(a.inputs, 'initial_view_preview_upright.jpg'), cv2.rotate(cv2.cvtColor(pv, cv2.COLOR_RGB2BGR), cv2.ROTATE_90_COUNTERCLOCKWISE))
    # top-view schematic (table plane coordinates: right = r_t, away = f_t)
    def tv(p): return np.array([p@r_t, p@f_t])
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6)); Ot = np.array([tv(p) for p in O[::50]])
    ax.scatter(Ot[:, 0], Ot[:, 1], s=1, c='tab:green', label='objects (SAM 3)')
    for nm, p, cl in (('head', head, 'k'), (f'{side} shoulder', shoulder, 'tab:purple'), ('D405 start', T[:3, 3], 'tab:red')):
        q = tv(p); ax.scatter(*q, s=60, c=cl, label=nm)
    q = tv(T[:3, 3]); zf = tv(T[:3, 3] + 0.2*T[:3, 2]) - q; ax.arrow(*q, *zf, color='tab:red', width=0.004)
    ax.add_patch(plt.Circle(tv(shoulder), 0.641, fill=False, ls='--', color='tab:purple', label='arm reach 641 mm'))
    ax.set_aspect('equal'); ax.set_xlabel('right (m)'); ax.set_ylabel('away from the robot (m)'); ax.legend(loc='lower left', fontsize=8)
    ax.set_title('top view: robot side at the bottom'); fig.savefig(os.path.join(a.inputs, 'initial_view_topdown.png'), dpi=110); plt.close(fig)
    out = dict(azimuth_deg=float(az), optical_axis_from_down_deg=float(el), distance_to_objects_centre_m=float(dist),
               camera_height_above_table_m=float(T[:3, 3]@n + d), head_height_above_table_m=float(head_h),
               reach_from_shoulder_m=float(np.linalg.norm(T[:3, 3]-shoulder)), side=side, cover=a.cover, n_objects=len(boxes), n_walls=len(walls),
               c2w=T.tolist(), note='first-camera frame, OpenCV axes; azimuth 0 = from the viewer side, +90 = from the right')
    json.dump(out, open(os.path.join(a.inputs, 'initial_view.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'c2w'}))


if __name__ == '__main__':
    main()
