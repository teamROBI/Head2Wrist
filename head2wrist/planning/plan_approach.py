from head2wrist.config import H2W_EGODEX_ROOT, H2W_OUTPUT_ROOT
#!/usr/bin/env python3
"""
Draw the overview -> pre-grasp -> grasp trajectory in TCP (gripper) space; the D405 follows via the URDF mount.

  keyposes: P0 = TCP for the approved overview camera pose, P2 = grasp aligned with the human hand at the lift
            moment, raised so the RH-P12-RN fingertips clear the table (robot/finger_clearance.py);
            moment (wrist_pose.hand_aligned_grasp: TCP at the thumb-index midpoint, gripper pointing the way the
            fingers point, D405 on the back of the hand, jaws along thumb->index kept level), P1 = P2 moved 12 cm
            back along the human's final approach direction
  position: cubic Bezier P0 -> P1 (end tangent = descent direction; control points raised for clearance) + line P1 -> P2
  orientation: jaw axis (= upright-image horizontal) kept in the table plane on every frame, so the horizon never
            leans: ease-in/out interpolation of its heading and of the approach pitch overview -> grasp over the
            transit, fixed during the descent; frames where the phone would leave the image get a corrective turn
            toward it (smoothed, then re-levelled)
  timing: arc-length parameterised; speed <= 1.2 cm/frame transit, <= 0.8 cm/frame last 12 cm, 0 at both ends,
          applied to BOTH the TCP and the camera; slowed where view change > 2.5 deg/frame or roll > 5 deg/frame
Outputs approach_plan.npz (TCP + camera poses in the first-camera frame), a top/side-view figure and upright previews.
"""
import argparse, json, os, sys
import numpy as np, cv2, h5py
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.ndimage import gaussian_filter1d

from head2wrist.wrist_pose import palm_normal, hand_aligned_grasp, relevel, to_upright, T_TCP_D405
from head2wrist.planning.make_hover_tour import fit_plane



def pitch(Rm):
    k = np.cross(n, Rm[:, 1]); return np.arctan2(Rm[:, 2]@k, -(Rm[:, 2]@n))


def ori(sv):
    if sv >= L_tr: return P2[:3, :3]
    x = sv/L_tr; e = x*x*x*(10-15*x+6*x*x)
    y = R.from_rotvec(e*psi2*n).apply(y0); k = np.cross(n, y); th = th0 + e*(th2-th0)
    z = -np.cos(th)*n + np.sin(th)*k
    return np.stack([np.cross(y, z), y, z], 1)


def vcap(sv):
    base = V_T if sv < L - DESC else V_D
    ramp = min(1.0, np.sqrt(max(sv, 1e-4)/0.04), np.sqrt(max(L-sv, 1e-4)/0.03))  # smooth start / stop
    return max(base*ramp, 0.0015)


def cam_of(T): return T @ T_TCP_D405


def proj(C, p):
    E = np.linalg.inv(C); c = E[:3, :3]@p + E[:3, 3]
    return (FX*c[0]/c[2]+W/2, FX*c[1]/c[2]+H/2, c[2])


def draw_view(C, T, i):
    c = (Pall-C[:3, 3])@C[:3, :3]; m = c[:, 2] > 0.02
    u = (FX*c[m, 0]/c[m, 2]+W/2).astype(int); v2 = (FX*c[m, 1]/c[m, 2]+H/2).astype(int); zz = c[m, 2]
    ib = (u >= 0) & (u < W) & (v2 >= 0) & (v2 < H); od = np.argsort(-zz[ib])
    pv = np.zeros((H, W, 3), np.uint8); pv[v2[ib][od], u[ib][od]] = call[m][ib][od]; pv = cv2.dilate(pv, np.ones((3, 3), np.uint8))
    for sgn in (-1, 1):                                   # jaws: pads at +/- half the opening along the closing axis
        pad = T[:3, 3] + sgn*0.5*ap_open*T[:3, 1]; root = pad - 0.05*T[:3, 2]
        a_, b_ = proj(C, pad), proj(C, root)
        if a_[2] > 0 and b_[2] > 0:
            cv2.line(pv, (int(a_[0]), int(a_[1])), (int(b_[0]), int(b_[1])), (0, 255, 255), 4)
    uu_, vv_, zc = proj(C, contact)
    if zc > 0: cv2.circle(pv, (int(uu_), int(vv_)), 10, (255, 255, 0), 2)
    pv = np.ascontiguousarray(to_upright(pv))
    lean = np.degrees(np.arcsin(np.clip(C[:3, 1]@n, -1, 1)))
    cv2.putText(pv, f'{i}/{len(CAM)-1} ' + ('transit' if i < n_tr else 'descent'), (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(pv, f'horizon {lean:+.1f} deg', (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    return pv


def main():
    global A_, C, CAM, D, DESC, EP, F0, FINGER_CLEAR, FX, GR, H, I, J, L, L_tr, MAX_APPROACH_DEG, MESH, O, Ob, Og, Osub, P0, P1, P2, Pall, Pc, ROT_R, ROT_V, Ra, Rb, Rg, Rot_, SIDE, SQUEEZE, STANDOFF, T, TCP, THIN_H, THIN_TILT_DEG, Ta, Tb, VAR, V_D, V_T, Vo, Vt, W, _, _a, _hz, _os, _sys, a_, ang, ap_, ap_open, app_human_deg, ax, axs, b1, b2, b_, c_xy, call, cam, cam_step, cand_, chord, clear_obj, clear_tab, col, contact, corr, cp, cu_, curve, cv__, d, dcs, detect_pick_place, dist_, dorsal, e1, e2, end_dir, er, ev, f, far, fig, fing, finger_pts, fn, ft, fwd, fx, fy, fz, g, g_c, g_close, g_hold, gm_, grasp_raise, grasp_shift, grasp_support, grid, h, hT, headcam, hgt_, hi_, hv, i, ij, info, inside, inv0, j, k, k_, lab, lift_c, line, lo, lo_, lost, m, m_, matplotlib, meta, n, n_tr, near_, near_contact, nm, nxt, obj_h, objm, ok, ov_cam, p, p0, p1, p2, p_, p_h, p_xy, path, pinch, plt, psi2, q, raise_for_fingers, roll, rot, rt, s, s_, seg, sel, sl_, sm, sm_, st, step, subprocess, support_height, sv, sv_, svals, t0, t_rel, tab_need, tf0, tg, th0, th2, tiles, tilt_cut, tmpv, tot, tp, tv, u, up_w, uu, v, v_, vf, vis, vv, vw, w, width_, wr_c, x_, xy, y0, y_, z, z_, zt
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--inputs', default=f'{H2W_OUTPUT_ROOT}/lyra2/explore_inputs/basic_pick_place_1')
    ap_.add_argument('--root', default=f'{H2W_EGODEX_ROOT}')
    ap_.add_argument('--ep', default=None, help='EgoDex episode (default: meta.json of --inputs)')
    ap_.add_argument('--side', default=None, choices=['right', 'left'], help='demo hand = robot arm (default: meta.json)')
    ap_.add_argument('--approach', choices=['direct', 'human'], default='direct',
                     help='direct: nearly straight from the overview to a pre-grasp 6 cm back along the gripper\'s own axis, then down that axis '
                          '(real cube_box: the human-direction pre-grasp sent the gripper sideways and up first); human: the pre-grasp '
                          '12 cm back along the human\'s final approach, the curve arriving along it (the sim experiments)')
    A_ = ap_.parse_args()
    I = A_.inputs
    pass
    from head2wrist.pipeline.demo_events import detect_pick_place
    W, H = 848, 480
    FX = (W/2)/np.tan(np.radians(87/2))
    STANDOFF, V_T, V_D, ROT_V, ROT_R = 0.12, 0.012, 0.008, 2.5, 5.0
    meta = json.load(open(I+'/meta.json'))
    t0 = meta['t0']
    SIDE = A_.side or meta.get('side', 'right')
    EP = os.path.join(A_.root, A_.ep or meta['ep'])
    ov_cam = np.array(json.load(open(I+'/initial_view.json'))['c2w'])
    f = h5py.File(EP+'.hdf5')
    cam = f['transforms/camera'][:].astype(float)
    J = {k: f['transforms/'+k][:, :3, 3].astype(float) for k in f['transforms'].keys() if k.startswith(SIDE)}
    hT = f[f'transforms/{SIDE}Hand'][:].astype(float)
    inv0 = np.linalg.inv(cam[t0])
    up_w = np.array([0, 1, 0.])
    g = (J[f'{SIDE}ThumbTip']+J[f'{SIDE}IndexFingerTip'])/2
    ev = detect_pick_place(J, SIDE)
    assert ev is not None, f'no pick-and-place found for the {SIDE} hand'
    tg, t_rel = ev
    v = g[tg-3]-g[tg-12]
    v /= np.linalg.norm(v)
    import os as _os, sys as _sys
    pass
    import head2wrist.headcam as headcam
    F0 = headcam.focal(I)
    D = np.load(I+'/first_depth.npy')
    h, w = D.shape
    fx, fy = F0*w/1920, F0*h/1080
    vv, uu = np.mgrid[0:h, 0:w]
    ok = D > 0.05
    Pc = np.stack([(uu-w/2)/fx*D, (vv-h/2)/fy*D, D], -1)
    n, d, _ = fit_plane(Pc[(vv > 0.45*h) & ok][::7])
    n, d = (-n, -d) if n[1] > 0 else (n, d)
    tf0 = lambda p: inv0[:3, :3]@p + inv0[:3, 3]
    g_c, wr_c = tf0(g[tg]), tf0(J[f'{SIDE}Hand'][tg])
    pinch = inv0[:3, :3]@(J[f'{SIDE}IndexFingerTip'][tg]-J[f'{SIDE}ThumbTip'][tg])
    dorsal = -(inv0[:3, :3]@palm_normal(J, SIDE, tg))
    P2 = hand_aligned_grasp(g_c, wr_c, pinch, dorsal, n)
    p_h = P2[:3, 3].copy()
    cu_, cv__ = int(fx*g_c[0]/g_c[2] + w/2), int(fy*g_c[1]/g_c[2] + h/2)
    sm_ = [cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0 for m in np.load(I+'/sam3_object_masks.npy') if 0 < m.mean() < 0.15]
    hgt_ = Pc@n + d
    cand_ = [m & ok & (hgt_ > 0.003) & (hgt_ < 0.25) for m in sm_]
    dist_ = [np.linalg.norm(Pc[m][::5] - g_c, axis=1).min() if m.sum() > 30 else 9.0 for m in cand_]
    grasp_shift = np.zeros(3)
    if min(dist_) < 0.08:
        near_ = [j for j in range(len(cand_)) if dist_[j] < min(dist_) + 0.01]
        Og = Pc[cand_[min(near_, key=lambda j: cand_[j].sum())]]                             # the smallest nearby object
        e1 = np.cross(n, [1, 0, 0.]); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
        xy = np.stack([Og@e1, Og@e2], 1); lo = xy.min(0) - 0.02
        ij = ((xy - lo)/0.001).astype(int); grid = np.zeros(ij.max(0) + 41, np.uint8); grid[ij[:, 0], ij[:, 1]] = 1
        grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        er = min(10, int(0.25*min(np.ptp(ij[:, 0]), np.ptp(ij[:, 1]))))
        inside = cv2.erode(grid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*er+1, 2*er+1))) if er > 0 else grid
        c_xy = np.median(xy, 0); p_xy = np.array([g_c@e1, g_c@e2])
        for s_ in np.linspace(0, 1, 101):
            q = p_xy + s_*(c_xy - p_xy); k_ = ((q - lo)/0.001).astype(int)
            if 0 <= k_[0] < inside.shape[0] and 0 <= k_[1] < inside.shape[1] and inside[k_[0], k_[1]]: break
        grasp_shift = (q - p_xy)[0]*e1 + (q - p_xy)[1]*e2
        P2[:3, 3] += grasp_shift
    THIN_H, THIN_TILT_DEG, FINGER_CLEAR, SQUEEZE = 0.03, 20.0, 0.004, 0.15
    pass
    import head2wrist.rendering.gripper_render as GR
    from head2wrist.robot.finger_clearance import finger_pts
    g_close = g_hold = None
    width_ = None
    VAR = json.load(open(I + '/grasp_variant.json')) if os.path.exists(I + '/grasp_variant.json') else {}
    THIN_TILT_DEG = VAR.get('tilt_cap', THIN_TILT_DEG) if 'tilt_cap' in VAR else THIN_TILT_DEG
    MESH = I + '/grasp_object.npz'
    obj_h = None
    tilt_cut = 0.0
    if os.path.exists(MESH) and VAR.get('mesh', True):
        from scipy.spatial.transform import Rotation as Rot_
        Vo = np.load(MESH)['V'].astype(float); hv = Vo @ n + d; obj_h = float(hv.max() - max(hv.min(), 0.0))
        P2[:3, 3] = p_h
        if obj_h < THIN_H and THIN_TILT_DEG is not None:
            zt = P2[:3, 2]; ang = float(np.degrees(np.arccos(np.clip(-(zt @ n), -1, 1))))
            if ang > THIN_TILT_DEG:                    # rotate about the (level) jaw axis: z x (-n) is horizontal, along y
                k_ = np.cross(zt, -n); k_ /= np.linalg.norm(k_); tilt_cut = ang - THIN_TILT_DEG
                P2[:3, :3] = Rot_.from_rotvec(np.radians(tilt_cut)*k_).as_matrix() @ P2[:3, :3]
        Rg = P2[:3, :3]; Vt = (Vo - p_h) @ Rg; lo_, hi_ = Vt.min(0), Vt.max(0); m_ = np.minimum(0.015, 0.5*(hi_ - lo_))
        x_ = float(np.clip(0.0, lo_[0] + m_[0], hi_[0] - m_[0])); sl_ = np.abs(Vt[:, 0] - x_) < 0.01
        if sl_.sum() < 20: sl_ = np.ones(len(Vt), bool)
        y_ = float((Vt[sl_, 1].min() + Vt[sl_, 1].max())/2)
        width_ = float(np.ptp(Vt[sl_, 1])); g_close = GR.joint_for_gap(min(width_, GR.jaw_gap(0.0))); g_hold = min(g_close + SQUEEZE, GR.G_MAX)
        if width_ > GR.jaw_gap(0.0) - 0.01: print(f'WARNING: object {100*width_:.1f} cm across the jaws, opening {100*GR.jaw_gap(0.0):.1f} cm')
        z_ = (0.0 if VAR.get('depth') == 'human' else float(Vt[sl_, 2].max()) - float(finger_pts(g_hold)[:, 2].max()))  # closed tips at the object's far end
        z_ += VAR.get('dz', 0.0)
        P2[:3, 3] = p_h + Rg @ np.array([x_, y_, z_]); grasp_shift = P2[:3, 3] - p_h
        print(f'grasp from the object mesh (height {100*obj_h:.1f} cm, tilt cut {tilt_cut:.0f} deg): moved {100*np.linalg.norm(grasp_shift):.1f} cm '
              f'(pads x {100*x_:+.1f}, closing axis {100*y_:+.1f}, approach {100*z_:+.1f} cm)')
    else:
        print(f'grasp point moved {100*np.linalg.norm(grasp_shift):.1f} cm onto the object')
    pass
    from head2wrist.robot.finger_clearance import raise_for_fingers, support_height
    if os.path.exists(MESH):
        gm_ = cv2.resize(np.load(I + '/grasp_object_mask.npy').astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        grasp_support = support_height(P2[:3, 3], n, d, Pc[ok & ~cv2.dilate(gm_.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)])
    else:
        grasp_support = 0.0
    P2, grasp_raise = raise_for_fingers(P2, n, d, support=grasp_support, clearance=VAR.get('clearance', FINGER_CLEAR),
                                        g=None if VAR.get('clear') == 'open' else g_hold)
    fing = (g_c-wr_c)/np.linalg.norm(g_c-wr_c)
    vf = inv0[:3, :3]@v
    MAX_APPROACH_DEG = 60.0 if (obj_h is None or obj_h >= THIN_H or THIN_TILT_DEG is None) else THIN_TILT_DEG
    app_human_deg = float(np.degrees(np.arccos(np.clip(-vf@n, -1, 1))))
    if app_human_deg > MAX_APPROACH_DEG:
        _hz = vf - (vf@n)*n; _hz /= np.linalg.norm(_hz); _a = np.radians(MAX_APPROACH_DEG)
        vf = -np.cos(_a)*n + np.sin(_a)*_hz
    P1 = P2.copy()
    P1[:3, 3] = P2[:3, 3] - STANDOFF*vf
    DESC = STANDOFF
    end_dir = vf
    if A_.approach == 'direct':
        ax = P2[:3, 2]/np.linalg.norm(P2[:3, 2])                  # the gripper's pointing axis at the grasp
        if ax@n < -0.2:                                           # it points down: the last 6 cm go along it, the jaws come down around the object
            DESC = 0.06; end_dir = ax; P1 = P2.copy(); P1[:3, 3] = P2[:3, 3] - DESC*ax
        else: print(f'gripper axis {np.degrees(np.arccos(np.clip(-ax@n, -1, 1))):.0f} deg from straight down: pre-grasp along the human approach', flush=True)
    P0 = ov_cam @ np.linalg.inv(T_TCP_D405)
    P0[:3, :3] = relevel(P0[:3, :3], n)
    contact = P2[:3, 3]
    sm = np.load(I+'/sam3_object_masks.npy')
    objm = np.zeros((h, w), bool)
    for m in sm:
        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        if 0 < m.mean() < 0.15: objm |= m
    O = Pc[objm & ok]
    Osub = O[::max(1, len(O)//3000)]
    col = cv2.resize(cv2.cvtColor(cv2.imread(I+'/first_frame.png'), cv2.COLOR_BGR2RGB), (w, h), interpolation=cv2.INTER_AREA)
    p0, p1, p2 = P0[:3, 3], P1[:3, 3], P2[:3, 3]
    chord = np.linalg.norm(p1-p0)
    lift_c = 0.0
    tab_need = min(0.05, float(p0@n + d) - 0.005)
    for _ in range(20):
        if A_.approach == 'direct':                               # nearly straight: a short end arm bends it into the final axis
            b1 = p0 + (p1-p0)/3 + lift_c*n; b2 = p1 - end_dir*min(chord/3, 0.03) + lift_c*n*0.5
        else: b1 = p0 + (p1-p0)/3 + lift_c*n; b2 = p1 - vf*chord/3 + lift_c*n*0.5
        s = np.linspace(0, 1, 400)[:, None]
        curve = (1-s)**3*p0 + 3*(1-s)**2*s*b1 + 3*(1-s)*s**2*b2 + s**3*p1
        clear_tab = (curve@n + d).min()
        far = np.linalg.norm(Osub[None]-curve[:, None], axis=2).min(1)
        near_contact = np.linalg.norm(curve - contact, axis=1) < 0.15      # ignore the target itself near the end
        clear_obj = far[~near_contact].min() if (~near_contact).any() else 1.0
        if clear_tab >= tab_need and clear_obj > 0.05: break
        lift_c += 0.02
    else:
        print(f'WARNING: transit clearance not reached (table {100*clear_tab:.1f} cm, objects {100*clear_obj:.1f} cm)')
    line = np.linspace(p1, p2, 60)
    path = np.vstack([curve, line[1:]])
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    L_tr = seg[len(curve)-1]
    L = seg[-1]
    y0 = P0[:3, 1]
    th0, th2 = pitch(P0[:3, :3]), pitch(P2[:3, :3])
    psi2 = np.arctan2(n@np.cross(y0, P2[:3, 1]), y0@P2[:3, 1])
    svals = [0.0]
    while svals[-1] < L - 1e-4:
        sv = svals[-1]; step = vcap(sv)
        for _ in range(12):                                    # shrink the step until rotation limits hold
            nxt = min(L, sv+step); Ra, Rb = ori(sv), ori(nxt)
            fz = np.degrees(np.arccos(np.clip(Ra[:, 2]@Rb[:, 2], -1, 1)))
            tot = np.degrees(R.from_matrix(Ra.T@Rb).magnitude()); roll = np.sqrt(max(tot**2-fz**2, 0))
            # the camera (what Lyra-2 renders) must also move <= V_T per frame: it swings around the TCP when rotating
            Ta = np.eye(4); Ta[:3, :3] = Ra; Ta[:3, 3] = [np.interp(sv, seg, path[:, k]) for k in range(3)]
            Tb = np.eye(4); Tb[:3, :3] = Rb; Tb[:3, 3] = [np.interp(nxt, seg, path[:, k]) for k in range(3)]
            cam_step = np.linalg.norm((Tb @ T_TCP_D405)[:3, 3] - (Ta @ T_TCP_D405)[:3, 3])
            if fz <= ROT_V and roll <= ROT_R and cam_step <= max(vcap(sv), 0.0015)*1.0 + 1e-9 or step < 2e-4: break
            step *= 0.7
        svals.append(nxt)
    svals = np.array(svals)
    TCP = []
    for sv in svals:
        T = np.eye(4); T[:3, :3] = ori(sv); T[:3, 3] = np.array([np.interp(sv, seg, path[:, k]) for k in range(3)]); TCP.append(T)
    TCP = np.stack(TCP)
    corr = np.zeros((len(TCP), 3))
    lost = 0
    for i, T in enumerate(TCP):
        C = cam_of(T); u, v_, z = proj(C, contact)
        if z <= 0.02 or not (0.05*W < u < 0.95*W and 0.05*H < v_ < 0.95*H):
            lost += 1
            a_ = C[:3, 2]; b_ = (contact - C[:3, 3]); b_ /= np.linalg.norm(b_)
            ax = np.cross(a_, b_); s_ = np.linalg.norm(ax)
            if s_ > 1e-8: corr[i] = ax/s_*np.arcsin(min(s_, 1.0))*0.6
    corr = gaussian_filter1d(corr, 4, axis=0)
    for i in range(len(TCP)):
        if np.linalg.norm(corr[i]) > 0: TCP[i][:3, :3] = relevel(R.from_rotvec(corr[i]).as_matrix() @ TCP[i][:3, :3], n)
    CAM = np.stack([cam_of(T) for T in TCP])
    vis = [(0.0 <= proj(C, contact)[0] < W and 0 <= proj(C, contact)[1] < H and proj(C, contact)[2] > 0) for C in CAM]
    st = np.linalg.norm(np.diff(TCP[:, :3, 3], axis=0), axis=1)
    rot = [np.degrees(R.from_matrix(CAM[i][:3, :3].T@CAM[i+1][:3, :3]).magnitude()) for i in range(len(CAM)-1)]
    n_tr = int(np.searchsorted(svals, L_tr))
    np.savez(I+'/approach_plan.npz', tcp=TCP, cam=CAM, n_transit=n_tr, t_grasp=tg, t_release=t_rel, contact=contact, side=SIDE, grasp_shift=grasp_shift,
             **({} if g_close is None else dict(gripper_close=g_close, gripper_hold=g_hold, object_width=width_)))
    dcs = np.linalg.norm(np.diff(CAM[:, :3, 3], axis=0), axis=1)
    info = dict(grasp_variant=VAR.get('name', 'A'), grasp_shift_cm=float(100*np.linalg.norm(grasp_shift)), grasp_from_mesh=bool(obj_h is not None),
                object_width_cm=None if width_ is None else 100*width_, gripper_close=g_close, gripper_hold=g_hold,
                grasp_tcp_raised_for_closed_fingers_cm=100*grasp_raise,
                object_height_cm=None if obj_h is None else 100*obj_h, thin_tilt_cut_deg=tilt_cut, grasp_support_cm=100*grasp_support, approach_human_from_down_deg=app_human_deg, approach_used_from_down_deg=float(np.degrees(np.arccos(np.clip(-vf@n, -1, 1)))), max_cam_step_cm=float(100*dcs.max()), t_grasp=tg, approach_from_vertical_deg=float(np.degrees(np.arccos(np.clip(-v@up_w, -1, 1)))),
                frames_total=len(TCP), frames_transit=n_tr, frames_descent=len(TCP)-n_tr, transit_length_m=float(L_tr),
                clearance_table_m=float(clear_tab), clearance_objects_m=float(clear_obj), control_point_raise_m=float(lift_c),
                grasp_tcp_raised_for_fingers_cm=float(100*grasp_raise), max_step_cm=float(100*st.max()), max_rot_deg_per_frame=float(max(rot)), frames_phone_out_of_view_before_fix=lost,
                phone_in_view_frac=float(np.mean(vis)),
                max_horizon_lean_deg=float(max(np.degrees(np.arcsin(abs(C[:3, 1]@n))) for C in CAM)),
                grasp_approach_from_down_deg=float(np.degrees(np.arccos(np.clip(-P2[:3, 2]@n, -1, 1)))),
                grasp_approach_vs_human_fingers_deg=float(np.degrees(np.arccos(np.clip(P2[:3, 2]@fing, -1, 1)))),
                camera_side_vs_back_of_hand_deg=float(np.degrees(np.arccos(np.clip(P2[:3, 0]@dorsal, -1, 1)))),
                heading_change_deg=float(np.degrees(psi2)), pitch_overview_deg=float(np.degrees(th0)), pitch_grasp_deg=float(np.degrees(th2)))
    json.dump(info, open(I+'/approach_plan.json', 'w'), indent=1)
    print(json.dumps(info))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fwd = np.array([0, 0, 1.])
    ft = fwd-(fwd@n)*n
    ft /= np.linalg.norm(ft)
    rt = np.cross(ft, n)
    if rt[0] < 0: rt = -rt
    tv = lambda p: np.stack([p@rt, p@ft], -1)
    sv_ = lambda p: np.stack([p@ft, p@n + d], -1)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, fn, lab in ((axs[0], tv, ('right (m)', 'away from robot (m)')), (axs[1], sv_, ('away from robot (m)', 'height above table (m)'))):
        Ob = fn(O[::40]); ax.scatter(Ob[:, 0], Ob[:, 1], s=1, c='tab:green', label='objects')
        tp = fn(TCP[:, :3, 3]); cp = fn(CAM[:, :3, 3])
        ax.plot(tp[:, 0], tp[:, 1], '-', c='tab:blue', label='gripper (TCP) path'); ax.plot(cp[:, 0], cp[:, 1], '--', c='tab:red', label='D405 path')
        for p, nm in ((p0, 'P0 overview'), (p1, 'P1 pre-grasp'), (p2, 'P2 grasp')):
            q = fn(p[None])[0]; ax.scatter(*q, s=40, c='k'); ax.annotate(nm, q, fontsize=8)
        ax.set_xlabel(lab[0]); ax.set_ylabel(lab[1]); ax.set_aspect('equal'); ax.grid(alpha=.3)
    axs[0].legend(fontsize=8, loc='lower left')
    axs[0].set_title('top view')
    axs[1].set_title('side view')
    fig.tight_layout()
    fig.savefig(I+'/approach_plan_views.png', dpi=110)
    plt.close(fig)
    Pall = Pc[ok]
    call = col[ok]
    ap_open = min(np.linalg.norm(J[f'{SIDE}IndexFingerTip'][tg-15]-J[f'{SIDE}ThumbTip'][tg-15]), 0.106)
    import subprocess
    tmpv = I+'/approach_preview_tmp.mp4'
    vw = cv2.VideoWriter(tmpv, cv2.VideoWriter_fourcc(*'mp4v'), 16, (H, W))
    for i in range(len(CAM)): vw.write(cv2.cvtColor(draw_view(CAM[i], TCP[i], i), cv2.COLOR_RGB2BGR))
    vw.release()
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', tmpv, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-movflags', '+faststart', I+'/approach_preview_upright.mp4'], check=True)
    os.remove(tmpv)
    for p_ in ('approach_preview.mp4', 'approach_plan_previews.jpg'):                     # old raw-frame previews
        if os.path.exists(I+'/'+p_): os.remove(I+'/'+p_)
    sel = list(np.linspace(0, n_tr, 5).astype(int)[:-1]) + [n_tr, len(CAM)-1]
    tiles = [draw_view(CAM[i], TCP[i], i) for i in sel]
    cv2.imwrite(I+'/approach_preview_upright_stills.jpg', cv2.cvtColor(cv2.resize(np.concatenate(tiles, 1), None, fx=0.45, fy=0.45), cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    main()
