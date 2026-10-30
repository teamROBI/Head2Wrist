from head2wrist.config import H2W_EGODEX_ROOT, H2W_OUTPUT_ROOT
#!/usr/bin/env python3
"""
Draw grasp -> lift -> transport -> place in TCP space, continuing approach_plan.npz (plan_approach.py).
The human path is not copied; its shape is abstracted and a smooth robot path is drawn with the same features:

  from the demo: release time tr (lowest pinch midpoint after the grasp), place point = pinch midpoint at tr,
                 lift-off / set-down directions (first / last 3 cm of height change), arch peak height, top-view bow
                 (signed deviation from the grasp -> place chord), phone heading change (level pinch axis tg -> tr)
  robot path:    lift-off line (4 cm along the human's lift-off direction) -> cubic Bezier whose control points are
                 solved so its peak height and top-view bow match the human's -> set-down line (5 cm along the
                 human's set-down direction) into the place point
  orientation:   jaws level on every frame (level horizon); approach pitch kept at the grasp pitch so the phone stays
                 flat and the gripper stays aligned with the hand direction (grasp from wrist_pose.hand_aligned_grasp);
                 the heading turns when the human's hand turns (the demo turns the phone ~90 deg to fit the lid, most
                 of it in the first half of the carry): smoothed human pinch-axis heading vs progress along the path
  timing:        as the approach: <= 1.2 cm/frame transport, <= 0.8 cm/frame lift-off / set-down, both the TCP and the
                 D405, <= 2.5 deg/frame view change, <= 5 deg/frame roll, rest at the grasp and at the place
Outputs transport_plan.{npz,json}, transport_plan_views.png and an upright preview of the whole sequence
(overview -> place) with the phone moved rigidly with the gripper after the grasp.
"""
import argparse, json, os, sys, subprocess
import numpy as np, cv2, h5py
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d

from head2wrist.wrist_pose import to_upright, T_TCP_D405
from head2wrist.planning.make_hover_tour import fit_plane



def level(p): p = p - (p@n)*n; return p/np.linalg.norm(p)


def bez(a, b):
    b1 = q1 + a*u_up + b*side; b2 = q2 - a*u_dn + b*side
    return (1-s)**3*q1 + 3*(1-s)**2*s*b1 + 3*(1-s)*s**2*b2 + s**3*q2


def feats(c):
    dv = (c-q0)@side; return hgt(c).max(), dv[np.argmax(np.abs(dv))]


def ori(sv):
    psi = np.interp(sv/L, u_h, yaw_h)
    y = np.cos(psi)*P2[:3, 1] + np.sin(psi)*np.cross(n, P2[:3, 1])
    k = np.cross(n, y); z = -np.cos(th)*n + np.sin(th)*k
    return np.stack([np.cross(y, z), y, z], 1)


def proj(C, p):
    E = np.linalg.inv(C); c = E[:3, :3]@p + E[:3, 3]
    return FX*c[0]/c[2]+W/2, FX*c[1]/c[2]+H/2, c[2]


def vcap(sv):
    near = min(max(0, sv-L1), max(0, L2-sv))                                  # distance into the transport
    base = V_D + (V_T-V_D)*min(1.0, near/0.03)
    ramp = min(1.0, np.sqrt(max(sv, 1e-4)/0.03), np.sqrt(max(L-sv, 1e-4)/0.03))
    return max(base*ramp, 0.0015)


def pose(sv):
    T = np.eye(4); T[:3, :3] = ori(sv); T[:3, 3] = [np.interp(sv, seg, path[:, k]) for k in range(3)]; return T


def phone_at(T): return Ophone@(T@rel)[:3, :3].T + (T@rel)[:3, 3]


def phone_layer(C, M):
    E = np.linalg.inv(C) @ M; Hm = K_w @ (E[:3, :3] - np.outer(E[:3, 3], n_p)/d_p) @ np.linalg.inv(K_in)
    return cv2.warpPerspective(img_full, Hm, (W, H), flags=cv2.INTER_LINEAR), cv2.warpPerspective(phone_full_u8, Hm, (W, H), flags=cv2.INTER_NEAREST) > 127


def phase(i):
    if i < n_tr_a: return 'approach'
    if i < na: return 'descent'
    j = i - na + 1
    return 'lift-off' if j < n_up else ('set-down' if j >= len(TCP) - n_dn else 'transport')


def draw(i):
    C, T = ALL_C[i], ALL_T[i]
    c = (static_nophone-C[:3, 3])@C[:3, :3]; m = c[:, 2] > 0.02
    u = (FX*c[m, 0]/c[m, 2]+W/2).astype(int); v = (FX*c[m, 1]/c[m, 2]+H/2).astype(int); zz = c[m, 2]
    ib = (u >= 0) & (u < W) & (v >= 0) & (v < H); od = np.argsort(-zz[ib])
    pv = np.zeros((H, W, 3), np.uint8); pv[v[ib][od], u[ib][od]] = col_nophone[m][ib][od]; pv = cv2.dilate(pv, np.ones((3, 3), np.uint8))
    ph, pm = phone_layer(C, np.eye(4) if i < na else T @ rel); pv[pm] = ph[pm]
    opening = ap_open if i < na - 1 else ap_closed + 0.01
    for sgn in (-1, 1):
        pad = T[:3, 3] + sgn*0.5*opening*T[:3, 1]; root = pad - 0.05*T[:3, 2]
        a_, b_ = proj(C, pad), proj(C, root)
        if a_[2] > 0 and b_[2] > 0: cv2.line(pv, (int(a_[0]), int(a_[1])), (int(b_[0]), int(b_[1])), (0, 255, 255), 4)
    tgt = AP['contact'] if i < na else q3
    uu_, vv_, zc = proj(C, tgt)
    if zc > 0: cv2.circle(pv, (int(uu_), int(vv_)), 10, (255, 255, 0) if i < na else (255, 0, 255), 2)
    pv = np.ascontiguousarray(to_upright(pv))
    cv2.putText(pv, f'{i}/{len(ALL_C)-1} {phase(i)}', (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    return pv


def main():
    global ALL_C, ALL_T, AP, A_, C, CAM, CAMa, D, EP, F0, FX, H, I, J, K_in, K_w, L, L1, L2, L_DN, L_UP, Ob, Oc, Oother, Ophone, P2, Pc, Pk, Ps, RELEASE_GAP, ROT_R, ROT_V, SIDE, T, T3, TCP, TCPa, Ta, Tb, V_D, V_T, Vo_, W, _, _k, _os, _sys, _th, _y3, _z3, a, a_c, ap_, ap_closed, ap_open, ax, axs, b, b_c, best, bow_h, bow_r, bw, cam, cam_step, ch_t, chord, clear, col, col_nophone, container, cp, cu, curve, cv_, d, d_p, dcs, dist, dpsi, dts, dxy, e, e1_, e2_, f, fig, fn, ft, fwd, fx, fy, fz, g, g_hold_, h, hang, headcam, hg, hgt, hits, hk, hp, i, img_full, in_tr, info, inv0, j, j_ph, k, k2, kind, lab, m, masks, masks_full, matplotlib, meta, n, n_dn, n_p, n_tr_a, n_up, na, nm, nophone, nxt, objm, ok, p, path, peak, phone_full, phone_full_u8, phone_m, pinch, pk, pk_r, place_raise, place_support, placed, plt, q, q0, q1, q2, q3, rad_, raise_for_fingers, rel, relP, release_raise, rim, roll, rot, rt, s, s0, seg, sel, sh_, side, static_nophone, step, support_height, surf, sv, sv2, svals, t, t0, t_dn, t_pk, t_up, tf_, tg, th, tmp, tot, tp, tr, tv, u_dn, u_h, u_up, uu, vis, vv, vw, w, yaw_h, yg, yr
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--inputs', default=f'{H2W_OUTPUT_ROOT}/lyra2/explore_inputs/basic_pick_place_1')
    ap_.add_argument('--root', default=f'{H2W_EGODEX_ROOT}')
    A_ = ap_.parse_args()
    I = A_.inputs
    W, H = 848, 480
    FX = (W/2)/np.tan(np.radians(87/2))
    V_T, V_D, ROT_V, ROT_R = 0.012, 0.008, 2.5, 5.0
    L_UP, L_DN = 0.04, 0.05
    meta = json.load(open(I+'/meta.json'))
    t0 = meta['t0']
    EP = os.path.join(A_.root, meta['ep'])
    AP = np.load(I+'/approach_plan.npz')
    TCPa, CAMa = AP['tcp'], AP['cam']
    tg = int(AP['t_grasp'])
    n_tr_a = int(AP['n_transit'])
    SIDE = str(AP['side']) if 'side' in AP else 'right'
    P2 = TCPa[-1]
    f = h5py.File(EP+'.hdf5')
    cam = f['transforms/camera'][:].astype(float)
    inv0 = np.linalg.inv(cam[t0])
    J = {k: f['transforms/'+k][:, :3, 3].astype(float) for k in (f'{SIDE}ThumbTip', f'{SIDE}IndexFingerTip')}
    tf_ = lambda p: p@inv0[:3, :3].T + inv0[:3, 3]
    g = tf_((J[f'{SIDE}ThumbTip']+J[f'{SIDE}IndexFingerTip'])/2)
    pinch = (J[f'{SIDE}IndexFingerTip']-J[f'{SIDE}ThumbTip'])@inv0[:3, :3].T
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
    hgt = lambda p: p@n + d
    cu, cv_ = int(fx*P2[0, 3]/P2[2, 3] + w/2), int(fy*P2[1, 3]/P2[2, 3] + h/2)
    cu, cv_ = min(max(cu, 0), w - 1), min(max(cv_, 0), h - 1)
    masks_full = [m for m in np.load(I+'/sam3_object_masks.npy') if 0 < m.mean() < 0.15]
    masks = [cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0 for m in masks_full]
    hits = [j for j, m in enumerate(masks) if cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8))[cv_, cu]]
    if not hits:          # the grasp point sits a few cm above the table: it projects just past the object's mask edge
        dist = [cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5)[cv_, cu] for m in masks]       # basic_pick_place/73
        hits = [int(np.argmin(dist))] if masks and min(dist) < 60*w/1920 else []
    assert hits, 'no SAM 3 mask at (or within 60 px of) the grasp point'
    j_ph = min(hits, key=lambda j: masks[j].sum())
    phone_full = masks_full[j_ph]
    phone_m = masks[j_ph] & ok & (hgt(Pc) < 0.03)
    if not phone_m.any():             # a taller object (box, cup): none of it within 3 cm of the table (bpp/22) -> the whole object
        phone_m = masks[j_ph] & ok & (hgt(Pc) < 0.25); print(f'grasped object taller than 3 cm: using its points up to 25 cm ({phone_m.sum()})', flush=True)
    assert phone_m.any(), 'the grasped object has no valid depth points in the head frame'
    objm = np.zeros((h, w), bool)
    for m in masks: objm |= m
    Ophone = Pc[phone_m]
    Oother = Pc[objm & ~cv2.dilate(phone_m.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & ok]
    Oother = Oother[::max(1, len(Oother)//3000)]
    hg = hgt(g)
    tr = int(AP['t_release']) if 't_release' in AP else int(tg + 10 + np.argmin(hg[tg+10:tg+50]))
    t_pk = tg + int(np.argmax(hg[tg:tr + 1]))
    t_up = next((t for t in range(tg, tr) if hg[t] >= hg[tg] + 0.03), t_pk)
    t_dn = next((t for t in range(tr, tg, -1) if hg[t] >= hg[tr] + 0.03), t_pk if t_pk < tr else tr - 1)
    u_up = g[t_up]-g[tg]
    u_up /= np.linalg.norm(u_up)
    u_dn = g[tr]-g[t_dn]
    u_dn /= np.linalg.norm(u_dn)
    peak = float(hg[tg:tr+1].max())
    chord = g[tr]-g[tg]
    ch_t = chord - (chord@n)*n
    ch_t /= np.linalg.norm(ch_t)
    side = np.cross(n, ch_t)
    bow_h = float(((g[tg:tr+1]-g[tg])@side)[np.argmax(np.abs((g[tg:tr+1]-g[tg])@side))])
    yg, yr = level(pinch[tg]), level(pinch[tr])
    if yg @ P2[:3, 1] < 0: yg, yr = -yg, -yr
    dpsi = float(np.arctan2(n@np.cross(yg, yr), yg@yr))
    kind = 'arch' if peak - max(hg[tg], hg[tr]) > 0.05 else ('around' if abs(bow_h) > 0.15*np.linalg.norm(chord) else 'straight')
    pass
    from head2wrist.robot.finger_clearance import raise_for_fingers, support_height
    _k = np.cross(n, P2[:3, 1])
    _th = np.arctan2(P2[:3, 2]@_k, -(P2[:3, 2]@n))
    _y3 = np.cos(dpsi)*P2[:3, 1] + np.sin(dpsi)*np.cross(n, P2[:3, 1])
    _z3 = -np.cos(_th)*n + np.sin(_th)*np.cross(n, _y3)
    T3 = np.eye(4)
    T3[:3, :3] = np.stack([np.cross(_y3, _z3), _y3, _z3], 1)
    T3[:3, 3] = g[tr]
    sh_ = AP['grasp_shift'] if 'grasp_shift' in AP else np.zeros(3)
    T3[:3, 3] += np.cos(dpsi)*sh_ + np.sin(dpsi)*np.cross(n, sh_) + (1 - np.cos(dpsi))*(sh_ @ n)*n
    place_support = support_height(T3[:3, 3], n, d, Pc[ok])
    g_hold_ = float(AP['gripper_hold']) if 'gripper_hold' in AP else None
    T3, place_raise = raise_for_fingers(T3, n, d, support=place_support, g=g_hold_)
    RELEASE_GAP = 0.01
    release_raise = 0.0
    if os.path.exists(I + '/grasp_object.npz'):
        Vo_ = np.load(I + '/grasp_object.npz')['V'].astype(float)
        hang = float((P2[:3, 3] @ n + d) - (Vo_ @ n + d).min())                             # object bottom below the TCP
        # into a container (a rim around the release point higher than its floor by > 2 cm: the bowls of
        # basic_pick_place/74 and /73, pushed by the fingers lowered inside): release above the rim, drop it in
        e1_ = np.cross(n, [1, 0, 0.]); e1_ /= np.linalg.norm(e1_); e2_ = np.cross(n, e1_); Pk = Pc[ok]
        rad_ = 0.5*float(np.ptp(np.stack([Vo_ @ e1_, Vo_ @ e2_], 1), 0).max()) + 0.02
        dxy = np.linalg.norm(np.stack([(Pk - T3[:3, 3]) @ e1_, (Pk - T3[:3, 3]) @ e2_], 1), axis=1)
        hk = Pk[dxy < rad_] @ n + d; rim = float(np.percentile(hk, 95)) if len(hk) > 50 else place_support
        container = rim - place_support > 0.02
        surf = rim if container else place_support
        release_raise = max(0.0, surf + RELEASE_GAP + hang - float(T3[:3, 3] @ n + d))
        T3[:3, 3] += release_raise*n
        if container: T3, _ = raise_for_fingers(T3, n, d, support=rim, g=g_hold_)                 # the closed tips too
        print(f'release: object bottom {100*hang:.1f} cm below the TCP, support {100*place_support:.1f} cm'
              f'{f", container rim {100*rim:.1f} cm" if container else ""} -> raised {100*release_raise:.1f} cm')
    q0 = P2[:3, 3]
    q1 = q0 + L_UP*u_up
    q3 = T3[:3, 3].copy()
    q2 = q3 - L_DN*u_dn
    s = np.linspace(0, 1, 400)[:, None]
    best = None
    for a in np.linspace(0.0, 0.4, 41):
        for b in np.linspace(-0.2, 0.2, 41):
            pk, bw = feats(bez(a, b)); e = (pk-peak)**2 + (bw-bow_h)**2
            if best is None or e < best[0]: best = (e, a, b)
    _, a_c, b_c = best
    curve = bez(a_c, b_c)
    pk_r, bow_r = feats(curve)
    path = np.vstack([np.linspace(q0, q1, 20)[:-1], curve, np.linspace(q2, q3, 25)[1:]])
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    L = seg[-1]
    L1 = np.linalg.norm(q1-q0)
    L2 = L - np.linalg.norm(q3-q2)
    k2 = np.cross(n, P2[:3, 1])
    th = np.arctan2(P2[:3, 2]@k2, -(P2[:3, 2]@n))
    s0 = 1.0 if level(pinch[tg]) @ yg > 0 else -1.0
    yaw_h = np.array([np.arctan2(n@np.cross(yg, s0*level(pinch[t])), yg@(s0*level(pinch[t]))) for t in range(tg, tr+1)])
    yaw_h = gaussian_filter1d(np.unwrap(yaw_h), 3, mode='nearest')
    yaw_h += np.linspace(0, 1, len(yaw_h))*(dpsi - yaw_h[-1])
    u_h = np.r_[0, np.cumsum(np.linalg.norm(np.diff(g[tg:tr+1], axis=0), axis=1))]
    u_h /= u_h[-1]
    svals = [0.0]
    while svals[-1] < L - 1e-4:
        sv = svals[-1]; step = vcap(sv)
        for _ in range(12):
            nxt = min(L, sv+step); Ta, Tb = pose(sv), pose(nxt)
            fz = np.degrees(np.arccos(np.clip(Ta[:3, 2]@Tb[:3, 2], -1, 1)))
            tot = np.degrees(R.from_matrix(Ta[:3, :3].T@Tb[:3, :3]).magnitude()); roll = np.sqrt(max(tot**2-fz**2, 0))
            cam_step = np.linalg.norm((Tb@T_TCP_D405)[:3, 3]-(Ta@T_TCP_D405)[:3, 3])
            if fz <= ROT_V and roll <= ROT_R and cam_step <= vcap(sv) + 1e-9 or step < 2e-4: break
            step *= 0.7
        svals.append(nxt)
    svals = np.array(svals)
    TCP = np.stack([pose(sv) for sv in svals])
    CAM = np.stack([T@T_TCP_D405 for T in TCP])
    in_tr = (svals > L1) & (svals < L2)
    vis = [(0 <= proj(C, q3)[0] < W and 0 <= proj(C, q3)[1] < H and proj(C, q3)[2] > 0) for C in CAM]
    rel = np.linalg.inv(P2)
    Ps = Ophone[::max(1, len(Ophone)//600)]
    relP = lambda T: Ps@(T@rel)[:3, :3].T + (T@rel)[:3, 3]
    clear = [np.linalg.norm(Oother[None]-relP(T)[:, None], axis=2).min() for T, sv in zip(TCP, svals) if L1 < sv < L2 - 0.02]\
        if len(Oother) and len(Ps) else []
    placed = phone_at(TCP[-1])
    dcs = np.linalg.norm(np.diff(CAM[:, :3, 3], axis=0), axis=1)
    dts = np.linalg.norm(np.diff(TCP[:, :3, 3], axis=0), axis=1)
    rot = [np.degrees(R.from_matrix(CAM[i][:3, :3].T@CAM[i+1][:3, :3]).magnitude()) for i in range(len(CAM)-1)]
    n_up, n_dn = int(np.searchsorted(svals, L1)), int(len(svals) - np.searchsorted(svals, L2))
    np.savez(I+'/transport_plan.npz', tcp=TCP, cam=CAM, t_release=tr, place=q3, n_liftoff=n_up, n_setdown=n_dn)
    info = dict(t_grasp=tg, t_release=tr, place_support_height_cm=100*place_support, release_raise_cm=100*release_raise, place_tcp_raised_for_fingers_cm=100*place_raise, human_path=kind, human_peak_height_cm=100*peak, robot_peak_height_cm=100*float(pk_r),
                human_bow_cm=100*bow_h, robot_bow_cm=100*float(bow_r), heading_change_deg=float(np.degrees(dpsi)),
                liftoff_from_vertical_deg=float(np.degrees(np.arccos(np.clip(u_up@n, -1, 1)))),
                setdown_from_vertical_deg=float(np.degrees(np.arccos(np.clip(-u_dn@n, -1, 1)))),
                path_length_m=float(L), frames=len(TCP), frames_liftoff=n_up, frames_setdown=n_dn,
                max_tcp_step_cm=100*float(dts.max()), max_cam_step_cm=100*float(dcs.max()), max_rot_deg_per_frame=float(max(rot)),
                max_horizon_lean_deg=float(max(np.degrees(np.arcsin(abs(C[:3, 1]@n))) for C in CAM)),
                place_in_view_frac=float(np.mean(vis)), place_in_view_frac_transport=float(np.mean(np.array(vis)[in_tr])),
                held_phone_clearance_cm=100*float(min(clear)) if len(clear) else None, placed_phone_height_above_table_cm=[100*float(hgt(placed).min()), 100*float(hgt(placed).max())])
    json.dump(info, open(I+'/transport_plan.json', 'w'), indent=1)
    print(json.dumps(info))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fwd = np.array([0, 0, 1.])
    ft = fwd-(fwd@n)*n
    ft /= np.linalg.norm(ft)
    rt = np.cross(ft, n)
    rt = -rt if rt[0] < 0 else rt
    tv = lambda p: np.stack([p@rt, p@ft], -1)
    sv2 = lambda p: np.stack([(p-q0)@ch_t, hgt(p)], -1)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ax, fn, lab in ((axs[0], tv, ('right (m)', 'away from robot (m)')), (axs[1], sv2, ('along grasp -> place (m)', 'height above table (m)'))):
        Ob = fn(Pc[objm & ok][::40]); ax.scatter(Ob[:, 0], Ob[:, 1], s=1, c='tab:green', label='objects')
        hp = fn(g[tg:tr+1]); ax.plot(hp[:, 0], hp[:, 1], ':', c='k', label='human pinch point')
        ap_ = fn(TCPa[:, :3, 3]); ax.plot(ap_[:, 0], ap_[:, 1], '-', c='tab:gray', lw=1, label='approach (TCP)')
        tp = fn(TCP[:, :3, 3]); cp = fn(CAM[:, :3, 3])
        ax.plot(tp[:, 0], tp[:, 1], '-', c='tab:blue', label='transport (TCP)'); ax.plot(cp[:, 0], cp[:, 1], '--', c='tab:red', label='D405')
        for p, nm in ((q0, 'grasp'), (q1, 'lift-off'), (q2, 'pre-place'), (q3, 'place')):
            q = fn(p[None])[0]; ax.scatter(*q, s=30, c='k'); ax.annotate(nm, q, fontsize=8)
        ax.set_xlabel(lab[0]); ax.set_ylabel(lab[1]); ax.set_aspect('equal'); ax.grid(alpha=.3)
    axs[0].legend(fontsize=7, loc='lower left')
    axs[0].set_title('top view')
    axs[1].set_title(f'side view (human path: {kind})')
    fig.tight_layout()
    fig.savefig(I+'/transport_plan_views.png', dpi=110)
    plt.close(fig)
    img_full = cv2.cvtColor(cv2.imread(I+'/first_frame.png'), cv2.COLOR_BGR2RGB)
    col = cv2.resize(img_full, (w, h), interpolation=cv2.INTER_AREA)
    nophone = ok & ~cv2.dilate(masks[j_ph].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    static_nophone, col_nophone = Pc[nophone], col[nophone]
    Oc = Ophone.mean(0)
    n_p = np.linalg.svd(Ophone - Oc)[2][2]
    d_p = -n_p@Oc
    K_in = np.array([[F0, 0, 960], [0, F0, 540], [0, 0, 1.]])
    K_w = np.array([[FX, 0, W/2], [0, FX, H/2], [0, 0, 1.]])
    phone_full_u8 = (phone_full > 0).astype(np.uint8)*255
    ap_open = min(np.linalg.norm(J[f'{SIDE}IndexFingerTip'][tg-15]-J[f'{SIDE}ThumbTip'][tg-15]), 0.106)
    ap_closed = float(np.ptp((Ophone-q0)@P2[:3, 1]))
    ALL_T = np.concatenate([TCPa, TCP[1:]])
    ALL_C = np.concatenate([CAMa, CAM[1:]])
    na = len(TCPa)
    tmp = I+'/pickplace_tmp.mp4'
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), 16, (H, W))
    for i in range(len(ALL_C)): vw.write(cv2.cvtColor(draw(i), cv2.COLOR_RGB2BGR))
    vw.release()
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', tmp, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-movflags', '+faststart',
                    I+'/pickplace_preview_upright.mp4'], check=True)
    os.remove(tmp)
    sel = [0, n_tr_a, na-1, na-1+n_up, na-1+len(TCP)//3, na-1+2*len(TCP)//3, na-1+len(TCP)-n_dn, len(ALL_C)-1]
    cv2.imwrite(I+'/pickplace_preview_upright_stills.jpg',
                cv2.cvtColor(cv2.resize(np.concatenate([draw(i) for i in sel], 1), None, fx=0.4, fy=0.4), cv2.COLOR_RGB2BGR))
    print('total frames overview -> place:', len(ALL_C))


if __name__ == '__main__':
    main()
