from head2wrist.config import H2W_OUTPUT_ROOT
#!/usr/bin/env python3
"""
Placing the robot at the planned stance by hand: what the robot's head camera (ZED, left) SHOULD see at the planned base
pose, rendered from the demo's own input frame (the human head frame + its metric depth, the generator's input: no robot
pre-scan), over what it DOES see (a bridge snap: egoh2w_bridge.py --mode snap). When the two line up, the base is where
the plan put it. Where the scene has texture, the live frame is also registered to the demo frame (SIFT + PnP on the
metric depth) and the base error is estimated: forward / left / yaw in the planned base frame.
  placement_view.py --ep <task/k> --snap <snap.npz> [--out <dir>]
  sim test: placement_view.py --ep sim/stack_18 --sim_demo <demos npz> --sim_base '<4x4 json>' (the head frame + base
            of a recorded Isaac demo; the sim ZED is 336x188, f 166.8, head pitch 0.549)
Head camera = URDF chain to head_link2 (lift, head_joint1/2 from the snap) + the zedm mount (ffw_follower_body.xacro:
zedm_joint, zedm.urdf.xacro: centre +1.325 cm z, left camera +3.15 cm y, optical frame).
Writes <out>/placement.jpg (live | expected | blend with the expected view's edges) and placement.json.
"""
import argparse, json, os, sys
import numpy as np, cv2, pinocchio as pin
from scipy.spatial.transform import Rotation as R
HERE = os.path.dirname(os.path.abspath(__file__)); pass; pass
import head2wrist.robot.sg2_kin as K; import head2wrist.headcam as headcam



def T_xyz(x, y, z, rpy=(0, 0, 0)):
    T = np.eye(4); T[:3, :3] = R.from_euler('xyz', rpy).as_matrix(); T[:3, 3] = [x, y, z]; return T


def base_T_zed(lift, h1, h2):
    q = pin.neutral(K.FULL); jid = lambda n: K.FULL.idx_qs[K.FULL.getJointId(n)]
    q[jid('lift_joint')] = lift; q[jid('head_joint1')] = h1; q[jid('head_joint2')] = h2
    d = K.FULL.createData(); pin.framesForwardKinematics(K.FULL, d, q); return d.oMf[K.FULL.getFrameId('head_link2')].homogeneous @ MOUNT


def render(T_w_cam):
    """z-buffered splat of the world cloud into the ZED at T_w_cam"""
    Ti = np.linalg.inv(T_w_cam); P = Pw @ Ti[:3, :3].T + Ti[:3, 3]; z = P[:, 2]; m = z > 0.05
    u = np.round(Kz[0, 0]*P[m, 0]/z[m] + Kz[0, 2]).astype(int); v = np.round(Kz[1, 1]*P[m, 1]/z[m] + Kz[1, 2]).astype(int); c = col[m]; zz = z[m]
    out_ = np.zeros((Hz, Wz, 3), np.uint8); zb = np.full((Hz, Wz), np.inf); r = max(1, int(round(Wz/400)))
    order = np.argsort(-zz)                                             # far first, near overwrite
    for du in range(-r, r + 1):
        for dv in range(-r, r + 1):
            uu_, vv_ = u[order] + du, v[order] + dv; k = (uu_ >= 0) & (uu_ < Wz) & (vv_ >= 0) & (vv_ < Hz)
            out_[vv_[k], uu_[k]] = c[order][k]; zb[vv_[k], uu_[k]] = np.minimum(zb[vv_[k], uu_[k]], zz[order][k])
    return out_


def main():
    global B, D, E, F0, Hz, I, Kz, MOUNT, Pc, Pw, S, T, TW, TWP, Tb_act, Tb_plan, Tb_true, Thw, Wz, X3, Z, a, ap, b2, blend, col, dep, dh, dist, dl, e2, edges, est, et, exp_, f, fx, fy, g, ge_, gh, gl, gl_, gp_, h, h1, h2, i, img, inl, jv, kh, kl, lift, live, m_, msg, mt, n_, name, names, ok, okp, out, panels, pos, res, rv, sheet, sift, sx, t, titles, tv, up, uu, uvh, uvl, vv, w, x
    ap = argparse.ArgumentParser()
    ap.add_argument('--ep', required=True)
    ap.add_argument('--snap', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--twin', default=None, help='default: sim_twin/<name>/twin.json (sim/<scene> -> the scene\'s twin)')
    ap.add_argument('--sim_demo', default=None)
    ap.add_argument('--sim_frame', type=int, default=0)
    ap.add_argument('--sim_base', default=None)
    ap.add_argument('--sim_lift', type=float, default=-0.1)
    ap.add_argument('--min_inliers', type=int, default=30)
    a = ap.parse_args()
    name = a.ep.replace('/', '_')
    I = f'{H2W_OUTPUT_ROOT}/lyra2/explore_inputs/{name}'
    TWP = a.twin or (f'{H2W_OUTPUT_ROOT}/sim_twin/{a.ep.split("/", 1)[1]}/twin.json' if a.ep.startswith('sim/') else f'{H2W_OUTPUT_ROOT}/sim_twin/{name}/twin.json')
    TW = json.load(open(TWP))
    Thw = np.array(TW['human_head_world'], float)
    Tb_plan = np.array(TW['robot']['base_pose'], float)
    out = a.out or os.path.dirname(os.path.abspath(a.snap or a.sim_demo))
    os.makedirs(out, exist_ok=True)
    MOUNT = T_xyz(0.0238122, -0.00651797, -0.0242094) @ T_xyz(0, 0, 0.01325) @ T_xyz(0, 0.0315, 0) @ T_xyz(0, 0, 0, (-np.pi/2, 0, -np.pi/2))
    if a.snap:
        S = np.load(a.snap); live = S['image']; Kz = S['K'].astype(float); names = [str(x) for x in S['joint_names']]; pos = S['joint_pos']
        jv = lambda n, d=0.0: float(pos[names.index(n)]) if n in names else d
        lift, h1, h2 = jv('lift_joint', TW['robot']['lift']), jv('head_joint1'), jv('head_joint2')
    else:
        Z = np.load(a.sim_demo); live = Z['head'][a.sim_frame]; f = 10.4/20.955*live.shape[1]; Kz = np.array([[f, 0, live.shape[1]/2], [0, f, live.shape[0]/2], [0, 0, 1.]])
        lift, h1, h2 = a.sim_lift, 0.549, 0.0
    Hz, Wz = live.shape[:2]
    B = base_T_zed(lift, h1, h2)
    img = cv2.cvtColor(cv2.imread(f'{I}/first_frame.png'), cv2.COLOR_BGR2RGB)
    D = np.load(f'{I}/first_depth.npy')
    h, w = D.shape
    F0 = headcam.focal(I)
    fx, fy = F0*w/1920, F0*h/1080
    vv, uu = np.mgrid[0:h, 0:w]
    ok = (D > 0.05) & (D < 4)
    Pc = np.stack([(uu - w/2)/fx*D, (vv - h/2)/fy*D, D], -1)[ok]
    col = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)[ok]
    Pw = Pc @ Thw[:3, :3].T + Thw[:3, 3]
    exp_ = render(Tb_plan @ B)
    res = dict(ep=a.ep, lift=lift, head=[h1, h2], planned_base=Tb_plan.tolist())
    sift = cv2.SIFT_create(nfeatures=8000)
    gh = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kh, dh = sift.detectAndCompute(gh, None)
    up = max(1.0, 1280/Wz)
    gl = cv2.resize(cv2.cvtColor(live, cv2.COLOR_RGB2GRAY), None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
    kl, dl = sift.detectAndCompute(gl, None)
    est = None
    if dh is not None and dl is not None and len(kh) > 8 and len(kl) > 8:
        mt = [m_ for m_, n_ in cv2.BFMatcher().knnMatch(dl, dh, k=2) if m_.distance < 0.8*n_.distance]
        if len(mt) >= 8:
            uvh = np.array([kh[m_.trainIdx].pt for m_ in mt]); dep = D[np.clip((uvh[:, 1]*h/img.shape[0]).astype(int), 0, h - 1), np.clip((uvh[:, 0]*w/img.shape[1]).astype(int), 0, w - 1)]
            sx = w/img.shape[1]; X3 = np.c_[(uvh[:, 0]*sx - w/2)/fx*dep, (uvh[:, 1]*sx - h/2)/fy*dep, dep]; uvl = np.array([kl[m_.queryIdx].pt for m_ in mt])/up; g = dep > 0.05
            if g.sum() >= 8:
                okp, rv, tv, inl = cv2.solvePnPRansac(X3[g], uvl[g], Kz, None, reprojectionError=max(3.0, Wz/200), iterationsCount=5000, flags=cv2.SOLVEPNP_SQPNP)
                res.update(matches=int(len(mt)), inliers=int(0 if inl is None else len(inl)))
                if okp and inl is not None and len(inl) >= a.min_inliers:
                    T = np.eye(4); T[:3, :3] = cv2.Rodrigues(rv)[0]; T[:3, 3] = tv[:, 0]                 # demo camera -> live camera
                    Tb_act = Thw @ np.linalg.inv(T) @ np.linalg.inv(B); E = np.linalg.inv(Tb_plan) @ Tb_act
                    est = dict(forward_cm=100*E[0, 3], left_cm=100*E[1, 3], up_cm=100*E[2, 3], yaw_deg=float(np.degrees(np.arctan2(E[1, 0], E[0, 0]))), actual_base=Tb_act.tolist())
                    res['estimate'] = est
    if est:
        msg = (f'the base is {est["forward_cm"]:+.1f} cm forward, {est["left_cm"]:+.1f} cm left, {est["yaw_deg"]:+.1f} deg (CCW) of the plan '
               f'({res["inliers"]} inliers): move it {-est["forward_cm"]:+.1f} cm forward, {-est["left_cm"]:+.1f} cm left, turn {-est["yaw_deg"]:+.1f} deg'
               + ('  -> within the trained range' if abs(est['forward_cm']) < 3 and abs(est['left_cm']) < 3 and abs(est['yaw_deg']) < 5 else ''))
    else: msg = f'no reliable registration ({res.get("inliers", 0)} inliers of {res.get("matches", 0)} matches): align by the overlay'
    print(msg)
    res['message'] = msg
    edges = cv2.Canny(cv2.cvtColor(exp_, cv2.COLOR_RGB2GRAY), 40, 120)
    blend = (0.5*live + 0.5*exp_).astype(np.uint8)
    blend[edges > 0] = (0, 255, 0)
    panels, titles = [live, exp_, blend], ['live head camera', 'expected at the planned stance', 'overlay (green: expected edges)']
    if a.sim_base:                                 # sim test: the expected view at the demo's TRUE base should line up with the live frame
        Tb_true = np.array(json.loads(a.sim_base), float); et = render(Tb_true @ B); e2 = cv2.Canny(cv2.cvtColor(et, cv2.COLOR_RGB2GRAY), 40, 120)
        b2 = (0.5*live + 0.5*et).astype(np.uint8); b2[e2 > 0] = (0, 255, 0); panels.append(b2); titles.append('overlay at the TRUE base')
        E = np.linalg.inv(Tb_plan) @ Tb_true; res['true_offset'] = dict(forward_cm=100*E[0, 3], left_cm=100*E[1, 3], yaw_deg=float(np.degrees(np.arctan2(E[1, 0], E[0, 0]))))
        gl_, ge_, gp_ = [cv2.Canny(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), 40, 120) > 0 for x in (live, et, exp_)]
        dist = cv2.distanceTransform((~gl_).astype(np.uint8), cv2.DIST_L2, 3)      # mean distance of expected edges to live edges
        res['edge_dist_true_px'] = float(dist[ge_].mean()) if ge_.any() else None; res['edge_dist_planned_px'] = float(dist[gp_].mean()) if gp_.any() else None
        print(f'sim: true offset {res["true_offset"]}; expected-to-live edge distance at the true base {res["edge_dist_true_px"]:.1f} px, at the planned base {res["edge_dist_planned_px"]:.1f} px')
    sheet = np.concatenate(panels, 1)
    for i, t in enumerate(titles):
        cv2.putText(sheet, t, (8 + i*Wz, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5*max(1, Wz/640), (255, 255, 0), 1)
    cv2.imwrite(f'{out}/placement.jpg', cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    json.dump(res, open(f'{out}/placement.json', 'w'), indent=1)
    cv2.imwrite(f'{out}/expected_head.png', cv2.cvtColor(exp_, cv2.COLOR_RGB2BGR))
    print(f'-> {out}/placement.jpg')


if __name__ == '__main__':
    main()
