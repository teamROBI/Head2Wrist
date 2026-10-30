from head2wrist.config import H2W_DATA_ROOT, H2W_OUTPUT_ROOT
#!/usr/bin/env python3
"""
Complete wrist-view clip for one demo: GEN3C background + the grasped object (SAM 3D mesh) + the real-textured
RH-P12-RN gripper, at the D405 stream's 424x240, from the overview pose (end of the dive) to the end of the release.
  background  GEN3C frames (task_<ep>_dense/gen3c.mp4), per-axis resize of the same camera to 424x240
  scene depth ray-cast of a grid mesh of the head frame's depth with the object removed (task_inputs first_depth,
              edges cut at depth jumps): lets the bowl rim hide the block
  object      object_mesh.npz (refine_object.py) in the head camera frame; at rest until the jaws close on it, then
              rigid with the TCP until the jaws open at release, then left where it was released, dropped onto the
              surface under it (the place pose was raised for finger clearance)
  gripper     gripper_render.render_textured; opening: pre-grasp gap = object width across the jaws + 4 cm (<= fully
              open), closes to the object width over the first 8 grasp-hold frames, opens again over the first 8
              release frames
  shading     vertex colours (calibrated to the input frame in refine_object) x (0.65 + 0.35 |N.up|), light from above
  shadow      soft darkening under the object while it is within 4 cm of the surface (light from above); with a light
              in task_plan.npz (gen_episode --shadow) also the gripper's, the arm's and the held object's cast shadows
Writes $H2W_OUTPUT_ROOT/composite/<ep>/: wrist_upright.mp4 (the clip), layers_upright.mp4
(background | + object | + gripper), stills.jpg, clip.npz (frames raw BGR, gripper joint, object pose per frame).
"""
import argparse, json, os, subprocess, sys
import numpy as np, cv2, open3d as o3d
HERE = os.path.dirname(os.path.abspath(__file__))
import head2wrist.rendering.gripper_render as GR
from head2wrist.robot.finger_clearance import support_height



def rays(c2w, K, ss=1):
    Ws, Hs = W*ss, H*ss; Ks = K.copy(); Ks[:2] *= ss
    y, x = np.mgrid[0:Hs, 0:Ws]; dc = np.stack([(x + 0.5 - Ks[0, 2])/Ks[0, 0], (y + 0.5 - Ks[1, 2])/Ks[1, 1], np.ones_like(x, float)], -1)
    dw = dc @ c2w[:3, :3].T; o = np.broadcast_to(c2w[:3, 3], dw.shape)
    return np.concatenate([o, dw], -1).reshape(-1, 6).astype(np.float32), (Hs, Ws)


def scene_depth(c2w, K):
    r, sh = rays(c2w, K); return scene.cast_rays(o3d.core.Tensor(r))['t_hit'].numpy().reshape(sh)


def render_object(Tobj, c2w, K, ss=2):
    """object at pose Tobj (4x4 applied to the rest-pose vertices) -> rgb (H,W,3) BGR float, alpha, z-depth"""
    V = Vo @ Tobj[:3, :3].T + Tobj[:3, 3]
    sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(Fo.astype(np.uint32)))
    r, (Hs, Ws) = rays(c2w, K, ss); res = sc.cast_rays(o3d.core.Tensor(r))
    t = res['t_hit'].numpy().reshape(Hs, Ws); hit = np.isfinite(t)
    pid = res['primitive_ids'].numpy().reshape(Hs, Ws); uv = res['primitive_uvs'].numpy().reshape(Hs, Ws, 2)
    img = np.zeros((Hs, Ws, 3), np.float32)
    if hit.any():
        f = Fo[pid[hit]]; b1, b2 = uv[hit][:, 0:1], uv[hit][:, 1:2]
        shd = 0.65 + 0.35*np.abs(res['primitive_normals'].numpy().reshape(Hs, Ws, 3)[hit] @ n)            # light from above
        img[hit] = (((1 - b1 - b2)*Co[f[:, 0]] + b1*Co[f[:, 1]] + b2*Co[f[:, 2]])*shd[:, None])[:, ::-1]     # RGB -> BGR
    al = cv2.resize(hit.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    rgb = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)/np.maximum(al[..., None], 1e-6)
    dep = cv2.resize(np.where(hit, t, 0).astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)/np.maximum(al, 1e-6)
    return rgb, al, np.where(al > 0.01, dep, np.inf)


def shadow_factor(i, c2w, K, dep):
    """1 - strength x (fraction of the penumbra cone the gripper, the active arm or the held object blocks), per pixel
    of the visible surface (depth dep), computed at half resolution"""
    sc = o3d.t.geometry.RaycastingScene(); add = lambda V, F: sc.add_triangles(o3d.core.Tensor(np.asarray(V, np.float32)), o3d.core.Tensor(np.asarray(F).astype(np.uint32)))
    Tp = GR.link_poses_tcp('right', float(g[i]))
    for l in GR.LINKS:
        m = GR._MESH[f'gripper_r_{l}']; Tl = TCP[i] @ Tp[l]; add(np.asarray(m.vertices) @ Tl[:3, :3].T + Tl[:3, 3], m.faces)
    dF = SK.FULL.createData(); pin.framesForwardKinematics(SK.FULL, dF, RBD.full_q(side, SH['q'][i], SH['lift']))
    for nm, V, F in RBD.arm_meshes(side):
        Tl = SH['Thb'] @ dF.oMf[SK.FULL.getFrameId(nm)].homogeneous; add(V @ Tl[:3, :3].T + Tl[:3, 3], F)
    add(Vo @ Tobj[i][:3, :3].T + Tobj[i][:3, 3], Fo)
    Ws, Hs = W//2, H//2; Ks = K.copy(); Ks[:2] *= 0.5; d_ = cv2.resize(dep.astype(np.float32), (Ws, Hs), interpolation=cv2.INTER_NEAREST)
    y, x = np.mgrid[0:Hs, 0:Ws]; dc = np.stack([(x + 0.5 - Ks[0, 2])/Ks[0, 0], (y + 0.5 - Ks[1, 2])/Ks[1, 1], np.ones_like(x, float)], -1)
    ok = np.isfinite(d_) & (d_ > 0.01) & (d_ < 5); X = (dc*np.where(ok, d_, 0)[..., None]) @ c2w[:3, :3].T + c2w[:3, 3]
    sh = np.zeros((Hs, Ws), np.float32)
    for dL in SH['dirs']:
        o = X + 0.004*dL; r = np.concatenate([o, np.broadcast_to(dL, o.shape)], -1).reshape(-1, 6).astype(np.float32)
        sh += np.isfinite(sc.cast_rays(o3d.core.Tensor(r))['t_hit'].numpy().reshape(Hs, Ws))
    sh = cv2.GaussianBlur(cv2.resize(np.where(ok, sh/len(SH['dirs']), 0).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR), (0, 0), 1.5)
    return 1 - SH['s']*sh


def write(path, fr):
    tmp = path + '.tmp.mp4'; vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), a.fps, (fr[0].shape[1], fr[0].shape[0]))
    for f in fr: vw.write(f)
    vw.release(); subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', tmp, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', path], check=True); os.remove(tmp)


def main():
    global B, BG, BODY, C2W, Co, DIVE, Dh, Dx, F0, Fo, H, I, K, Ktr, Lu, OUT, P, Ph, Px, RBD, SH, SK, T, TCP, Tc_, Td, Tg, Tobj, Ts, Vi, Vo, Vrel, Vt, W, _ap, _os, _sys, a, ap, ax, ba, band, bg, blur, bx, bz, c2w, cap, cen, comp, d, drop, dvis, dz, f, foot, frames, full, fx, fy, g, g_close, g_open, gal, grgb, gz, h, headcam, hi_, hmin, hull, i, i_attach, i_detach, i_grasp, i_rel, idx, k, l, layers, lo_, m_, n, name, oal, ob, ok, okp, om, orgb, oz, p, pc, pen, phase, pin, q, r, rdir, rec, scene, sel, shd, side, sm, ss_, sup, t, tr, tri, up, uu, uvp, vv, w, wc, width, with_obj, x_, yc, zpl, zs
    ap = argparse.ArgumentParser()
    ap.add_argument('--ep', required=True)
    ap.add_argument('--bg', default=None, help='background video (default: the dense GEN3C run)')
    ap.add_argument('--fps', type=float, default=16.0)
    ap.add_argument('--gripper_tex', default=f'{H2W_DATA_ROOT}/assets/gripper_texture.npz')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    name = a.ep.replace('/', '_')
    I = f'{H2W_OUTPUT_ROOT}/lyra2/explore_inputs/{name}'
    T = f'{H2W_OUTPUT_ROOT}/lyra2/task_inputs/{name}'
    BG = a.bg or f'{H2W_OUTPUT_ROOT}/gen3c/task_{name}_dense/gen3c.mp4'
    OUT = f'{H2W_OUTPUT_ROOT}/composite/{name}{a.tag}'
    os.makedirs(OUT, exist_ok=True)
    W, H = 424, 240
    side = json.load(open(f'{T}/meta.json')).get('side', 'right')
    P = np.load(f'{T}/task_plan.npz')
    C2W, TCP, phase = P['c2w'], P['tcp'], list(P['phase'])
    DIVE = int(P['dive_frames'])
    tr = np.load(f'{T}/trajectory.npz')
    Ktr = tr['intrinsics'].astype(np.float64).copy()
    Ktr[:, 0] *= W/float(tr['image_width'])
    Ktr[:, 1] *= H/float(tr['image_height'])
    cap = cv2.VideoCapture(BG)
    bg = []
    while True:
        ok, f = cap.read()
        if not ok: break
        bg.append(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA))
    if len(bg) == len(C2W) - DIVE: bg = [None]*DIVE + bg
    assert len(bg) == len(C2W), (len(bg), len(C2W))
    import os as _os, sys as _sys
    pass
    import head2wrist.headcam as headcam
    F0 = headcam.focal(T)
    Dh = np.load(f'{T}/first_depth.npy').astype(np.float32)
    h, w = Dh.shape
    fx, fy = F0*w/1920, F0*h/1080
    vv, uu = np.mgrid[0:h, 0:w]
    ob = np.load(f'{T}/object_mesh.npz')
    n, d = ob['n'], float(ob['d'])
    om = cv2.resize(cv2.dilate(np.load(f'{T}/object_mask.npy').astype(np.uint8), np.ones((31, 31), np.uint8)), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    rdir = np.stack([(uu - w/2)/fx, (vv - h/2)/fy, np.ones_like(uu, float)], -1)
    zpl = -d/np.minimum(rdir @ n, -1e-6)
    Dh = np.where(om & (zpl > 0) & (zpl < 5), zpl, Dh).astype(np.float32)
    Ph = np.stack([(uu - w/2)/fx*Dh, (vv - h/2)/fy*Dh, Dh], -1).reshape(-1, 3)
    idx = np.arange(h*w).reshape(h, w)
    q = [idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]]
    tri = np.concatenate([np.stack([q[0], q[2], q[1]], -1).reshape(-1, 3), np.stack([q[1], q[2], q[3]], -1).reshape(-1, 3)])
    dz = Dh.reshape(-1)[tri]
    tri = tri[(dz.max(1)/np.maximum(dz.min(1), 1e-3) < 1.04) & (dz.min(1) > 0.05)]
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(Ph.astype(np.float32)), o3d.core.Tensor(tri.astype(np.uint32)))
    Vo, Fo, Co = ob['V'].astype(np.float64), ob['F'], ob['C'].astype(np.float32)
    Dx = np.load(f'{I}/first_depth.npy').astype(np.float32)
    Px = np.stack([(uu - w/2)/fx*Dx, (vv - h/2)/fy*Dx, Dx], -1)[Dx > 0.05]
    i_grasp = phase.index('grasp')
    Tg = TCP[i_grasp]
    Vt = (Vo - Tg[:3, 3]) @ Tg[:3, :3]
    lo_, hi_ = Vt.min(0), Vt.max(0)
    m_ = np.minimum(0.015, 0.5*(hi_ - lo_))
    x_ = float(np.clip(0.0, lo_[0] + m_[0], hi_[0] - m_[0]))
    band = np.abs(Vt[:, 0] - x_) < 0.01
    width = float(np.ptp(Vt[band, 1])) if band.sum() > 20 else float(np.ptp(Vt[:, 1]))
    _ap = f"{H2W_OUTPUT_ROOT}/lyra2/explore_inputs/{a.ep.replace('/', '_')}/approach_plan.npz"
    if os.path.exists(_ap) and 'object_width' in np.load(_ap): width = float(np.load(_ap)['object_width'])
    g_close = GR.joint_for_gap(width)
    g_open = GR.joint_for_gap(min(width + 0.04, GR.jaw_gap(0.0)))
    ss_ = lambda x: x*x*(3 - 2*x)
    i_rel = phase.index('release')
    g = np.full(len(phase), g_open)
    for i in range(len(phase)):
        if phase[i] in ('grasp',): g[i] = g_open + (g_close - g_open)*ss_(min(1.0, (i - i_grasp)/7))
        elif phase[i] in ('lift-off', 'transport', 'set-down'): g[i] = g_close
        elif phase[i] == 'release': g[i] = g_close + (g_open - g_close)*ss_(min(1.0, (i - i_rel)/7))
    i_attach = i_grasp + 7
    i_detach = i_rel
    yc = 0.5*(Vt[band, 1].max() + Vt[band, 1].min()) if band.sum() > 20 else float(Vt[:, 1].mean())
    Tobj = [np.eye(4)]*len(phase)
    for i in range(i_grasp, i_attach):
        Ts = np.eye(4); Ts[:3, 3] = -ss_((i - i_grasp)/(i_attach - i_grasp))*yc*Tg[:3, 1]; Tobj[i] = Ts
    Tc_ = np.eye(4)
    Tc_[:3, 3] = -yc*Tg[:3, 1]
    for i in range(i_attach, i_detach + 1): Tobj[i] = TCP[i] @ np.linalg.inv(TCP[i_attach]) @ Tc_
    Vrel = Vo @ Tobj[i_detach][:3, :3].T + Tobj[i_detach][:3, 3]
    cen = Vrel.mean(0)
    sup = support_height(cen, n, d, Px)
    drop = float((Vrel @ n + d).min() - sup)
    for i in range(i_detach + 1, len(phase)):
        k = ss_(min(1.0, (i - i_detach)/3)); Td = np.eye(4); Td[:3, 3] = -k*max(drop, 0.0)*n; Tobj[i] = Td @ Tobj[i_detach]
    print(f'{a.ep}: object width across jaws {100*width:.1f} cm -> joint {g_close:.2f} (pre-grasp {g_open:.2f}), centred by {100*yc:+.1f} cm; '
          f'released {100*drop:.1f} cm above the surface (support {100*sup:.1f} cm)')
    BODY = dict(np.load(f'{T}/body_layer.npz')) if os.path.exists(f'{T}/body_layer.npz') else None
    SH = None
    if 'light_dir' in P.files:
        import head2wrist.rendering.robot_body as RBD; import head2wrist.robot.sg2_kin as SK; import pinocchio as pin
        SH = dict(s=float(P['shadow_strength']), q=P['arm_q'], Thb=P['T_head_base'], lift=float(P['lift']))
        Lu = -np.array(P['light_dir'], float); Lu /= np.linalg.norm(Lu); pen = np.radians(float(P['penumbra_deg']))   # toward the light
        ax = np.cross(Lu, [0, 0, 1.]) if abs(Lu[2]) < 0.9 else np.cross(Lu, [1., 0, 0]); ax /= np.linalg.norm(ax); bx = np.cross(Lu, ax)
        SH['dirs'] = [Lu] + [(np.cos(r)*Lu + np.sin(r)*(np.cos(t)*ax + np.sin(t)*bx)) for r in (pen/4, pen/2) for t in np.linspace(0, 2*np.pi, 4, endpoint=False)]
        for l in GR.LINKS:
            if f'gripper_r_{l}' not in GR._MESH: GR._MESH[f'gripper_r_{l}'] = GR._mesh(f'gripper_r_{l}')
    frames, layers, rec = [], [], dict(g=[], Tobj=[], frame=[])
    for i in range(DIVE, len(phase)):
        c2w, K = C2W[i], Ktr[i]; B = bg[i].astype(np.float32)
        zs = scene_depth(c2w, K)
        orgb, oal, oz = render_object(Tobj[i], c2w, K)
        oal = oal*(oz < zs + 0.015)                                                                     # scene clearly in front hides it (bowl rim)
        # contact shadow: object footprint straight down onto the surface, blurred, while near it
        Vi = Vo @ Tobj[i][:3, :3].T + Tobj[i][:3, 3]; hmin = float((Vi @ n + d).min())
        comp = B.copy()
        if hmin < 0.04:
            foot = Vi - np.outer(Vi @ n + d, n); wc = np.linalg.inv(c2w); pc = foot @ wc[:3, :3].T + wc[:3, 3]
            uvp = (pc @ K.T)[:, :2]/pc[:, 2:3]; sm = np.zeros((H, W), np.uint8)
            okp = (pc[:, 2] > 0.02) & (uvp[:, 0] >= 0) & (uvp[:, 0] < W) & (uvp[:, 1] >= 0) & (uvp[:, 1] < H)
            if okp.sum() > 10:
                hull = cv2.convexHull(uvp[okp].astype(np.float32)); cv2.fillConvexPoly(sm, hull.astype(np.int32), 1)
                blur = max(3, int(6 + 60*hmin)) | 1; shd = cv2.GaussianBlur(sm.astype(np.float32), (blur*3 | 1, blur*3 | 1), blur)
                comp *= (1 - 0.35*(1 - hmin/0.04)*shd)[..., None]
        if BODY is not None:                                    # the robot's own body (gen_episode --body), behind the held object
            ba = BODY['alpha'][i].astype(np.float32)/255; bz = BODY['depth'][i].astype(np.float32)
            comp = comp*(1 - ba[..., None]) + BODY['bgr'][i].astype(np.float32)*ba[..., None]; oal = oal*((oz <= bz + 0.002) | (ba <= 0))
        with_obj = comp*(1 - oal[..., None]) + orgb*oal[..., None]
        if SH is not None:                                       # shadows on what is visible under the gripper layer
            dvis = np.where(oal > 0.5, np.minimum(zs, oz), zs)
            if BODY is not None: dvis = np.where(ba > 0.5, np.minimum(dvis, bz), dvis)
            with_obj = with_obj*shadow_factor(i, c2w, K, dvis)[..., None]
        grgb, gal, gz = GR.render_textured(side, float(g[i]), tex=a.gripper_tex)
        gal = gal*(gz <= oz + 0.002)                                                                    # object between the pads may hide finger parts
        full = with_obj*(1 - gal[..., None]) + grgb.astype(np.float32)*gal[..., None]
        up = lambda x: cv2.rotate(np.clip(x, 0, 255).astype(np.uint8), cv2.ROTATE_90_COUNTERCLOCKWISE)
        frames.append(np.clip(full, 0, 255).astype(np.uint8)); layers.append(np.concatenate([up(B), up(with_obj), up(full)], 1))
        rec['g'].append(float(g[i])); rec['Tobj'].append(Tobj[i]); rec['frame'].append(i)
    write(f'{OUT}/wrist_upright.mp4', [cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE) for f in frames])
    write(f'{OUT}/layers_upright.mp4', layers)
    sel = [phase.index(p) - DIVE for p in ('approach', 'descent', 'grasp', 'transport', 'set-down', 'release')] + [len(frames) - 1]
    sel[2] += 8
    sel[3] += 10
    cv2.imwrite(f'{OUT}/stills.jpg', np.concatenate([cv2.rotate(frames[k], cv2.ROTATE_90_COUNTERCLOCKWISE) for k in sel], 1))
    np.savez_compressed(f'{OUT}/clip.npz', frames_raw_bgr=np.stack(frames), gripper_joint=np.array(rec['g']),
                        object_pose=np.stack(rec['Tobj']), plan_frame=np.array(rec['frame']), side=side)
    print('wrote', OUT, len(frames), 'frames')


if __name__ == '__main__':
    main()
