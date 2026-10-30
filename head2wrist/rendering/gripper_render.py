from head2wrist.config import H2W_DATA_ROOT
"""
RH-P12-RN gripper as the FFW-SG2 wrist D405 sees it, from the URDF (ai_worker ffw_sg2_rev1_follower).
  gripper joint g: 0 = open, 1.1 = closed; gripper_<c>_joint2..4 mimic joint1 (multiplier 1)
  camera: the D405 optical frame (OpenCV) at T_TCP_D405 in the TCP frame, same for both arms (URDF); raw orientation
          (jaws at the left edge), pinhole with 87 deg horizontal FoV
render(side, g) ray-casts the five gripper links (open3d, CPU) -> depth (m), camera-frame normals, link index per pixel.
"""
import os, sys
import numpy as np, cv2, trimesh, open3d as o3d, pinocchio as pin
HERE = os.path.dirname(os.path.abspath(__file__))
import head2wrist.robot.sg2_kin as K
from head2wrist.wrist_pose import T_TCP_D405

LINKS = ('rh_p12_rn_base', 'rh_p12_rn_r1', 'rh_p12_rn_r2', 'rh_p12_rn_l1', 'rh_p12_rn_l2')
G_MAX = 1.1


def _mesh(link):
    L = [l for l in K._root.findall('link') if l.get('name') == link][0]; g = L.find('visual') or L.find('collision')
    mh = g.find('geometry/mesh'); o = g.find('origin')
    fn = mh.get('filename').replace('package://ffw_description', K.URDF_ROOT).replace('file://', '')
    m = trimesh.load(fn, force='mesh'); m.apply_scale([float(x) for x in (mh.get('scale') or '1 1 1').split()])
    To = np.eye(4)
    if o is not None:
        To[:3, :3] = pin.rpy.rpyToMatrix(*[float(x) for x in (o.get('rpy') or '0 0 0').split()])
        To[:3, 3] = [float(x) for x in (o.get('xyz') or '0 0 0').split()]
    m.apply_transform(To)
    return m


_MESH = {}
def link_poses_tcp(side, g):
    """{link: 4x4 pose in the TCP frame} for gripper joint g"""
    c = side[0]; m = K.FULL; d = m.createData(); q = pin.neutral(m)
    for j in range(1, 5): q[m.idx_qs[m.getJointId(f'gripper_{c}_joint{j}')]] = g
    pin.forwardKinematics(m, d, q); pin.updateFramePlacements(m, d)
    Tb = d.oMf[m.getFrameId(f'gripper_{c}_rh_p12_rn_base')]; Tt = Tb*pin.SE3(np.eye(3), np.array([0, 0, 0.0965]))
    return {l: Tt.actInv(d.oMf[m.getFrameId(f'gripper_{c}_{l}')]).homogeneous for l in LINKS}


def intrinsics(W=424, H=240, hfov=87.0):
    f = (W/2)/np.tan(np.radians(hfov/2)); return np.array([[f, 0, W/2], [0, f, H/2], [0, 0, 1.]])


def render(side, g, W=424, H=240, hfov=87.0, K_=None):
    """-> depth (H,W) m (inf = no gripper), normals (H,W,3) camera frame, link index (H,W) (-1 = none)"""
    c = side[0]; Tp = link_poses_tcp(side, g); Tc = np.linalg.inv(T_TCP_D405)          # TCP -> camera
    sc = o3d.t.geometry.RaycastingScene(); ids = []
    for l in LINKS:
        key = f'gripper_{c}_{l}'
        if key not in _MESH: _MESH[key] = _mesh(key)
        mm = _MESH[key].copy(); mm.apply_transform(Tc @ Tp[l])
        ids.append(sc.add_triangles(o3d.core.Tensor(mm.vertices.astype(np.float32)), o3d.core.Tensor(mm.faces.astype(np.uint32))))
    Kc = intrinsics(W, H, hfov) if K_ is None else K_
    v, u = np.mgrid[0:H, 0:W]; dirs = np.stack([(u + 0.5 - Kc[0, 2])/Kc[0, 0], (v + 0.5 - Kc[1, 2])/Kc[1, 1], np.ones_like(u, float)], -1)
    rays = np.concatenate([np.zeros_like(dirs), dirs], -1).reshape(-1, 6).astype(np.float32)
    r = sc.cast_rays(o3d.core.Tensor(rays))
    t = r['t_hit'].numpy().reshape(H, W); gid = r['geometry_ids'].numpy().reshape(H, W).astype(np.int64)
    hit = np.isfinite(t); lid = np.full((H, W), -1); lid[hit] = np.searchsorted(ids, gid[hit])
    nrm = r['primitive_normals'].numpy().reshape(H, W, 3)
    return np.where(hit, t, np.inf), nrm, lid                                           # depth along z: t (dirs have z = 1)


def shade(depth, nrm, base=(22, 24, 26), light=np.array([0.3, -0.4, -1.0])):
    """quick flat preview: matte black plastic, a head light"""
    l = light/np.linalg.norm(light); lam = np.clip(-(nrm @ l), 0, 1)
    img = np.array(base, float)[None, None]*(0.6 + 0.8*lam[..., None]) + 40*lam[..., None]**8
    return np.where(np.isfinite(depth)[..., None], np.clip(img, 0, 255), 0).astype(np.uint8)


def jaw_gap(g):
    """distance between the two finger pads across the closing axis (TCP y) at gripper joint g, m"""
    Tp = link_poses_tcp('right', g); ys = []
    for l in ('rh_p12_rn_r2', 'rh_p12_rn_l2'):
        key = f'gripper_r_{l}'
        if key not in _MESH: _MESH[key] = _mesh(key)
        P = _MESH[key].vertices @ Tp[l][:3, :3].T + Tp[l][:3, 3]
        P = P[np.abs(P[:, 2]) < 0.012]                          # pad region around the TCP (pad centre = TCP)
        ys.append(P[:, 1])
    a, b = ys; return float(max(a.min() - b.max(), b.min() - a.max(), 0.0))


_GAP = None
def joint_for_gap(gap):
    """gripper joint that closes the pads to `gap` (m); 0 = fully open"""
    global _GAP
    if _GAP is None:
        gs = np.linspace(0, G_MAX, 111); _GAP = (gs, np.array([jaw_gap(x) for x in gs]))
    gs, gp = _GAP; return float(np.interp(-gap, -gp, gs))


_TEX = None
def render_textured(side, g, W=424, H=240, ss=2, tex=f'{H2W_DATA_ROOT}/assets/gripper_texture.npz'):
    """gripper at joint g with real colours (gripper_texture.py) -> rgb (H,W,3) uint8 BGR, alpha (H,W) float32 in
    [0,1] (ss x ss supersampled edges), depth (H,W) m. Both arms look the same in their own camera."""
    global _TEX
    if _TEX is None:
        z = np.load(tex); _TEX = [(z[f'v{i}'], z[f'f{i}'], z[f'c{i}'].astype(np.float32)) for i in range(len(LINKS))]
    Tp = link_poses_tcp('right', g); Tc = np.linalg.inv(T_TCP_D405)
    sc = o3d.t.geometry.RaycastingScene(); ids = []
    for li, (V, F, _) in enumerate(_TEX):
        T = Tc @ Tp[LINKS[li]]; P = (V @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
        ids.append(sc.add_triangles(o3d.core.Tensor(P), o3d.core.Tensor(F.astype(np.uint32))))
    Ws, Hs = W*ss, H*ss; Kc = intrinsics(Ws, Hs)
    v, u = np.mgrid[0:Hs, 0:Ws]; dirs = np.stack([(u + 0.5 - Kc[0, 2])/Kc[0, 0], (v + 0.5 - Kc[1, 2])/Kc[1, 1], np.ones_like(u, float)], -1)
    r = sc.cast_rays(o3d.core.Tensor(np.concatenate([np.zeros_like(dirs), dirs], -1).reshape(-1, 6).astype(np.float32)))
    t = r['t_hit'].numpy().reshape(Hs, Ws); hit = np.isfinite(t)
    gid = r['geometry_ids'].numpy().reshape(Hs, Ws); pid = r['primitive_ids'].numpy().reshape(Hs, Ws)
    uv = r['primitive_uvs'].numpy().reshape(Hs, Ws, 2)
    img = np.zeros((Hs, Ws, 3), np.float32)
    for li, (V, F, C) in enumerate(_TEX):
        m = hit & (gid == ids[li])
        if not m.any(): continue
        f = F[pid[m]]; b1, b2 = uv[m][:, 0:1], uv[m][:, 1:2]
        img[m] = (1 - b1 - b2)*C[f[:, 0]] + b1*C[f[:, 1]] + b2*C[f[:, 2]]
    al = cv2.resize(hit.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    rgb = cv2.resize(img*hit[..., None], (W, H), interpolation=cv2.INTER_AREA)/np.maximum(al[..., None], 1e-6)
    dep = cv2.resize(np.where(hit, t, 0).astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)/np.maximum(al, 1e-6)
    return np.clip(rgb, 0, 255).astype(np.uint8), al, np.where(al > 0, dep, np.inf)
