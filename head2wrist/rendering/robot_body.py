#!/usr/bin/env python3
"""
The robot's own body as a layer of the generated wrist view: the idle arm and its gripper, the chest, the head, the
lift column and the active arm's links 1-6. The true wrist camera sees them (basic_pick_place/1: up to ~30% of the
image from approach to lift-off, exp-r9); the generated views did not draw them. The active arm's link 7 and gripper
are the gripper layer's (gripper_render.py).
  geometry  URDF visual meshes (ffw_description) posed by forward kinematics of the full model: the active arm from the
            recorded joints, the idle arm at zero (as collect_demos / the planner leave it), the head pitch, the lift;
            the lift column (fixed to the base; its URDF mesh is missing) = the convex hull of its points from the
            Isaac asset (sg2_kin COLUMN_PTS)
  colour    the Isaac asset's materials (FFW_SG2.glb): arms / grippers black, chest / head white, column grey, with a
            headlight Lambert term
render(q_full, T_base_cam) -> bgr (H,W,3) uint8, alpha (H,W) float32 in [0,1] (ss x ss supersampled edges), depth (H,W) m
full_q(side, q_arm, lift, head_pitch) -> the full configuration
"""
import os, sys
import numpy as np, cv2, open3d as o3d, trimesh, pinocchio as pin
HERE = os.path.dirname(os.path.abspath(__file__)); pass
import head2wrist.robot.sg2_kin as K
import head2wrist.rendering.gripper_render as GR

ALBEDO = {'black': (30, 30, 30), 'white': (225, 225, 225), 'grey': (84, 83, 83)}
_MESH = {}


def _colour(link):
    if link in ('arm_base_link',) or link.startswith('head_link'): return ALBEDO['white']
    return ALBEDO['black']


def _link_mesh(l, max_faces=6000):
    """URDF link element -> (V (link frame), F) of its visual mesh, decimated; None if the mesh file is missing"""
    mh = l.find('visual/geometry/mesh')
    if mh is None: return None
    fn = mh.get('filename').replace('package://ffw_description', K.URDF_ROOT).replace('file://', '')
    if not os.path.exists(fn): return None
    m = trimesh.load(fn, force='mesh'); m.apply_scale([float(x) for x in (mh.get('scale') or '1 1 1').split()])
    o = l.find('visual/origin')
    if o is not None:                                                             # the visual's offset in the link frame
        T = np.eye(4); T[:3, :3] = pin.rpy.rpyToMatrix(*[float(x) for x in (o.get('rpy') or '0 0 0').split()]); T[:3, 3] = [float(x) for x in (o.get('xyz') or '0 0 0').split()]
        m.apply_transform(T)
    V, F = np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int32)
    if len(F) > max_faces:
        om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F)).simplify_quadric_decimation(max_faces)
        V, F = np.asarray(om.vertices, np.float64), np.asarray(om.triangles, np.int32)
    return V, F


_ARM = {}
def arm_meshes(side):
    """(name, V, F) of the ACTIVE arm's links 1-7 (shadow casters for compose_wrist)"""
    if side not in _ARM:
        c = side[0]; _ARM[side] = [(l.get('name'),) + r for l in K._root.findall('link') if l.get('name') in [f'arm_{c}_link{k}' for k in range(1, 8)]
                                   for r in [_link_mesh(l, 3000)] if r is not None]
    return _ARM[side]


def _meshes(side, max_faces=6000):
    """(name, V (link frame), F, albedo) of every body link drawn for this active side, decimated"""
    if side in _MESH: return _MESH[side]
    c = side[0]; out = []
    for l in K._root.findall('link'):
        n = l.get('name')
        if n == f'arm_{c}_link7' or n.startswith(f'gripper_{c}_') or 'camera' in n or 'wheel' in n or n in ('base_link', 'lift_link'): continue
        r = _link_mesh(l, max_faces)
        if r is not None: out.append((n, r[0], r[1], _colour(n)))
    hull = trimesh.convex.convex_hull(K.COLUMN_PTS)                               # base_link frame, static
    out.append(('lift_link', np.asarray(hull.vertices, np.float64), np.asarray(hull.faces, np.int32), ALBEDO['grey']))
    _MESH[side] = out
    return out


def full_q(side, q_arm, lift, head_pitch=0.69):
    q = pin.neutral(K.FULL); jid = lambda n: K.FULL.idx_qs[K.FULL.getJointId(n)]
    q[jid('lift_joint')] = lift; q[jid('head_joint1')] = head_pitch
    for j, n in enumerate(K.ARM(side)): q[jid(n)] = q_arm[j]
    return q


def render(side, q, T_base_cam, W=424, H=240, ss=2):
    d = K.FULL.createData(); pin.framesForwardKinematics(K.FULL, d, q); Tci = np.linalg.inv(T_base_cam)
    sc = o3d.t.geometry.RaycastingScene(); ids, alb = [], []
    for n, V, F, col in _meshes(side):
        Tl = Tci if n == 'lift_link' else Tci @ d.oMf[K.FULL.getFrameId(n)].homogeneous
        P = (V @ Tl[:3, :3].T + Tl[:3, 3]).astype(np.float32)
        if (P[:, 2] < 0.01).all(): continue                                       # behind the camera
        ids.append(sc.add_triangles(o3d.core.Tensor(P), o3d.core.Tensor(F.astype(np.uint32)))); alb.append(col)
    Ws, Hs = W*ss, H*ss; Kc = GR.intrinsics(Ws, Hs)
    v, u = np.mgrid[0:Hs, 0:Ws]; dirs = np.stack([(u + 0.5 - Kc[0, 2])/Kc[0, 0], (v + 0.5 - Kc[1, 2])/Kc[1, 1], np.ones_like(u, float)], -1)
    if not ids: return np.zeros((H, W, 3), np.uint8), np.zeros((H, W), np.float32), np.full((H, W), np.inf, np.float32)
    r = sc.cast_rays(o3d.core.Tensor(np.concatenate([np.zeros_like(dirs), dirs], -1).reshape(-1, 6).astype(np.float32)))
    t = r['t_hit'].numpy().reshape(Hs, Ws); hit = np.isfinite(t) & (t > 0.005)
    gid = r['geometry_ids'].numpy().reshape(Hs, Ws); nrm = r['primitive_normals'].numpy().reshape(Hs, Ws, 3)
    dn = dirs/np.linalg.norm(dirs, axis=-1, keepdims=True); lam = np.abs((nrm*dn).sum(-1))         # headlight
    img = np.zeros((Hs, Ws, 3), np.float32); A = np.array(alb, np.float32)[:, ::-1]                 # BGR like the gripper layer
    for k, gi in enumerate(ids):
        m = hit & (gid == gi); img[m] = A[k]*(0.45 + 0.55*lam[m, None])
    al = cv2.resize(hit.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    bgr = cv2.resize(img*hit[..., None], (W, H), interpolation=cv2.INTER_AREA)/np.maximum(al[..., None], 1e-6)
    dep = cv2.resize(np.where(hit, t, 0).astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)/np.maximum(al, 1e-6)
    return np.clip(bgr, 0, 255).astype(np.uint8), al, np.where(al > 0, dep, np.inf).astype(np.float32)
