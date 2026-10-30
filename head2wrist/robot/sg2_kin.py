from head2wrist.config import H2W_DATA_ROOT, H2W_EGODEX_ROOT, H2W_URDF_ROOT
"""
FFW-SG2 kinematics for planning synthetic wrist trajectories (pinocchio).
  model   ai_worker ffw_sg2_rev1_follower URDF; one arm (side 'right' / 'left', 7 joints) free, the rest locked; lift set
          per setup; TCP = RH-P12-RN pad centre (gripper base +0.0965 m along z); TCP->D405 is the same for both arms and
          matches head2wrist/assets/T_tcp_d405_right.npy
  scene   robot world: z up, x = the demonstrator's head heading on the floor, table top at table_h above the floor;
          table frame: origin on the table plane under the head camera, f = forward (away from the demonstrator),
          r = right, n = up
  setup   the robot's key setting per demo: base (forward, right, heading) in the table frame + lift
"""
import json, os, sys
import numpy as np, h5py, trimesh, pinocchio as pin, xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
URDF_ROOT = os.path.expanduser(f'{H2W_URDF_ROOT}')
URDF = URDF_ROOT + '/urdf/ffw_sg2_rev1_follower/ffw_sg2_follower.urdf'
def ARM(side): return [f'arm_{side[0]}_joint{i}' for i in range(1, 8)]
def FINGERS(side): return (f'gripper_{side[0]}_rh_p12_rn_r2', f'gripper_{side[0]}_rh_p12_rn_l2')
# base footprint in base_link (wheel centres at x +0.137 / -0.290, y +-0.255 plus a margin)
FOOTPRINT = np.array([[0.25, 0.33], [0.25, -0.33], [-0.40, -0.33], [-0.40, 0.33]])





def reduced(lift, side='right'):
    _load_assets()
    q0 = pin.neutral(FULL); q0[FULL.idx_qs[FULL.getJointId('lift_joint')]] = lift
    lock = [FULL.getJointId(n) for n in FULL.names[1:] if n not in ARM(side)]
    m = pin.buildReducedModel(FULL, lock, q0)
    gb = m.frames[m.getFrameId(f'gripper_{side[0]}_rh_p12_rn_base')]
    m.addFrame(pin.Frame('tcp', gb.parentJoint, gb.parentFrame, gb.placement*pin.SE3(np.eye(3), np.array([0, 0, 0.0965])),
                         pin.FrameType.OP_FRAME))
    return m


def mesh_pts(link, n=150):
    _load_assets()
    L = [l for l in _root.findall('link') if l.get('name') == link][0]; g = L.find('collision') or L.find('visual')
    mh = g.find('geometry/mesh'); o = g.find('origin')
    fn = mh.get('filename').replace('package://ffw_description', URDF_ROOT).replace('file://', '')
    sc = np.array([float(x) for x in (mh.get('scale') or '1 1 1').split()])
    V = trimesh.load(fn, force='mesh').vertices*sc; V = V[np.linspace(0, len(V)-1, min(n, len(V))).astype(int)]
    To = pin.SE3(pin.rpy.rpyToMatrix(*[float(x) for x in (o.get('rpy') if o is not None else '0 0 0').split()]),
                 np.array([float(x) for x in (o.get('xyz') if o is not None else '0 0 0').split()]))
    return (To.rotation @ V.T).T + To.translation


_PROXY_CACHE = {}
def proxies(side):
    """(arm link points in their link frames, gripper + finger points in the TCP frame) for one arm"""
    _load_assets()
    if side not in _PROXY_CACHE:
        c = side[0]; arm = {l: mesh_pts(l) for l in [f'arm_{c}_link{i}' for i in range(3, 8)]}
        m0 = reduced(-0.3, side); d0 = m0.createData(); pin.forwardKinematics(m0, d0, pin.neutral(m0)); pin.updateFramePlacements(m0, d0)
        Mt = d0.oMf[m0.getFrameId('tcp')]; grip = {}
        for l in (f'gripper_{c}_rh_p12_rn_base', f'gripper_{c}_rh_p12_rn_r2', f'gripper_{c}_rh_p12_rn_l2', f'arm_{c}_link7'):
            M = Mt.actInv(d0.oMf[m0.getFrameId(l)]); grip[l] = (M.rotation @ mesh_pts(l).T).T + M.translation
        _PROXY_CACHE[side] = (arm, grip)
    return _PROXY_CACHE[side]


# body surface samples for the table check (base_link frame, metres): mobile base + lift column from the Isaac asset



def _surface(link, n=20000):                                          # points sampled on a URDF link's mesh surface
    _load_assets()
    L = [l for l in _root.findall('link') if l.get('name') == link][0]; g = L.find('collision') or L.find('visual')
    mh = g.find('geometry/mesh'); fn = mh.get('filename').replace('package://ffw_description', URDF_ROOT).replace('file://', '')
    m = trimesh.load(fn, force='mesh'); m.apply_scale([float(x) for x in (mh.get('scale') or '1 1 1').split()])
    return trimesh.sample.sample_surface(m, n, seed=0)[0]

ARM_BASE_OFFSET = np.array([-0.0199, 0.0, 1.4316])                     # base_link -> arm_base_link at lift 0


_IDLE = {}
def idle_arm_pts(side, lift):
    """surface points (base_link frame) of the arm NOT used, joints at zero as the plan leaves it (hanging down),
    at this lift: the Isaac replay of basic_pick_place/25 found it hanging through the tabletop, holding the lift up"""
    _load_assets()
    c = side[0]
    if c not in _IDLE:
        _IDLE[c] = {l.get('name'): _surface(l.get('name'), 400) for l in _root.findall('link')
                    if l.get('name').startswith((f'arm_{c}_link', f'gripper_{c}_')) and l.find('visual/geometry/mesh') is not None}
    q = pin.neutral(FULL); q[FULL.idx_qs[FULL.getJointId('lift_joint')]] = lift; d = FULL.createData()
    pin.framesForwardKinematics(FULL, d, q)
    return np.concatenate([P @ d.oMf[FULL.getFrameId(n)].rotation.T + d.oMf[FULL.getFrameId(n)].translation for n, P in _IDLE[c].items()])


def body_table_hits(Tb, lift, table_h, table_edge_fwd, origin, f, thickness=0.04, margin=0.01, idle_side=None):
    """robot body points (base, lift column, chest, the idle arm if idle_side) inside the tabletop slab: in front of
    the edge and between the underside and the top (+ margin). The base may slide under the table; the column, chest
    and idle arm must not hit it."""
    _load_assets()
    chest = CHEST_PTS + ARM_BASE_OFFSET + np.array([0, 0, lift])
    out = {}
    parts = [('base', BASE_PTS), ('column', COLUMN_PTS), ('chest', chest)] + ([('idle_arm', idle_arm_pts(idle_side, lift))] if idle_side else [])
    for nm, P in parts:
        W = (Tb[:3, :3] @ P.T).T + Tb[:3, 3]
        inside = (((W[:, :2] - origin[:2]) @ f[:2]) >= table_edge_fwd) & (W[:, 2] >= table_h - thickness - margin) & (W[:, 2] <= table_h + margin)
        out[nm] = int(inside.sum())
    return out


class Scene:
    """first-camera frame <-> robot world <-> table frame for one demo (task_inputs folder)"""
    def __init__(self, inputs, egodex=None, table_h=0.74, table_edge=-0.08):
        from head2wrist.planning.headframe_cloud import headframe_cloud
        from head2wrist.planning.make_hover_tour import fit_plane
        self.meta = json.load(open(os.path.join(inputs, 'meta.json')))
        self.idle = 'left' if self.meta.get('side', 'right') == 'right' else 'right'          # the arm not used
        self.root = egodex or self.meta.get('root', f'{H2W_EGODEX_ROOT}')     # the dataset the demo is in
        with h5py.File(os.path.join(self.root, self.meta['ep'] + '.hdf5')) as f:
            self.cam0 = f['transforms/camera'][self.meta['t0']].astype(float)
        img, D, Pc, ok = headframe_cloud(inputs); low = ok.copy(); low[:int(0.45*D.shape[0])] = False
        n, d, _ = fit_plane(Pc[low][::40]); n, d = (-n, -d) if n[1] > 0 else (n, d)
        fwd_a = self.cam0[:3, 2].copy(); fwd_a[1] = 0; fwd_a /= np.linalg.norm(fwd_a)          # OpenCV camera: +z = forward
        self.A2R = np.stack([fwd_a, np.cross(np.array([0, 1., 0]), fwd_a), np.array([0, 1., 0])])
        tp = self.cam0[:3, :3] @ (-d*n) + self.cam0[:3, 3]
        self.shift = np.array([0, 0, table_h - (self.A2R @ tp)[2]]); self.table_h = table_h; self.table_edge = table_edge
        o = self.cam_to_world(np.eye(4))[:3, 3]; self.origin = np.array([o[0], o[1], table_h])   # table point under the head
        self.f = np.array([1., 0, 0]); self.r = np.array([0, -1., 0])  # robot world x = head heading; right = -y
        self.head_height_above_table = float(d)

    def cam_to_world(self, T):
        Tw = self.cam0 @ T; R = np.eye(4); R[:3, :3] = self.A2R; R[:3, 3] = self.shift; return R @ Tw

    def base_pose(self, fwd, right, heading_deg):
        """base_link pose in the robot world from table-frame coordinates (heading 0 = facing +f)"""
        p = self.origin + fwd*self.f + right*self.r; yaw = np.radians(-heading_deg)          # + heading = turned right
        T = np.eye(4); c, s = np.cos(yaw), np.sin(yaw); T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]; T[:3, 3] = [p[0], p[1], 0]
        return T

    def body_hits(self, Tb, lift):
        return body_table_hits(Tb, lift, self.table_h, self.table_edge, self.origin, self.f, idle_side=self.idle)

    def column_front(self, Tb):
        """front of the robot's column (base_link x ~ +0.15 m), along f: must stay behind the table edge even when the
        base slides under the tabletop"""
        return float((Tb[:2, :2] @ np.array([0.15, 0.0]) + Tb[:2, 3] - self.origin[:2]) @ self.f[:2])

    def footprint_front(self, Tb):
        """most forward point of the base footprint, in metres along f from the table-frame origin"""
        P = (Tb[:2, :2] @ FOOTPRINT.T).T + Tb[:2, 3]; return float(((P - self.origin[:2]) @ self.f[:2]).max())


def ik(m, dm, T_des, q, iters=150):
    fid = m.getFrameId('tcp'); oM = pin.SE3(T_des[:3, :3], T_des[:3, 3])
    for _ in range(iters):
        pin.forwardKinematics(m, dm, q); pin.updateFramePlacement(m, dm, fid)
        err = pin.log6(dm.oMf[fid].actInv(oM)).vector
        if np.linalg.norm(err[:3]) < 2e-3 and np.linalg.norm(err[3:]) < np.radians(1.5): break
        J = pin.computeFrameJacobian(m, dm, q, fid, pin.LOCAL)
        q = np.clip(q + 0.8*J.T @ np.linalg.solve(J @ J.T + 1e-4*np.eye(6), err), m.lowerPositionLimit, m.upperPositionLimit)
    pin.forwardKinematics(m, dm, q); pin.updateFramePlacement(m, dm, fid)
    err = pin.log6(dm.oMf[fid].actInv(oM)).vector
    return q, float(np.linalg.norm(err[:3])), float(np.degrees(np.linalg.norm(err[3:])))


def reached(pe, re): return pe < 5e-3 and re < 3


def arm_clearance(m, dm, q, table_h, side='right'):
    pin.forwardKinematics(m, dm, q); pin.updateFramePlacements(m, dm)
    return min(float(((dm.oMf[m.getFrameId(l)].rotation @ V.T).T + dm.oMf[m.getFrameId(l)].translation)[:, 2].min()) for l, V in proxies(side)[0].items()) - table_h


def grip_clearance(T_tcp_world, table_h, side='right'):
    """lowest finger point and lowest gripper-body point above the table, from the TCP pose alone"""
    c = side[0]; h = {l: float(((T_tcp_world[:3, :3] @ V.T).T + T_tcp_world[:3, 3])[:, 2].min()) - table_h for l, V in proxies(side)[1].items()}
    return min(h[l] for l in FINGERS(side)), min(h[f'gripper_{c}_rh_p12_rn_base'], h[f'arm_{c}_link7'])


def solve_path(Tb, lift, T_world_list, table_h, seeds=6, seed=0, q_start=None, side='right'):
    """IK along a list of TCP world poses from one base pose + lift; one continuous branch where possible"""
    rng = np.random.default_rng(seed); m = reduced(lift, side); dm = m.createData(); Tbi = np.linalg.inv(Tb)
    lo, hi = np.maximum(m.lowerPositionLimit, -2.5), np.minimum(m.upperPositionLimit, 2.5)
    rows, q = [], q_start
    for T in T_world_list:
        Tl = Tbi @ T; sols = []
        if q is not None: sols.append(ik(m, dm, Tl, q.copy()))
        if q is None or not reached(sols[-1][1], sols[-1][2]):
            for _ in range(seeds): sols.append(ik(m, dm, Tl, rng.uniform(lo, hi), 300))
        good = [x for x in sols if reached(x[1], x[2])]
        if good: qn, pe, re = min(good, key=lambda x: 0 if q is None else float(np.abs(x[0] - q).max()))
        else: qn, pe, re = min(sols, key=lambda x: x[1] + 0.01*np.radians(x[2]))
        rows.append(dict(ok=reached(pe, re), pos_err_mm=1000*pe, rot_err_deg=re,
                         dq=0.0 if q is None else float(np.abs(qn - q).max()),
                         margin=float(np.min(np.minimum(qn - m.lowerPositionLimit, m.upperPositionLimit - qn))),
                         arm_clear=arm_clearance(m, dm, qn, table_h, side), q=qn.tolist()))
        q = qn
    return rows, m

def _load_assets():
    global FULL, _root, _BODY, BASE_PTS, COLUMN_PTS, CHEST_PTS
    if "FULL" in globals():
        return
    FULL = pin.buildModelFromUrdf(URDF)
    _root = ET.parse(URDF).getroot()
    _BODY = np.load(f'{H2W_DATA_ROOT}/assets/ffw_sg2/body_meshes_base_frame.npz')
    BASE_PTS, COLUMN_PTS = _BODY['base'], _BODY['column']
    CHEST_PTS = _surface('arm_base_link')                                  # in arm_base_link, which rides the lift


def __getattr__(name):
    if name in {'BASE_PTS', 'FULL', 'COLUMN_PTS', '_root', 'CHEST_PTS', '_BODY'}:
        _load_assets()
        return globals()[name]
    raise AttributeError(name)
