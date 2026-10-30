from head2wrist.config import H2W_DATA_ROOT
"""
Proprioception / action representations for the sim policy test (representation sweep), shared by the dataset
converter (to_lerobot_repr.py), the training transform (train_repr.py) and the policy server (policy_server.py --repr).
Frames: the end effector (EEF) is the planner's TCP (sg2_kin reduced model, right arm, fixed lift) in the robot base_link
frame - independent of where the base stands. Rotations as 6D (the first two columns of R). Gripper as joint (rad) or
jaw width (m, assets/gripper_gap_table.npz).
  joint_abs  state q(7)+g(1)                                  action next absolute q*(7)+g*(1)              (ACT / ALOHA)
  joint_rel  state q(7)+g(1)                                  action q* - q_now (per chunk) + absolute g*
  eef_abs    state p(3)+r6(6)+w(1), base frame                action next absolute p*+r6*+w*
  eef_rel    state UMI-style: previous-frame pose relative to the current (p 3 + r6 6), rotation relative to the
             episode's first pose (r6 6), width (1) = 16          action pose relative to the CURRENT pose
             inv(T_now) @ T* (p 3 + r6 6) + absolute width       (UMI 'relative' obs + action repr)
"""
import os, sys
import numpy as np
X = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); pass
REPRS = {'joint_abs': 'joint', 'joint_rel': 'joint', 'eef_abs': 'eef', 'eef_rel': 'umi', 'eef_rel_ns': 'umi', 'eef_hybrid': 'eef'}   # repr -> stored state type
# eef_hybrid: the absolute base-frame EEF state (eef_abs's) with eef_rel's action (relative to the current pose): the absolute
# anchor that helped generated-view policies + the relative action that generalized on true views
# eef_rel_ns: eef_rel with the proprio reduced to the jaw width (the pose dims zeroed): the "state shortcut" test
# (state-free policies generalize better to new object positions - arxiv 2509.18644)


def mask_state(repr_, sv):
    if repr_ == 'eef_rel_ns': sv = sv.copy(); sv[..., :-1] = 0
    return sv
_GJ = _GG = None

def _gap():
    global _GJ, _GG
    if _GJ is None:
        with np.load(H2W_DATA_ROOT + '/assets/gripper_gap_table.npz') as gap:
            _GJ, _GG = gap['joint'].copy(), gap['gap'].copy()



def width_of(g):
    _gap(); return np.interp(g, _GJ, _GG)                                 # gripper joint -> jaw width (m)
def joint_of(w):
    _gap(); return np.interp(-np.asarray(w), -_GG, _GJ)                   # jaw width -> gripper joint


def r6(R): return np.concatenate([R[..., :, 0], R[..., :, 1]], -1)
def R_of(v):
    a, b = v[..., :3], v[..., 3:6]; x = a/np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - (x*b).sum(-1, keepdims=True)*x; y = b/np.linalg.norm(b, axis=-1, keepdims=True); z = np.cross(x, y)
    return np.stack([x, y, z], -1)
def T_of(p9):
    T = np.zeros(p9.shape[:-1] + (4, 4)); T[..., :3, :3] = R_of(p9[..., 3:9]); T[..., :3, 3] = p9[..., :3]; T[..., 3, 3] = 1; return T
def p9_of(T): return np.concatenate([T[..., :3, 3], r6(T[..., :3, :3])], -1)


class Kin:
    """FK / IK of the right arm in base_link at a fixed lift (the planner's reduced model + tcp frame)"""
    def __init__(self, lift, side='right'):
        import pinocchio as pin; import head2wrist.robot.sg2_kin as K
        self.pin, self.K = pin, K; self.m = K.reduced(min(max(lift, -0.5 + 1e-4), -1e-4), side); self.d = self.m.createData()
        self.f = self.m.getFrameId('tcp')
    def fk(self, q):
        self.pin.framesForwardKinematics(self.m, self.d, np.asarray(q, float)); return self.d.oMf[self.f].homogeneous.copy()
    def ik(self, T, q0):
        q, pe, re = self.K.ik(self.m, self.d, T, np.asarray(q0, float).copy(), iters=100); return q, pe, re


def state_vec(kind, q, g, T, T_prev=None, T_start=None):
    if kind == 'joint': return np.r_[q, g]
    if kind == 'eef': return np.r_[p9_of(T), width_of(g)]
    rel = np.linalg.inv(T) @ (T_prev if T_prev is not None else T)
    return np.r_[p9_of(rel), r6(T_start[:3, :3].T @ T[:3, :3]), width_of(g)]      # 'umi'


def to_rel(repr_, chunk, state, T_now):
    """training: stored absolute action chunk (H, D) -> the representation's action (same shape)"""
    out = chunk.copy()
    if repr_ == 'joint_rel': out[:, :7] = chunk[:, :7] - state[:7]
    elif repr_ in ('eef_rel', 'eef_rel_ns', 'eef_hybrid'): out[:, :9] = p9_of(np.linalg.inv(T_now) @ T_of(chunk[:, :9]))
    return out


def to_abs(repr_, chunk, q_now, T_now):
    """inference: the policy's chunk (H, D) -> absolute targets in the stored space (joints or base-frame EEF pose + width)"""
    out = np.array(chunk, float).copy()
    if repr_ == 'joint_rel': out[:, :7] = out[:, :7] + q_now
    elif repr_ in ('eef_rel', 'eef_rel_ns', 'eef_hybrid'): out[:, :9] = p9_of(T_now @ T_of(out[:, :9]))
    return out
