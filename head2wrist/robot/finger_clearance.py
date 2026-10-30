from head2wrist.config import H2W_REPO_ROOT
"""
Finger-aware contact heights for the RH-P12-RN (grasp / place keyposes).
The fingertips reach 2.0 cm past the TCP (pad centre) along the approach, and further down when the gripper is tilted,
so a TCP copied from the human pinch point (about 1 cm above the table) puts the fingers into the table
(exp-r6-sg2-reachability: -1.0 cm at the grasp, -1.4 cm at the release). The TCP is raised along the table normal until the
lowest finger point clears the supporting surface by `clearance`.
finger_pts_tcp.npy: finger mesh points (open) in the TCP frame, from sg2_kin.GRIP_PROXY.
"""
import os
import numpy as np

FINGER_PTS = np.load(os.path.join(H2W_REPO_ROOT, 'head2wrist', 'assets', 'finger_pts_tcp.npy')).astype(float)
_FP = {}
def finger_pts(g):
    """finger link points (TCP frame) at gripper joint g. The RH-P12-RN fingers swing on linkages: closing onto a 4.7 cm
    object (g 0.7) the tips reach 4.5 cm past the TCP instead of 2.0 cm open, so a clearance checked with OPEN fingers
    let the closing tips jam on the table (Isaac replays, planner v3)."""
    k = round(float(g), 2)
    if k not in _FP:
        import sys; pass
        import head2wrist.rendering.gripper_render as GR
        Tp = GR.link_poses_tcp('right', k); P = []
        for l in ('rh_p12_rn_r1', 'rh_p12_rn_r2', 'rh_p12_rn_l1', 'rh_p12_rn_l2'):
            key = f'gripper_r_{l}'
            if key not in GR._MESH: GR._MESH[key] = GR._mesh(key)
            V = np.asarray(GR._MESH[key].vertices); V = V[np.linspace(0, len(V) - 1, min(len(V), 300)).astype(int)]
            P.append(V @ Tp[l][:3, :3].T + Tp[l][:3, 3])
        _FP[k] = np.concatenate(P)
    return _FP[k]


def lowest_finger_height(T, n, d, g=None):
    """height above the table plane (n.p + d, n up) of the lowest finger point for TCP pose T (fingers open, or at
    gripper joint g)"""
    P = FINGER_PTS if g is None else finger_pts(g)
    return float((((T[:3, :3] @ P.T).T + T[:3, 3]) @ n + d).min())


def support_height(p, n, d, pts, radius=0.015, min_pts=30):
    """height of the surface right under p from a point cloud (the lid's floor, a box top), >= the table plane:
    median height within `radius` (horizontal), widened up to 3x only when fewer than min_pts points fall inside.
    The earlier 30th percentile within 3 cm mixed in the table next to a box: 3.7-4.0 cm for the 7.0 cm box top of
    stack/18 (the place plans did not change: every place already cleared the corrected surface)."""
    rel = pts - p; horiz = np.linalg.norm(rel - np.outer(rel @ n, n), axis=1)
    for r in (radius, 1.5*radius, 2*radius, 3*radius):
        near = horiz < r
        if near.sum() >= min_pts: return float(np.clip(np.median(pts[near] @ n + d), 0.0, 0.15))
    return 0.0


def raise_for_fingers(T, n, d, support=0.0, clearance=0.003, g=None):
    """-> (T raised along n so the fingers (open, or at gripper joint g) clear `support` + clearance, the raise in m)"""
    need = support + clearance - lowest_finger_height(T, n, d, g)
    T = T.copy()
    if need > 0: T[:3, 3] += need*n
    return T, max(0.0, need)
