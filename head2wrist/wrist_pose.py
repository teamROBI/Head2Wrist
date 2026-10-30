"""
Wrist-camera pose from an EgoDex hand, using the FFW-SG2 URDF (ai_worker/ffw_description, ffw_sg2_rev1_follower).

TCP frame (RH-P12-RN jaw centre, gripper-base axes): z = approach, y = closing axis (fingers at +/-y), x = y cross z.
URDF chain (right arm): arm_r_link7 -> camera_r_bottom_screw_frame -> camera_r_link -> optical (z fwd, x right, y down);
arm_r_link7 -> gripper_r_rh_p12_rn_base (xyz 0 0 -0.078, rpy 0 pi pi) -> pad centre at +0.0965 m along base z.
Result: D405 optical centre at (x, y, z) = (0.098, 0, -0.102) m in the TCP frame, optical axis 5.5 deg off the
approach axis; the jaw centre projects near the image's left edge (u ~ 70 of 848) at ~11 cm depth.

Human hand -> TCP: origin = thumb-index midpoint; z (approach) = PALM NORMAL (out of the palm; sign from handedness:
(index knuckle - wrist) x (little knuckle - wrist) for the right hand, negated for the left); y (jaw closing axis) =
thumb -> index direction orthogonalised to z; x = y cross z.
Why the palm normal: the earlier approach (wrist -> thumb-index midpoint) runs along the extended fingers, so during the
reach it points horizontal/up (98 deg from straight down at EgoDex basic_pick_place/1 t=24, palm 79 -> 19 deg) and the
D405 looked at the wall; at contact across 31 EgoDex grasps it is 97 deg from down (median) vs 58 deg for the palm
normal. The pinch axis stays well away from the palm normal (median 65-72 deg; <30 deg in 3%), so y is stable.
approach='fingers' keeps the old definition. The jaw axis fixes the
wrist roll only up to 180 deg (the gripper is symmetric): the sign of y is chosen so the D405 (+x side) faces the
head camera, and kept continuous along a trajectory (tcp_poses). A back-of-hand rule was tried first and rejected:
its jaw axis was a median ~70 deg off the human pinch direction on EgoDex (|cos| 0.35 at contact, 20 grasps).
Left arm: the same TCP -> D405 transform as the right (URDF; see cam_from_tcp).

IMAGE CONVENTION (): wrist views are used UPRIGHT = the raw D405 frame rotated 90 deg
counter-clockwise (cv2.ROTATE_90_COUNTERCLOCKWISE, 848x480 -> 480x848), so the jaws are at the bottom centre and
world-up is image-up. Always render/generate in the raw sensor frame (the real 87 deg FoV) and rotate the image;
apply the same rotation to the real D405 stream at deployment. The raw stream (and the URDF / Isaac config) put
the jaws at the left centre; verified on ffw_sg2_rev1 recordings and the host Isaac Lab config.
"""
import os
import numpy as np

T_TCP_D405 = np.load(os.path.join(os.path.dirname(__file__), 'assets', 'T_tcp_d405_right.npy'))
FINGERS = ['Index', 'Middle', 'Ring', 'Little']


def dorsal_dir(J, side, t):
    """unit vector pointing out of the back of the hand: opposite to where the flexed fingertips curl"""
    h = J[f'{side}Hand'][t]
    k_i, k_l = J[f'{side}IndexFingerKnuckle'][t], J[f'{side}LittleFingerKnuckle'][t]
    n = np.cross(k_i - h, k_l - h); n /= np.linalg.norm(n)
    palm = (h + k_i + k_l) / 3
    tips = np.mean([J[f'{side}{f}FingerTip'][t] for f in ('Middle', 'Ring', 'Little')], 0)
    if np.dot(tips - palm, n) > 0:          # fingertips curl toward the palm side
        n = -n
    return n


def palm_normal(J, side, t):
    h = J[f'{side}Hand'][t]
    n = np.cross(J[f'{side}IndexFingerKnuckle'][t] - h, J[f'{side}LittleFingerKnuckle'][t] - h); n /= np.linalg.norm(n)
    return n if side == 'right' else -n


def tcp_pose(J, handT, side, t, head_pos=None, prev_y=None, approach='palm'):
    g = (J[f'{side}ThumbTip'][t] + J[f'{side}IndexFingerTip'][t]) / 2
    if approach == 'palm':
        z = palm_normal(J, side, t)
    else:
        z = g - handT[t, :3, 3]; z /= np.linalg.norm(z)
    y = J[f'{side}IndexFingerTip'][t] - J[f'{side}ThumbTip'][t]; y = y - np.dot(y, z)*z; y /= np.linalg.norm(y)
    if prev_y is not None:                  # continuity along a trajectory
        if np.dot(y, prev_y) < 0: y = -y
    elif head_pos is not None:              # D405 side (+x = y cross z) faces the head
        if np.dot(np.cross(y, z), head_pos - g) < 0: y = -y
    x = np.cross(y, z)
    T = np.eye(4); T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, g
    return T


def tcp_poses(J, handT, side, ts, head_pos, approach='palm'):
    """TCP poses over times ts with a continuous roll; the first frame's sign faces the head"""
    out, prev = {}, None
    for t in ts:
        T = tcp_pose(J, handT, side, t, head_pos=head_pos if prev is None else None, prev_y=prev, approach=approach)
        out[t] = T; prev = T[:3, 1]
    return out


def cam_from_tcp(Ttcp, side):
    """camera-to-world (OpenCV optical axes) of the FFW-SG2 D405 given a TCP pose. Both arms: the same transform. The
    URDF mounts camera_l exactly as camera_r relative to its own gripper (checked : identical translation and
    rotation in the TCP frame); an earlier mirrored version for the left arm was wrong by a 180 deg roll."""
    return Ttcp @ T_TCP_D405


def wrist_cam_c2w(J, handT, side, t, head_pos, approach='palm'):
    return cam_from_tcp(tcp_pose(J, handT, side, t, head_pos=head_pos, approach=approach), side)


def hand_aligned_grasp(g, wrist, pinch, dorsal, up, max_pitch_deg=45.0):
    """Robot grasp keypose ALIGNED with the human hand, not a copy of it (): the gripper points
    the way the fingers point (wrist -> thumb-index midpoint), the D405 (+x) is on the back of the hand, the jaws
    close along the thumb -> index axis. Adjustments: the jaw axis is kept LEVEL (both pads at the same height = level
    horizon in the upright image; camera y = TCP y), the approach loses its component along the jaws (so it is
    perpendicular to them) and its tilt from straight down is capped at max_pitch_deg (the gripper body must clear the
    table). All vectors in one frame; g = thumb-index midpoint (TCP origin), pinch = index tip - thumb tip,
    dorsal = out of the back of the hand.
    Replaces an earlier rule (straight-down approach, roll picked by input-frame coverage) that put the D405 on the
    palm side: at EgoDex basic_pick_place/1 t=91 its image-up pointed 134 deg away from the fingers."""
    y = pinch - (pinch @ up)*up; y /= np.linalg.norm(y)
    f = g - wrist; f /= np.linalg.norm(f)
    z = f - (f @ y)*y; z /= np.linalg.norm(z)
    th = np.arccos(np.clip(-z @ up, -1, 1)); th_max = np.radians(max_pitch_deg)
    if th > th_max:
        hz = z - (z @ up)*up; hz /= np.linalg.norm(hz); z = -np.cos(th_max)*up + np.sin(th_max)*hz
    x = np.cross(y, z)
    if x @ dorsal < 0: x, y = -x, -y
    T = np.eye(4); T[:3, :3] = np.stack([x, y, z], 1); T[:3, 3] = g
    return T


def relevel(Rm, up):
    """remove the roll of a TCP/camera rotation so its y axis (upright-image horizontal) lies in the table plane"""
    y = Rm[:, 1] - (Rm[:, 1] @ up)*up; y /= np.linalg.norm(y)
    z = Rm[:, 2] - (Rm[:, 2] @ y)*y; z /= np.linalg.norm(z)
    return np.stack([np.cross(y, z), y, z], 1)


def to_upright(img_raw):
    """raw D405 image (jaws at the left) -> upright wrist view (jaws at the bottom)"""
    import cv2
    return cv2.rotate(img_raw, cv2.ROTATE_90_COUNTERCLOCKWISE)
