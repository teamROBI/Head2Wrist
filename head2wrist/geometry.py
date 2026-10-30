import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rot, Slerp
CAPTIONS = {
    'basic_pick_place/1': 'A static indoor tabletop scene: a white cardboard box lid, a rainbow pop-it toy, red pliers, a Rubik\'s cube and a smartphone on a beige table against white walls, soft indoor light.',
    'pour/1': 'A static indoor tabletop scene: a dark plastic cup and a bottle of iced tea on a green tablecloth against a white wall and a brown cardboard panel, soft indoor light.',
    'insert_remove_cups_from_rack/1': 'A static indoor tabletop scene: a wire rack holding blue, green, yellow and red plastic cups on a green tablecloth next to a brown wall and a white door, soft indoor light.',
    'stack_unstack_tupperware/1': 'A static indoor scene: a dark wooden cabinet with shelves holding stacked glass food containers, a toy oven, a light wooden table surface, soft indoor light.',
}


def interp_pose(A, B, a):
    R = Slerp([0, 1], Rot.from_matrix([A[:3, :3], B[:3, :3]]))([a]).as_matrix()[0]
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = (1-a)*A[:3, 3] + a*B[:3, 3]
    return T


def K_at(fov_deg, W=1920, H=1080):
    f = (W/2)/np.tan(np.radians(fov_deg/2))
    return np.array([[f, 0, W/2], [0, f, H/2], [0, 0, 1]], np.float64)



def pose_at(poses, tf):
    """Interpolate a camera pose at a fractional frame index."""
    t0 = int(np.floor(tf))
    t1 = min(t0 + 1, max(poses))
    return poses[t0] if t1 == t0 else interp_pose(poses[t0], poses[t1], tf - t0)
