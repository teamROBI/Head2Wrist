"""
The input head frame as a metric point cloud at full image resolution (first-camera frame, OpenCV axes), shared by
the point-cloud baseline, the tracking scores and the dense GEN3C conditioning.
The stored depth is 756 px wide; nearest-neighbour upsampling makes flat ~2.5 px blocks that split into separate
clusters (stripes) when a close / grazing wrist view magnifies them, so depth is upsampled bilinearly, keeping
nearest-neighbour values only next to real depth edges (no flying pixels there).
"""
import os
import numpy as np, cv2
import os as _os, sys as _sys; import head2wrist.headcam as headcam

F0 = headcam.F_EGODEX    # EgoDex head-camera focal at 1920x1080 (headframe_cloud uses the episode's: headcam.focal)


def upsample_depth(Dn, W0, H0, edge_rel=0.05):
    e = np.zeros(Dn.shape, bool)
    jx = np.abs(np.diff(Dn, axis=1)) > edge_rel*Dn[:, 1:]; jy = np.abs(np.diff(Dn, axis=0)) > edge_rel*Dn[1:]
    e[:, 1:] |= jx; e[:, :-1] |= jx; e[1:] |= jy; e[:-1] |= jy
    eu = cv2.resize(e.astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST) > 0
    return np.where(eu, cv2.resize(Dn, (W0, H0), interpolation=cv2.INTER_NEAREST),
                    cv2.resize(Dn, (W0, H0), interpolation=cv2.INTER_LINEAR)).astype(np.float32)


def headframe_cloud(inputs):
    """-> image (H0, W0, 3) uint8 RGB, depth (H0, W0) metric, points (H0, W0, 3), valid mask (H0, W0)"""
    img = cv2.cvtColor(cv2.imread(os.path.join(inputs, 'first_frame.png')), cv2.COLOR_BGR2RGB); H0, W0 = img.shape[:2]
    D = upsample_depth(np.load(os.path.join(inputs, 'first_depth.npy')).astype(np.float32), W0, H0)
    vv, uu = np.mgrid[0:H0, 0:W0]
    f0 = headcam.focal(inputs); P = np.stack([(uu-W0/2)/f0*D, (vv-H0/2)/f0*D, D], -1).astype(np.float32)
    return img, D, P, D > 0.05
