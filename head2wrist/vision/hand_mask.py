import numpy as np
import cv2
from scipy.spatial.transform import Rotation, Slerp
FINGERS = ['Index', 'Middle', 'Ring', 'Little']


def bones(J, s):
    """(a, b, radius_m) segments for one side"""
    p = lambda n: J[f'{s}{n}']
    segs = [(p('Arm'), p('Forearm'), 0.055), (p('Forearm'), p('Hand'), 0.045)]
    for fn in FINGERS:
        ch = ['Hand', f'{fn}FingerMetacarpal', f'{fn}FingerKnuckle', f'{fn}FingerIntermediateBase',
              f'{fn}FingerIntermediateTip', f'{fn}FingerTip']
        for a, b in zip(ch[:-1], ch[1:]):
            segs.append((p(a), p(b), 0.025 if a == 'Hand' else 0.013))
    ch = ['Hand', 'ThumbKnuckle', 'ThumbIntermediateBase', 'ThumbIntermediateTip', 'ThumbTip']
    for a, b in zip(ch[:-1], ch[1:]):
        segs.append((p(a), p(b), 0.015))
    return segs


def hand_mask(J, t, E, K, shape, dil=1.35):
    """2D mask of both hands/arms at time t, drawn in the image of camera (E w2c, K)"""
    h, w = shape; m = np.zeros((h, w), np.uint8)
    for s in ('left', 'right'):
        for a, b, r in bones({k: v[t] for k, v in J.items()}, s):
            ca, cb = E[:3, :3]@a+E[:3, 3], E[:3, :3]@b+E[:3, 3]
            if ca[2] < 0.05 or cb[2] < 0.05: continue
            ua = (K[0, 0]*ca[0]/ca[2]+K[0, 2], K[1, 1]*ca[1]/ca[2]+K[1, 2])
            ub = (K[0, 0]*cb[0]/cb[2]+K[0, 2], K[1, 1]*cb[1]/cb[2]+K[1, 2])
            rad = int(np.ceil(dil*K[0, 0]*r/min(ca[2], cb[2])))
            try:
                cv2.line(m, tuple(int(x) for x in ua), tuple(int(x) for x in ub), 1, max(2*rad, 2))
                cv2.circle(m, tuple(int(x) for x in ub), rad, 1, -1)
            except (cv2.error, OverflowError):
                pass
    return m.astype(bool)

