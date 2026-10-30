import numpy as np
import cv2
from scipy.spatial.transform import Rotation, Slerp
def read_frames(path, want):
    cap = cv2.VideoCapture(path); out = {}; i = 0; mx = max(want)
    while i <= mx:
        ok, fr = cap.read()
        if not ok: break
        if i in want: out[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    return out

