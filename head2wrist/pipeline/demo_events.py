"""
Grasp / release detection for EgoDex pick-and-place demos, from hand motion (no object tracking).
The aperture-closing detector of the head-error pilot misses most grasps: many objects hold thumb and index apart
(the phone in basic_pick_place/1 kept them 7-10 cm apart), so there is no visible closing. Here a pick is:
  contact  the pinch point (thumb-index midpoint) at a local height minimum within `near` of the lowest pinch height
           of that hand in the episode (the table), moving slowly
  carry    between two contacts the pinch point moves >= min_carry horizontally, rises >= min_lift, and the aperture
           stays within +-hold_tol of its value at the first contact (the hand holds something)
grasp = the first contact of the first valid pair, release = the second.
"""
import numpy as np


def _smooth(x, k=5):
    return np.convolve(np.pad(x, (k//2, k//2), mode='edge'), np.ones(k)/k, mode='valid')


def detect_pick_place(J, side, max_speed=0.006, min_carry=0.08, min_lift=0.015, hold_tol=0.03, reach_down=0.01, t_min=5):
    """J: dict joint -> (T, 3) positions (ARKit world, y up). -> (t_grasp, t_release) or None
    grasp contact = a pause of the pinch point (local speed minimum < max_speed m/frame, local height minimum) with the
    hand reaching down to it (wrist >= reach_down above the pinch point; a resting hand has the wrist on the table and
    the fingertips above it). The finger gap is no cue (holding the phone in basic_pick_place/1 it widened 7.8 -> 9.8 cm).
    release = the first pause after >= min_carry horizontal travel and a lift of >= min_lift above both ends, with the
    gap within +-hold_tol in between (objects are often set down with a flat hand, so no posture test there).
    The grasp time is moved to the last frame before the lift starts (the hand pauses over the object before it)."""
    g = (J[f'{side}ThumbTip'] + J[f'{side}IndexFingerTip'])/2
    ap = _smooth(np.linalg.norm(J[f'{side}ThumbTip'] - J[f'{side}IndexFingerTip'], axis=1))
    h = _smooth(g[:, 1]); hw = _smooth(J[f'{side}Hand'][:, 1]); T = len(h)
    sp = _smooth(np.r_[0, np.linalg.norm(np.diff(g, axis=0), axis=1)])
    pause = [t for t in range(max(t_min, 4), T - 4) if sp[t] < max_speed and sp[t] <= sp[t-3:t+4].min() + 1e-6]
    grasp = [t for t in range(max(t_min, 4), T - 4)
             if sp[t] < max_speed and h[t] <= h[t-4:t+5].min() + 2e-3 and hw[t] - h[t] >= reach_down]
    for tc in grasp:
        for tr in pause:
            if tr - tc < 8: continue
            if np.abs(ap[tc:tr+1] - ap[tc]).max() > hold_tol: break
            if np.linalg.norm((g[tr] - g[tc])[[0, 2]]) < min_carry: continue
            if h[tc:tr+1].max() < max(h[tc], h[tr]) + min_lift: continue
            lift = next(t for t in range(tc, tr) if h[t] > h[tc] + 0.01)          # lift-off starts
            return int(max(tc, lift - 1)), int(tr)
    return None
