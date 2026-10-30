from head2wrist.config import H2W_OUTPUT_ROOT
#!/usr/bin/env python3
"""
Lyra-2 inputs for the synthesized pick-and-place wrist trajectory (plan_approach.py + plan_transport.py).

  static layer only: the input frame (hands already inpainted) with the manipulated object removed too
                     (SAM 3 mask at the grasp point, dilated + rounded, LaMa); depth under every removed region
                     (hands + object) = table plane + a residual continued smoothly from the border (no step),
                     because the object is rendered separately (sim layer; for now a planar warp, see
                     compose_task_video.py) and must not stay on the table in the generated background
  camera path:       dive head(t0) -> overview (ease-in/out, FoV 105 -> 87 deg), approach, hold while the gripper
                     closes, lift / transport / place, hold while it opens; frame count 1 + 80k (Lyra chunks)
  captions:          the static scene without the object
Writes trajectory.npz (w2c + intrinsics at 1920x1080, first-camera frame), task_plan.npz (per-frame TCP / camera /
phase for compositing), first_frame.png, first_depth.npy, captions.json, meta.json and a mask/inpaint check image.
"""
import argparse, json, os, sys
import numpy as np, cv2

from head2wrist.geometry import interp_pose, K_at
from head2wrist.planning.make_hover_tour import fit_plane

D405_FOV = 87.0
CAPTION = ("A static indoor tabletop scene: a white cardboard box lid, a rainbow pop-it toy, red pliers and a "
           "Rubik's cube on a beige table against white walls, soft indoor light.")


def harmonic_fill(v, known, sigma=40, iters=400):
    """smooth fill of v where ~known (Laplace equation, values fixed on known): large-Gaussian normalised-convolution
    initial guess, then Jacobi relaxation (continuous at the border of the known region)"""
    v = v.astype(np.float32); known = known.astype(bool)
    num = cv2.GaussianBlur(np.where(known, v, 0).astype(np.float32), (0, 0), sigma)
    den = cv2.GaussianBlur(known.astype(np.float32), (0, 0), sigma)
    x = np.where(known, v, num/np.maximum(den, 1e-4)).astype(np.float32)
    k = np.array([[0, .25, 0], [.25, 0, .25], [0, .25, 0]], np.float32)
    for _ in range(iters):
        x = np.where(known, v, cv2.filter2D(x, -1, k, borderType=cv2.BORDER_REPLICATE))
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=f'{H2W_OUTPUT_ROOT}/lyra2/explore_inputs/basic_pick_place_1')
    ap.add_argument('--out', default=f'{H2W_OUTPUT_ROOT}/lyra2/task_inputs/basic_pick_place_1')
    ap.add_argument('--hold', type=int, default=12, help='frames the camera rests while the gripper closes / opens')
    ap.add_argument('--min_dive', type=int, default=36)
    ap.add_argument('--margin', type=int, default=41, help='mask dilation (px at 1920x1080)')
    ap.add_argument('--multiple', type=int, default=240, help='frames = 1 + multiple*k (240 fits both Lyra-2 (80) and GEN3C (120) chunks)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    A = np.load(os.path.join(a.src, 'approach_plan.npz')); Tp = np.load(os.path.join(a.src, 'transport_plan.npz'))
    meta_src = json.load(open(os.path.join(a.src, 'meta.json')))
    img = cv2.cvtColor(cv2.imread(os.path.join(a.src, 'first_frame.png')), cv2.COLOR_BGR2RGB); H0, W0 = img.shape[:2]
    D = np.load(os.path.join(a.src, 'first_depth.npy')).astype(np.float32); h, w = D.shape
    import os as _os, sys as _sys; import head2wrist.headcam as headcam; F0 = headcam.focal(a.src)
    fx, fy = F0*w/1920, F0*h/1080
    # ---- 1. object mask: the SAM 3 mask under the grasp point ----
    p = A['contact']; cu, cv_ = int(F0*p[0]/p[2] + W0/2), int(F0*p[1]/p[2] + H0/2)
    masks = [m for m in np.load(os.path.join(a.src, 'sam3_object_masks.npy')) if 0 < m.mean() < 0.15]
    if masks: cu, cv_ = min(max(cu, 0), masks[0].shape[1] - 1), min(max(cv_, 0), masks[0].shape[0] - 1)   # a grasp at the frame's edge (real cube_box/2)
    hits = [m for m in masks if cv2.dilate(m.astype(np.uint8), np.ones((25, 25), np.uint8))[cv_, cu]]
    if not hits:      # grasp contact a few cm above the table projects just past the mask edge (as in plan_transport)
        dist = [cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5)[cv_, cu] for m in masks]
        hits = [masks[int(np.argmin(dist))]] if masks and min(dist) < 60*W0/1920 else []
    assert hits, 'no SAM 3 mask at (or within 60 px of) the grasp point'
    m0 = min(hits, key=lambda m: m.sum()).astype(np.uint8)
    k = a.margin
    m = cv2.dilate(m0, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m = (cv2.GaussianBlur(m.astype(np.float32), (0, 0), k/4) > 0.25).astype(np.uint8)
    from simple_lama_inpainting import SimpleLama
    from PIL import Image
    inp = np.array(SimpleLama()(Image.fromarray(img), Image.fromarray(m*255)))[:H0, :W0]
    cv2.imwrite(os.path.join(a.out, 'first_frame.png'), cv2.cvtColor(inp, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(a.out, 'object_removed_check.jpg'), cv2.cvtColor(np.concatenate(
        [img, np.where(m[..., None] > 0, (255, 0, 255), img).astype(np.uint8), inp], 1)[::3, ::3], cv2.COLOR_RGB2BGR))
    np.save(os.path.join(a.out, 'object_mask.npy'), m0.astype(bool))
    # ---- 2. depth under every removed region (hands + object): table plane + a residual continued smoothly from the
    #      border. The hands' depth from make_hover_tour (single-view DA3 scaled by one factor) sits a few mm-cm off the
    #      surrounding table, a step along the mask outline that GEN3C / Lyra-2 render as a crease up close. ----
    vv, uu = np.mgrid[0:h, 0:w]; ok = D > 0.05
    Pc = np.stack([(uu-w/2)/fx*D, (vv-h/2)/fy*D, D], -1)
    n, d, _ = fit_plane(Pc[(vv > 0.45*h) & ok][::7])
    ray = np.stack([(uu-w/2)/fx, (vv-h/2)/fy, np.ones_like(D)], -1)
    Dplane = (-d/(ray@n)).astype(np.float32)                                             # n.(t*ray) + d = 0
    removed = m.astype(bool)
    hm_path = os.path.join(a.src, 'hand_removal_mask.npy')
    if os.path.exists(hm_path): removed |= np.load(hm_path)
    else: print('WARN: no hand_removal_mask.npy in --src; only the object region gets seamless depth')
    band = cv2.dilate(removed.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))   # covers the step
    bk = cv2.resize(band, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    known = ~bk & ok & (Dplane > 0.05)
    res_f = harmonic_fill(np.where(known, D - Dplane, 0), known)
    D2 = np.where(bk & (Dplane > 0.05), Dplane + res_f, D).astype(np.float32)
    np.save(os.path.join(a.out, 'first_depth.npy'), D2)
    np.save(os.path.join(a.out, 'removed_mask.npy'), removed)
    # ---- 3. camera path ----
    CA, TA = A['cam'], A['tcp']; CT, TT = Tp['cam'], Tp['tcp']
    rest = (len(CA)-1) + a.hold + (len(CT)-1) + a.hold
    N = 1 + a.multiple*int(np.ceil((a.min_dive + rest - 1)/a.multiple)); DIVE = N - rest
    fov_head = np.degrees(2*np.arctan(960/F0))
    c2w, K, tcp, phase = [], [], [], []
    for i in range(DIVE):
        s = i/(DIVE-1); e = s*s*s*(10-15*s+6*s*s)
        c2w.append(interp_pose(np.eye(4), CA[0], e)); K.append(K_at(fov_head*(1-e) + D405_FOV*e)); tcp.append(np.full((4, 4), np.nan)); phase.append('dive')
    tr_a = int(A['n_transit'])
    for i in range(1, len(CA)):
        c2w.append(CA[i]); K.append(K_at(D405_FOV)); tcp.append(TA[i]); phase.append('approach' if i < tr_a else 'descent')
    for _ in range(a.hold):
        c2w.append(CA[-1]); K.append(K_at(D405_FOV)); tcp.append(TA[-1]); phase.append('grasp')
    nu, nd = int(Tp['n_liftoff']), int(Tp['n_setdown'])
    for i in range(1, len(CT)):
        c2w.append(CT[i]); K.append(K_at(D405_FOV)); tcp.append(TT[i])
        phase.append('lift-off' if i < nu else ('set-down' if i >= len(CT)-nd else 'transport'))
    for _ in range(a.hold):
        c2w.append(CT[-1]); K.append(K_at(D405_FOV)); tcp.append(TT[-1]); phase.append('release')
    c2w = np.stack(c2w); assert len(c2w) == N, (len(c2w), N)
    np.savez(os.path.join(a.out, 'trajectory.npz'), w2c=np.linalg.inv(c2w).astype(np.float32),
             intrinsics=np.stack(K).astype(np.float32), image_height=1080, image_width=1920)
    grasp_idx = phase.index('grasp')
    np.savez(os.path.join(a.out, 'task_plan.npz'), c2w=c2w, tcp=np.stack(tcp), phase=np.array(phase),
             grasp_frame=grasp_idx, tcp_grasp=TA[-1], dive_frames=DIVE)
    json.dump({str(c): (CAPTION if c == 0 else 'Close-up view just above the table surface. ' + CAPTION)
               for c in range(0, N-1, 80)}, open(os.path.join(a.out, 'captions.json'), 'w'), indent=1)   # Lyra-2 only
    st = np.linalg.norm(np.diff(c2w[:, :3, 3], axis=0), axis=1)
    from scipy.spatial.transform import Rotation as R
    rot = [np.degrees(R.from_matrix(c2w[i][:3, :3].T@c2w[i+1][:3, :3]).magnitude()) for i in range(N-1)]
    info = dict(**({'head_focal': meta_src['head_focal']} if meta_src.get('head_focal') else {}), ep=meta_src['ep'], **({'root': meta_src['root']} if meta_src.get('root') else {}), t0=meta_src['t0'], side=meta_src.get('side', 'right'), n_frames=N, dive_frames=DIVE, hold_frames=a.hold,
                phase_counts={p_: phase.count(p_) for p_ in dict.fromkeys(phase)},
                object_mask_frac=float(m.mean()), max_cam_step_cm=float(100*st.max()), max_rot_deg=float(max(rot)),
                dive_max_step_cm=float(100*st[:DIVE-1].max()), dive_max_rot_deg=float(max(rot[:DIVE-1])),
                chunk_starts_lyra={str(c): phase[c] for c in range(0, N-1, 80)}, chunk_starts_gen3c={str(c): phase[c] for c in range(0, N-1, 120)})
    json.dump(info, open(os.path.join(a.out, 'meta.json'), 'w'), indent=1); print(json.dumps(info))


if __name__ == '__main__':
    main()
