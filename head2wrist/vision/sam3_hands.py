from head2wrist.config import H2W_DATA_ROOT
#!/usr/bin/env python3
"""SAM 3 (transformers) hand/arm masks with text prompts; optional EgoDex-joint sanity check."""
import argparse, os, sys, json
import numpy as np, cv2, torch
from PIL import Image
from transformers import Sam3Processor, Sam3Model

SAM3 = f'{H2W_DATA_ROOT}/checkpoints/sam3'


def load(device='cpu'):
    proc = Sam3Processor.from_pretrained(SAM3); model = Sam3Model.from_pretrained(SAM3).to(device).eval()
    return proc, model


def sam3_mask(proc, model, img_rgb, prompts=('hand', 'arm'), thr=0.4, device='cpu'):
    im = Image.fromarray(img_rgb); H, W = img_rgb.shape[:2]
    m = np.zeros((H, W), bool); info = []
    for p in prompts:
        inputs = proc(images=im, text=p, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**inputs)
        res = proc.post_process_instance_segmentation(out, threshold=thr, mask_threshold=0.5,
                                                      target_sizes=inputs.get('original_sizes').tolist())[0]
        for mk, sc in zip(res['masks'], res['scores']):
            mk = mk.cpu().numpy().astype(bool); m |= mk; info.append((p, float(sc), float(mk.mean())))
    return m, info


def fill_concave(m, protect=None, close_k=61, hull_reach=60, margin=31):
    """Hand mask for inpainting without concave gaps (user suggestion ): LaMa fills a smooth blob well
    (the phone mask) but struggles with the notches between fingers. A big plain dilation (the earlier 151 px margin)
    also closes the gaps but eats far more of the scene, like the EgoDex bone-capsule mask did.
      1. closing (close_k) fills gaps narrower than the kernel (between fingers);
      2. per connected component, the convex hull fills the remaining dents, limited to hull_reach px from the mask
         (a bent arm keeps its real shape);
      3. filled-in pixels never cover `protect` (other objects' masks); then a small rounded margin."""
    m = m.astype(np.uint8)
    ell = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    add = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ell(close_k))
    near = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5) < hull_reach
    ncomp, lab = cv2.connectedComponents(m)
    for c in range(1, ncomp):
        ys, xs = np.nonzero(lab == c)
        if len(xs) < 50: continue
        hull = cv2.convexHull(np.stack([xs, ys], 1).astype(np.int32)); hm = np.zeros_like(m)
        cv2.fillConvexPoly(hm, hull, 1); add |= hm & near.astype(np.uint8)
    add = add.astype(bool) & ~m.astype(bool)
    if protect is not None: add &= ~protect.astype(bool)
    out = (m.astype(bool) | add).astype(np.uint8)
    if margin > 0:
        out = cv2.dilate(out, ell(margin))
        if protect is not None: out = (out.astype(bool) & ~(protect.astype(bool) & ~m.astype(bool))).astype(np.uint8)
        out = (cv2.GaussianBlur(out.astype(np.float32), (0, 0), margin/4) > 0.25).astype(np.uint8)
    return out.astype(bool)


def hand_removal_mask(proc, model, img_rgb, device='cpu', margin=31, shadow_reach=61):
    """Default hand-removal mask for LaMa: SAM 3 hand/arm + the hands' own shadows (SAM 3 'shadow' / 'hand shadow'
    components touching the hand), concave gaps filled, other objects protected, small rounded margin.
    On EgoDex basic_pick_place/1 t=24 it inpaints as cleanly as the old 151 px dilation with 22% less area; without
    the shadow a faint ghost stays past the fingertips (the gaps alone were not the cause)."""
    hand, _ = sam3_mask(proc, model, img_rgb, device=device)
    objs, _ = sam3_mask(proc, model, img_rgb, prompts=('object', 'tool', 'toy', 'box', 'phone'), device=device)
    objs &= ~cv2.dilate(hand.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    sh, _ = sam3_mask(proc, model, img_rgb, prompts=('shadow', 'hand shadow'), thr=0.3, device=device)
    near = cv2.dilate(hand.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (shadow_reach, shadow_reach))).astype(bool)
    nc, lab = cv2.connectedComponents((sh & ~objs).astype(np.uint8)); keep = np.zeros_like(hand)
    for c in range(1, nc):
        comp = lab == c
        if (comp & near).any() and comp.mean() < 0.1: keep |= comp
    return fill_concave(hand | keep, protect=objs, margin=margin), dict(hand=hand, shadow=keep, objects=objs)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--img'); ap.add_argument('--out'); ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    img = cv2.cvtColor(cv2.imread(a.img), cv2.COLOR_BGR2RGB)
    proc, model = load(a.device)
    m, info = sam3_mask(proc, model, img, device=a.device)
    np.save(a.out + '_mask.npy', m)
    vis = img.copy(); vis[m] = (0.4*vis[m] + 0.6*np.array([255, 0, 255])).astype(np.uint8)
    cv2.imwrite(a.out + '_vis.jpg', cv2.cvtColor(vis[::2, ::2], cv2.COLOR_RGB2BGR))
    print(json.dumps({'instances': info, 'mask_frac': float(m.mean())}))
