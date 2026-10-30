#!/usr/bin/env python3
"""
Table objects in an input frame with SAM 3 generic prompts ('object', 'tool', 'toy', 'box', 'phone', 'cup', 'bowl',
'block', 'food', 'container', and supports: 'plate', 'basket', 'tray', 'dish', 'lid' - without them the plate and basket
of pick_place_food/11 were missed, the olive on the plate floated over a flat table in the scene model). Used by initial_view.py (the overview must see every object), plan_approach.py
(clearance) and make_task_inputs.py (the manipulated object = the mask under the grasp point).
Writes <inputs>/sam3_object_masks.npy (N, H, W bool) and sam3_objects.jpg.
"""
import argparse, os, sys
import numpy as np, cv2, torch
from PIL import Image
from head2wrist.vision.sam3_hands import load





def main():
    global _, a, allm, ap, c, im, img, inp, m, model, out, p, proc, res, rng, s, vis, xs, ys
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', required=True)
    ap.add_argument('--prompts', default='object,tool,toy,box,phone,cup,bowl,block,food,container,plate,basket,tray,dish,lid')
    ap.add_argument('--thr', type=float, default=0.4)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    img = cv2.cvtColor(cv2.imread(os.path.join(a.inputs, 'first_frame.png')), cv2.COLOR_BGR2RGB)
    proc, model = load(a.device)
    im = Image.fromarray(img)
    allm = []
    for p in a.prompts.split(','):
        inp = proc(images=im, text=p, return_tensors='pt').to(a.device)
        with torch.no_grad(): out = model(**inp)
        res = proc.post_process_instance_segmentation(out, threshold=a.thr, mask_threshold=0.5, target_sizes=inp.get('original_sizes').tolist())[0]
        for m, s in zip(res['masks'], res['scores']): allm.append((p, float(s), m.cpu().numpy().astype(bool)))
    vis = img.copy()
    rng = np.random.default_rng(0)
    for p, s, m in allm:
        c = rng.integers(60, 255, 3); vis[m] = (0.5*vis[m] + 0.5*c).astype(np.uint8)
        ys, xs = np.nonzero(m); cv2.putText(vis, p, (int(xs.mean()), int(ys.mean())), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.imwrite(os.path.join(a.inputs, 'sam3_objects.jpg'), cv2.cvtColor(vis[::2, ::2], cv2.COLOR_RGB2BGR))
    np.save(os.path.join(a.inputs, 'sam3_object_masks.npy'), np.stack([m for _, _, m in allm]) if allm else np.zeros((0,) + img.shape[:2], bool))
    print('masks', len(allm), [(p, round(s, 2)) for p, s, _ in allm])


if __name__ == '__main__':
    main()
