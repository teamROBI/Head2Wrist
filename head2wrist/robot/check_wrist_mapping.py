#!/usr/bin/env python3
"""
Which arm does each wrist camera topic show? From a bridge record (egoh2w_bridge.py --mode record -> record.npz: both
wrist streams + both arms' joints) the fingers visible in each stream are matched against the gripper joints: the
dark-pixel IoU of the URDF-rendered fingers at a joint value (rendering/gripper_texture.py finger_iou, the check that
found about half of the recorded SG2 datasets swapped). The two hypotheses are scored jointly on the frames where the two
grippers' openings differ (> --min_diff rad), where they predict different fingers:
  own      IoU(left topic, left joint) + IoU(right topic, right joint)
  swapped  IoU(left topic, right joint) + IoU(right topic, left joint)
During the recording open / close ONE gripper fully while the other stays still, so the openings differ for a while.
Writes <record dir>/wrist_mapping.json and wrist_mapping.jpg.
  check_wrist_mapping.py <record.npz> [--min_diff 0.15] [--margin 0.05]      (main .venv)
"""
import argparse, json, os, sys
import numpy as np, cv2
from head2wrist.rendering.gripper_texture import finger_iou





def main():
    global G, K, R, a, ap, arm, bgr, c, cx, cy, d, df, f_gen, flag, fx, fy, i, info, iou, k, msg, n, out_dir, own, res, s, sheet, sw, swapped, sx, t, ts, use, v, wins
    ap = argparse.ArgumentParser()
    ap.add_argument('record')
    ap.add_argument('--min_diff', type=float, default=0.15)
    ap.add_argument('--margin', type=float, default=0.05)
    ap.add_argument('--max_frames', type=int, default=40)
    a = ap.parse_args()
    R = np.load(a.record)
    out_dir = os.path.dirname(os.path.abspath(a.record))
    n = len(R['t'])
    G = {s: R[f'q_{s}'][:, 7].astype(float) for s in ('left', 'right')}
    print(f'{n} frames; gripper joint range left {G["left"].min():.2f}..{G["left"].max():.2f}, right {G["right"].min():.2f}..{G["right"].max():.2f}')
    use = np.where(np.abs(G['left'] - G['right']) > a.min_diff)[0]
    res = dict(frames=int(n), frames_used=int(len(use)))
    if len(use) < 6:
        msg = (f'UNCLEAR: the two grippers differ by > {a.min_diff} rad in only {len(use)} frames; open / close one gripper fully while '
               'the other stays still, and record again'); swapped = None
    else:
        ts = use[np.linspace(0, len(use) - 1, min(len(use), a.max_frames)).astype(int)]
        bgr = {c: {t: cv2.cvtColor(R[c][t], cv2.COLOR_RGB2BGR) for t in ts} for c in ('left', 'right')}   # record frames are RGB
        iou = {(c, arm): np.array([finger_iou(bgr[c][t], G[arm][t]) for t in ts]) for c in ('left', 'right') for arm in ('left', 'right')}
        own = iou[('left', 'left')] + iou[('right', 'right')]; sw = iou[('left', 'right')] + iou[('right', 'left')]
        d = float(np.mean(own - sw)); wins = float(np.mean(own > sw))
        res.update({f'iou_{c}_topic_vs_{arm}_joint': float(v.mean()) for (c, arm), v in iou.items()}, own=float(own.mean()), swapped_score=float(sw.mean()),
                   own_minus_swapped=d, frames_own_better=wins)
        for c in ('left', 'right'):
            print(f'  topic {str(R[f"{c}_topic"])}: finger IoU vs left joint {iou[(c, "left")].mean():.3f}, vs right joint {iou[(c, "right")].mean():.3f}')
        print(f'  {len(ts)} frames where the openings differ: own {own.mean():.3f} vs swapped {sw.mean():.3f} (own better in {100*wins:.0f}% of them)')
        swapped = None if abs(d) < a.margin else d < 0
        msg = ('UNCLEAR: the two hypotheses score alike; open / close one gripper fully while the other stays still, and record again' if swapped is None else
               'SWAPPED: the camera_left topic shows the RIGHT gripper; give the bridge --image_topic of the other wrist camera'
               if swapped else 'OK: each wrist topic shows its own arm')
    print(msg)
    res.update(swapped=swapped, message=msg)
    info = json.loads(str(R['camera_info'])) if 'camera_info' in R.files else {}
    for c, v in info.items():                     # the generator draws the wrist views with the nominal D405 (gripper_render.intrinsics)
        K = np.array(v['K']).reshape(3, 3); sx = 424/max(v['width'], 1); f_gen = (424/2)/np.tan(np.radians(87.0/2))
        fx, fy, cx, cy = K[0, 0]*sx, K[1, 1]*sx, K[0, 2]*sx, K[1, 2]*sx; res[f'intrinsics_{c}'] = dict(fx=fx, fy=fy, cx=cx, cy=cy, raw=v)
        df = 100*(fx/f_gen - 1)       # training augmentation (train_repr.py AUG=strong) randomizes scale +-5% and shifts +-4%
        flag = (f'OK (focal {df:+.1f}%, within the +-5% scale augmentation)' if abs(df) < 5 and abs(cx - 212) < 10 and abs(cy - 120) < 10 else
                f'DIFFERS (focal {df:+.1f}%, principal point off by ({cx - 212:+.0f}, {cy - 120:+.0f}) px): re-render the views with these intrinsics')
        print(f'  {c} camera_info at 424 px: fx {fx:.1f} fy {fy:.1f} cx {cx:.1f} cy {cy:.1f} (generator: f {f_gen:.1f}, cx 212, cy 120) -> {flag}')
    if not info: print('  (no camera_info recorded)')
    json.dump(res, open(f'{out_dir}/wrist_mapping.json', 'w'), indent=1)
    k = np.linspace(0, n - 1, 4).astype(int)
    sheet = np.concatenate([np.concatenate([cv2.cvtColor(R[c][t], cv2.COLOR_RGB2BGR) for t in k], 1) for c in ('left', 'right')], 0)
    for i, c in enumerate(('left', 'right')):
        cv2.putText(sheet, f'{c} topic' + ('' if swapped is None else f': shows the {(("right" if c == "left" else "left") if swapped else c)} gripper'), (8, 22 + 240*i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imwrite(f'{out_dir}/wrist_mapping.jpg', sheet)
    print(f'-> {out_dir}/wrist_mapping.json, wrist_mapping.jpg')


if __name__ == '__main__':
    main()
