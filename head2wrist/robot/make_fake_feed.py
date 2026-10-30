#!/usr/bin/env python3
"""
A feed for fake_sg2.py (the isolated bridge test): one recorded FFW-SG2 LeRobot episode -> <out>.npz with both wrist
streams (BGR 424x240, as the robot's D405 JPEG topics decode) and the joint states, at the recording's 30 fps.
  make_fake_feed.py --dataset <robot-dataset> --episode 0 --out <feed.npz> [--max_s 20]
The labels stay as recorded: the swapped datasets (memory: sg2-wrist-streams-swapped) give a swapped feed, which is what
the camera check should catch.
"""
import argparse, json
import numpy as np, cv2, pandas as pd





def main():
    global S, _, a, ap, cap, df, fr, hd, im, info, n, names, ok, out, s
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max_s', type=float, default=20.0)
    a = ap.parse_args()
    info = json.load(open(f'{a.dataset}/meta/info.json'))
    names = info['features']['observation.state']['names']
    df = pd.read_parquet(f'{a.dataset}/data/chunk-000/episode_{a.episode:06d}.parquet')
    S = np.stack(df['observation.state'].values).astype(np.float32)
    n = min(len(S), int(a.max_s*info['fps']))
    fr = {}
    for s in ('left', 'right'):
        cap = cv2.VideoCapture(f'{a.dataset}/videos/chunk-000/observation.images.cam_wrist_{s}/episode_{a.episode:06d}.mp4'); out = []
        for _ in range(n):
            ok, im = cap.read()
            if not ok: break
            out.append(cv2.resize(im, (424, 240), interpolation=cv2.INTER_AREA) if im.shape[:2] != (240, 424) else im)
        fr[s] = np.stack(out)
    cap = cv2.VideoCapture(f'{a.dataset}/videos/chunk-000/observation.images.cam_head/episode_{a.episode:06d}.mp4')
    hd = []
    for _ in range(n):
        ok, im = cap.read()
        if not ok: break
        hd.append(im)
    fr['head'] = np.stack(hd) if hd else np.zeros((1, 376, 672, 3), np.uint8)
    n = min(n, len(fr['left']), len(fr['right']))
    np.savez_compressed(a.out, left=fr['left'][:n], right=fr['right'][:n], head=fr['head'][:n], state=S[:n], names=np.array(names), fps=float(info['fps']), source=f'{a.dataset}#{a.episode}')
    print(f'{n} frames ({n/info["fps"]:.1f} s) -> {a.out}; right gripper range {S[:n, names.index("gripper_r_joint1")].min():.2f}..{S[:n, names.index("gripper_r_joint1")].max():.2f}, '
          f'left {S[:n, names.index("gripper_l_joint1")].min():.2f}..{S[:n, names.index("gripper_l_joint1")].max():.2f}, lift {S[0, names.index("lift_joint")]:.3f}')


if __name__ == '__main__':
    main()
