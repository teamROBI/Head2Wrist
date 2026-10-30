from head2wrist.config import H2W_DATA_ROOT
#!/usr/bin/env python3
"""
Real-pixel RH-P12-RN gripper bank for the FFW-SG2 wrist D405 (raw orientation, 424x240 = the recorded stream).
Per arm and opening bin (0.05 rad, 0 = open .. 1.1 = closed): RGB = per-pixel median over recorded frames at that
opening with the fingers still (the gripper is fixed to the camera, so arm motion does not blur it; the background
varies across frames and drops out of the median); alpha = the URDF render at the bin centre (gripper_render,
4x supersampled). Validated: the render outline matches the real fingers at 0.05-1.08 rad on both arms.
Sources: only datasets whose wrist videos match their gripper state channels. In ffw_sg2_rev1_pick_item, fold_towel,
pour_pot and clean_the_table* cam_wrist_left follows gripper_r_joint1 and cam_wrist_right gripper_l_joint1 (finger
darkness vs state correlation 0.67-0.86 crossed), i.e. the wrist streams are swapped there.
Writes $H2W_DATA_ROOT/assets/gripper_bank.npz and gripper_bank.jpg.
"""
import argparse, glob, os, sys
import numpy as np, cv2, pandas as pd
from multiprocessing import Pool
import head2wrist.rendering.gripper_render as GR

ROOT = f'{H2W_DATA_ROOT}/robot_recordings'
GOOD = ('pick_bowl', 'clear_item', 't4_rotate')            # wrist streams match their state channels
SIDES = {'right': (15, 'cam_wrist_right'), 'left': (7, 'cam_wrist_left')}
W, H, CROP = 424, 240, 160                                  # the gripper stays in x < CROP (raw frame)
BINS = np.round(np.arange(0.0, GR.G_MAX + 1e-6, 0.05), 3)


def vid(p, cam):
    d, rest = p.rsplit('/data/chunk-', 1); ch, ep = rest.split('/')
    return f'{d}/videos/chunk-{ch}/observation.images.{cam}/{ep.replace(".parquet", ".mp4")}'


def collect(job):
    """one episode, one arm -> {bin: [crops]} (at most per_ep frames per bin, spread over the episode)"""
    p, side, tol, per_ep = job; k, cam = SIDES[side]
    s = np.stack(pd.read_parquet(p, columns=['observation.state'])['observation.state'].values)[:, k]
    ds = np.abs(np.gradient(s)); want = {}
    for b in BINS:
        idx = np.nonzero((np.abs(s - b) < tol) & (ds < 0.003))[0]
        if len(idx): want[b] = set(idx[np.linspace(0, len(idx) - 1, min(per_ep, len(idx))).astype(int)].tolist())
    if not want: return side, {}
    cap = cv2.VideoCapture(vid(p, cam)); out = {b: [] for b in want}; t = 0; need = set().union(*want.values())
    while need:
        ok, im = cap.read()
        if not ok: break
        if t in need:
            for b, ts in want.items():
                if t in ts: out[b].append(im[:, :CROP].copy())
            need.discard(t)
        t += 1
    return side, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{H2W_DATA_ROOT}/assets/gripper_bank.npz')
    ap.add_argument('--tol', type=float, default=0.012)
    ap.add_argument('--per_ep', type=int, default=6)
    ap.add_argument('--max_per_bin', type=int, default=240)
    ap.add_argument('--workers', type=int, default=32)
    a = ap.parse_args()
    eps = [p for d in GOOD for p in sorted(glob.glob(f'{ROOT}/ffw_sg2_rev1_{d}/data/chunk-*/*.parquet'))]
    jobs = [(p, s, a.tol, a.per_ep) for p in eps for s in SIDES]
    acc = {s: {b: [] for b in BINS} for s in SIDES}
    with Pool(a.workers) as pool:
        for side, out in pool.imap_unordered(collect, jobs):
            for b, L in out.items(): acc[side][b].extend(L)
    res = {}
    for side in SIDES:
        rgb = np.zeros((len(BINS), H, W, 3), np.uint8); alpha = np.zeros((len(BINS), H, W), np.float32); n = np.zeros(len(BINS), int)
        for i, b in enumerate(BINS):
            L = acc[side][b]; n[i] = len(L)
            if len(L) >= 8:
                L = L[::max(1, len(L)//a.max_per_bin)]
                rgb[i, :, :CROP] = np.median(np.stack(L), 0).astype(np.uint8)
            D, _, _ = GR.render(side, float(b), W*4, H*4)
            alpha[i] = cv2.resize(np.isfinite(D).astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
        # bins without enough frames: nearest filled bin's pixels (alpha stays the bin's own render)
        filled = np.nonzero(n >= 8)[0]
        for i in np.nonzero(n < 8)[0]: rgb[i] = rgb[filled[np.argmin(np.abs(filled - i))]]
        res[side] = (rgb, alpha, n); print(side, 'frames per bin:', n.tolist())
    np.savez_compressed(a.out, bins=BINS, **{f'{s}_rgb': r[0] for s, r in res.items()},
                        **{f'{s}_alpha': r[1].astype(np.float16) for s, r in res.items()}, **{f'{s}_n': r[2] for s, r in res.items()})
    # preview: every 4th bin over grey, per arm
    rows = []
    for s, (rgb, al, n) in res.items():
        tiles = []
        for i in range(0, len(BINS), 4):
            t = (rgb[i]*al[i][..., None] + 128*(1 - al[i][..., None])).astype(np.uint8)
            cv2.putText(t, f'{s} {BINS[i]:.2f} n={n[i]}', (170, 20), 0, 0.5, (0, 0, 255), 1); tiles.append(t)
        rows.append(np.concatenate(tiles, 1))
    cv2.imwrite(a.out.replace('.npz', '.jpg'), np.concatenate(rows, 0))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
