from head2wrist.config import H2W_OUTPUT_ROOT
#!/usr/bin/env python3
"""
Generated vs real wrist views along the same trajectory (field gap G1: no wrist-view generator has been scored against the
real wrist camera). The robot replays the planned joint path (egoh2w_bridge.py --mode replay_plan -> plan_record.npz, a
real D405 frame at every planned pose); the generator's frame for the same plan row comes from the episode's v5 clip
(composite/<name>_v5/clip.npz). Per frame pair (424x240, raw D405 orientation):
  ssim    structural similarity of the grey images (Gaussian window 11, sigma 1.5)
  psnr    dB
  flow    median optical-flow magnitude from generated to real (Farneback) on textured pixels: the misalignment in px
  dino    cosine similarity of mean-pooled DINOv2 ViT-S/14 patch features (content)
reported per task phase. Reference scale (--sim_ref <scene>): the same metrics between the TRUE sim wrist view and the
generator's v5 view on that scene's demos, where generated-view policies matched true-view ones.
Writes <out>/compare.json, compare.mp4 (generated | real | |difference|) and sheet.jpg.
  compare_views.py --ep <task/k> --record <plan_record.npz> [--out dir] [--sim_ref stack_48]
  compare_views.py --sim_ref stack_48 --sim_eps 0,1,2 (reference only)
"""
import argparse, json, os, sys
os.environ.setdefault('XFORMERS_DISABLED', '1')     # DINOv2's xformers kernels do not support this GPU (sm_120): plain attention
import numpy as np, cv2



def ssim(x, y):
    x, y = x.astype(np.float64), y.astype(np.float64); c1, c2 = (0.01*255)**2, (0.03*255)**2; g = lambda im: cv2.GaussianBlur(im, (11, 11), 1.5)
    mx, my = g(x), g(y); sx, sy, sxy = g(x*x) - mx*mx, g(y*y) - my*my, g(x*y) - mx*my
    return float((((2*mx*my + c1)*(2*sxy + c2))/((mx*mx + my*my + c1)*(sx + sy + c2))).mean())


def flow_px(gen, real):
    g1, g2 = cv2.cvtColor(gen, cv2.COLOR_RGB2GRAY), cv2.cvtColor(real, cv2.COLOR_RGB2GRAY)
    f = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 4, 21, 5, 7, 1.5, 0); mag = np.linalg.norm(f, axis=-1)
    tex = cv2.Laplacian(g1, cv2.CV_64F, ksize=3); m = np.abs(tex) > np.percentile(np.abs(tex), 70)
    return float(np.median(mag[m]))


def dino_sim(gen, real):
    global _D
    import torch
    if _D is None:
        _D = torch.hub.load(os.path.expanduser('~/.cache/torch/hub/facebookresearch_dinov2_main'), 'dinov2_vits14_reg', source='local').cuda().eval()
    def feat(im):
        t = torch.from_numpy(cv2.resize(im, (420, 238))).float().div(255).permute(2, 0, 1)[None].cuda()   # 30 x 17 patches of 14 px
        t = (t - torch.tensor([0.485, 0.456, 0.406], device='cuda')[:, None, None])/torch.tensor([0.229, 0.224, 0.225], device='cuda')[:, None, None]
        with torch.no_grad(): return _D.forward_features(t)['x_norm_patchtokens'].mean(1)[0]
    import torch.nn.functional as Fn
    return float(Fn.cosine_similarity(feat(gen), feat(real), dim=0))


def metrics(pairs, phases):
    rows = []
    for (g, r), ph in zip(pairs, phases):
        d = dict(phase=str(ph), ssim=ssim(cv2.cvtColor(g, cv2.COLOR_RGB2GRAY), cv2.cvtColor(r, cv2.COLOR_RGB2GRAY)),
                 psnr=float(cv2.PSNR(g, r)), flow_px=flow_px(g, r))
        if not a.no_dino: d['dino'] = dino_sim(g, r)
        rows.append(d)
    keys = [k for k in rows[0] if k != 'phase']; summ = {'all': {k: float(np.median([x[k] for x in rows])) for k in keys}, 'n': len(rows)}
    for ph in dict.fromkeys(x['phase'] for x in rows):
        sel = [x for x in rows if x['phase'] == ph]; summ[ph] = {k: float(np.median([x[k] for x in sel])) for k in keys} | {'n': len(sel)}
    return rows, summ


def main():
    global C, D, DIVE, G, OUT, P, R, TP, _, _D, a, ap, d, e, g, i, j, k, ks, name, out, p, pairs, ph, phase, phs, pl, r, res, rows, sel, sheet, subprocess, summ, t, vw, x
    ap = argparse.ArgumentParser()
    ap.add_argument('--ep', default=None)
    ap.add_argument('--record', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--sim_ref', default=None)
    ap.add_argument('--sim_eps', default='0,2,3')
    ap.add_argument('--no_dino', action='store_true')
    a = ap.parse_args()
    OUT = f'{H2W_OUTPUT_ROOT}'
    _D = None
    res = {}
    if a.sim_ref:                                              # reference: true sim wrist vs generated v5, where policies transferred
        D = f'{OUT}/sim_policy/{a.sim_ref}'; P = json.load(open(f'{D}/plans_sr.json')); pl = {p['i']: p for p in P['plans']}; pairs, phs = [], []
        for e in [int(x) for x in a.sim_eps.split(',')]:
            t = np.load(f'{D}/demos_sr/ep_{e:04d}.npz')['wrist']; g = np.load(f'{D}/gen/wrist_sr/ep_{e:04d}.npy'); ph = pl[e]['phase']
            for i in range(0, min(len(t), len(g), len(ph)), 4): pairs.append((g[i], t[i])); phs.append(ph[i])
        _, res['sim_reference'] = metrics(pairs, phs); res['sim_reference']['scene'] = a.sim_ref
        print('sim reference (true vs generated v5):', json.dumps(res['sim_reference']['all']))
    if a.ep:
        name = a.ep.replace('/', '_'); R = np.load(a.record); C = np.load(f'{OUT}/composite/{name}_v5/clip.npz'); G = C['frames_raw_bgr'][..., ::-1]
        TP = np.load(f'{OUT}/lyra2/task_inputs/{name}/task_plan.npz'); DIVE = int(TP['dive_frames']); phase = TP['phase']
        ks = R['k']; assert ks.max() < len(G), f'record has plan rows beyond the clip ({ks.max()} >= {len(G)})'
        pairs = [(G[k], R['wrist'][j]) for j, k in enumerate(ks)]; phs = [phase[DIVE + k] if DIVE + k < len(phase) else 'end' for k in ks]
        rows, summ = metrics(pairs, phs); summ['reached'] = int(R['reached'].sum()); res['real'] = summ; res['real_frames'] = rows
        print('real (generated vs real wrist, same planned poses):', json.dumps(summ['all']), f"| {summ['n']} frames, {summ['reached']} poses reached")
        out = a.out or os.path.dirname(os.path.abspath(a.record)); os.makedirs(out, exist_ok=True)
        vw = cv2.VideoWriter(f'{out}/compare_tmp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 8, (3*240, 424))
        for g, r in pairs:
            d = np.clip(3*np.abs(g.astype(int) - r.astype(int)), 0, 255).astype(np.uint8)
            vw.write(cv2.cvtColor(np.concatenate([cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE) for x in (g, r, d)], 1), cv2.COLOR_RGB2BGR))
        vw.release()
        import subprocess                                                     # H.264 (mp4v / MPEG-4 Part 2 does not play in VS Code or browsers)
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', f'{out}/compare_tmp.mp4', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20',
                        f'{out}/compare.mp4']); os.remove(f'{out}/compare_tmp.mp4')
        sel = np.linspace(0, len(pairs) - 1, 5).astype(int)
        sheet = np.concatenate([np.concatenate([cv2.rotate(pairs[i][j], cv2.ROTATE_90_COUNTERCLOCKWISE) for i in sel], 1) for j in (0, 1)], 0)
        cv2.imwrite(f'{out}/sheet.jpg', cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)); json.dump(res, open(f'{out}/compare.json', 'w'), indent=1); print(f'-> {out}/compare.json, compare.mp4, sheet.jpg')
    elif a.sim_ref:
        json.dump(res, open(a.out or f'{OUT}/sim_policy/{a.sim_ref}/view_metrics_reference.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
