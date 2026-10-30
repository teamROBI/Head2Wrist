from head2wrist.config import H2W_OUTPUT_ROOT
#!/usr/bin/env python3
"""
Policy server for the closed-loop sim test: a trained LeRobot policy (ACT / Diffusion, a pretrained_model folder) behind
a localhost socket (multiprocessing.connection, stdlib on both ends). The Isaac collector (collect_demos.py --policy)
sends {'reset': True} at an episode start and {'state': (8,), 'wrist': (240,424,3) uint8, 'head': (270,480,3) uint8}
each frame (images as (bytes, shape): numpy-free messages); the server answers the next action (8,): 7 arm joint targets + gripper joint1.
Run in third_party/lerobot/.venv: policy_server.py --ckpt <.../checkpoints/last/pretrained_model> [--port 6123]
"""
import argparse
from multiprocessing.connection import Listener
import numpy as np, torch





def main():
    global EP, Kin, L, P, PreTrainedConfig, REPRS, R_of, TW_, T_of, XY, a, act, ap, c, cfg, img, joint_of, json, keys, kin, kind, m, mask_state, obs, os, pol, state_vec, sys, to_abs, v
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--port', type=int, default=6123)
    ap.add_argument('--policy', default='act')
    ap.add_argument('--n_action_steps', type=int, default=None, help='override: actions executed per inference')
    ap.add_argument('--repr', default=None, help='representation (repr_lib.py): the server builds the state, predicts chunks, decodes them to absolute, ensembles (ACT weights) and runs IK; the policy must be trained with n_action_steps 1 and no ensembling')
    ap.add_argument('--twin', default=f'{H2W_OUTPUT_ROOT}/sim_twin/stack_18/twin.json')
    ap.add_argument('--ensemble', default=None, help="override ACT temporal_ensemble_coeff: a number, or 'none'")
    ap.add_argument('--blank_image', action='store_true', help='state-only control (trained with BLANK=1): a constant grey image')
    ap.add_argument('--z_floor', type=float, default=None, help='real robot safety: the target TCP is never put below this z (base_link, the '
                   'model\'s lift) - set to the lowest TCP of the training episodes (cube_box trial 1 went 2 cm below it, into the table)')
    ap.add_argument('--xy_box', default=None, help='real robot safety: xmin,xmax,ymin,ymax (base_link, model frame): the target TCP is '
                   'kept inside the training episodes\' xy range (cube_bowl 80K carried the cube 5 cm beyond it, off the table)')
    ap.add_argument('--lift', type=float, default=None, help='real robot: the lift joint (m) instead of the twin json')
    ap.add_argument('--side', default=None, help='real robot: left / right')
    a = ap.parse_args()
    if a.policy == 'act':
        from lerobot.policies.act.modeling_act import ACTPolicy as P
    else:
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy as P
    from lerobot.configs.policies import PreTrainedConfig
    cfg = PreTrainedConfig.from_pretrained(a.ckpt)
    if a.ensemble is not None: cfg.temporal_ensemble_coeff = None if a.ensemble == 'none' else float(a.ensemble)
    if a.n_action_steps is not None: cfg.n_action_steps = a.n_action_steps
    pol = P.from_pretrained(a.ckpt, config=cfg).to('cuda').eval()
    print('execution: n_action_steps', cfg.n_action_steps, 'temporal_ensemble_coeff', getattr(cfg, 'temporal_ensemble_coeff', None), flush=True)
    keys = set(pol.config.input_features)
    print('policy inputs', sorted(keys), flush=True)
    if a.repr:
        import json, sys, os; pass
        from head2wrist.policy.repr_lib import Kin, state_vec, to_abs, T_of, R_of, joint_of, REPRS, mask_state
        TW_ = json.load(open(a.twin)) if (a.lift is None or a.side is None) else {}
        kin = Kin(a.lift if a.lift is not None else TW_['robot']['lift'], a.side or TW_['side']); kind = REPRS[a.repr]
        EP = {}; XY = [float(v) for v in a.xy_box.split(',')] if a.xy_box else None
        def repr_step(m):
            q = np.asarray(m['state'][:7], float); g = float(m['state'][7]); T = kin.fk(q)
            if EP.get('T_start') is None: EP.update(T_start=T, T_prev=T, t=0, preds=[])
            sv = mask_state(a.repr, state_vec(kind, q, g, T, EP['T_prev'], EP['T_start'])).astype(np.float32)
            obs = {'observation.state': torch.from_numpy(sv).unsqueeze(0).cuda(), 'observation.images.wrist': img(m['wrist'])}
            with torch.no_grad(): ch = pol.predict_action_chunk(obs)[0].cpu().numpy()
            t = EP['t']; EP['preds'] = [(t0, c) for t0, c in EP['preds'] if t - t0 < len(c)] + [(t, to_abs(a.repr, ch, q, T))]
            cand = np.stack([c[t - t0] for t0, c in EP['preds']]); w = np.exp(-0.01*np.arange(len(cand))); w = w/w.sum()   # oldest first (ACT)
            if kind == 'joint': tgt = (w[:, None]*cand).sum(0); out = list(tgt[:7]) + [float(tgt[7])]
            else:
                mean = (w[:, None]*cand).sum(0); Tt = T_of(mean[:9])
                if a.z_floor is not None and Tt[2, 3] < a.z_floor: Tt[2, 3] = a.z_floor        # never below the trained workspace
                if XY is not None: Tt[0, 3] = np.clip(Tt[0, 3], XY[0], XY[1]); Tt[1, 3] = np.clip(Tt[1, 3], XY[2], XY[3])   # nor outside its xy
                qt, pe, re = kin.ik(Tt, q)
                out = list(qt) + [float(joint_of(mean[9]))]
            EP['T_prev'] = T; EP['t'] = t + 1
            return [float(v) for v in out]
    img = lambda x: torch.frombuffer(bytearray(x[0]), dtype=torch.uint8).reshape(x[1]).permute(2, 0, 1).float().div(255).unsqueeze(0).cuda()
    if a.blank_image: img = lambda x: torch.full((1, 3) + tuple(x[1][:2]), 0.5, device='cuda')
    with Listener(('localhost', a.port), authkey=b'egoh2w') as L:
        print(f'listening on {a.port}', flush=True)
        while True:
            with L.accept() as c:
                while True:
                    try: m = c.recv()
                    except (EOFError, ConnectionResetError, BrokenPipeError): break     # the bridge stopped (Ctrl+C resets the socket)
                    if m.get('reset'):
                        pol.reset()
                        if a.repr: EP.clear()
                        c.send({'ok': True}); continue
                    if a.repr: c.send({'action': repr_step(m)}); continue
                    obs = {'observation.state': torch.tensor(m['state'], dtype=torch.float32).unsqueeze(0).cuda()}
                    if 'observation.images.wrist' in keys: obs['observation.images.wrist'] = img(m['wrist'])
                    if 'observation.images.head' in keys: obs['observation.images.head'] = img(m['head'])
                    with torch.no_grad(): act = pol.select_action(obs)[0].cpu().numpy()
                    c.send({'action': [float(v) for v in act]})


if __name__ == '__main__':
    main()
