from head2wrist.config import H2W_DATA_ROOT, H2W_OUTPUT_ROOT, H2W_PYTHON, H2W_REPO_ROOT
#!/usr/bin/env python3
"""
Generated-wrist arm of the sim policy test, per demo episode: the wrist video our pipeline makes for the episode's
camera path, from the scene reconstruction of gen_scene.py (head frame only). Per episode, a pseudo-episode
'sim/<scene>_e<i>' that copies the scene's inputs and proxy and has its own:
  camera path  the D405 pose at every recorded frame: FK of the RECORDED arm state (planner model, base pose, lift) x the
               link7 mount, in the head-camera frame - the same instant the sim wrist camera captured
  object       the grasped object's mesh at the episode's start pose (the twin's mesh, moved), with the table plane
  phases       the expert's (grasp / lift-off / ... / release: the object layer's jaw schedule and attachment)
then render_proxy (--renderer v5: the rasterized proxy; blender: Blender v3 on top of it) and compose_wrist (gripper +
object layers) -> raw-orientation wrist frames (T, 240, 424, 3) for to_lerobot.py --wrist_from.
Usage: gen_episode.py --scene stack/18 --demos <dir> [--episodes 0,1] [--renderer v5] [--out <dir>]
"""
import argparse, shutil, glob, json, os, subprocess, sys, time
import numpy as np, cv2
X = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); REPO = os.path.dirname(X); PY = H2W_PYTHON
import head2wrist.robot.sg2_kin as K; import pinocchio as pin
from head2wrist.wrist_pose import T_TCP_D405



def link(src, dst):
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


def main():
    global BA, BB, BZ, C2W, Cw, DIRS, E, GRD, I, I0, J, K1920, LI, Lw, M, N, OUT, PL, R, RB, RBD, Rw, T, T0, TCP, TW, Tb, Tb_e, Tc, Th, Thi, Vw, Xw, _, _Kw, _V, _cb, _cv2, _cx, _cy, _h, _l, _m0, _s, _u, _v, _w, a, al, ap, az, bg, bgr, c, c0, cap, cmds, d, dF, d_c, dep, dm, dxy, dyaw, ename, extra, f, f7, f_, files, fn, fr, ft, g, get_prompt, i, im, k_, keep, lift, m, meta, n_c, o, ok, okk, on, p, p0, ph, pre, ps, q, q0, qf, r, rb, rn, rs, rv, rw, sc, side, sname, sp, ss, t_, tag, tb, tilt, x, ys, z
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='stack/18')
    ap.add_argument('--demos', required=True)
    ap.add_argument('--episodes', default=None)
    ap.add_argument('--renderer', default='v5', choices=['v5', 'blender', 'transfer'],
                                                      help='transfer: Blender v3 + Cosmos-Transfer2.5 T2 run on another machine (pass 1 writes a job to '
                                                           '--transfer_jobs; pass 2 composes on --transfer_results/<episode>.mp4)')
    ap.add_argument('--transfer_jobs', default=f'{H2W_OUTPUT_ROOT}/transfer_jobs/jobs')
    ap.add_argument('--transfer_results', default=f'{H2W_OUTPUT_ROOT}/transfer_jobs/results')
    ap.add_argument('--variant', default='', help="optional scene-variant suffix")
    ap.add_argument('--inputs', default='sim', choices=['sim', 'real'], help="sim: the scene reconstruction gen_scene.py made from the twin's head render "
                   "(explore/task_inputs/sim_<scene>); real: the pipeline's own reconstruction from the REAL head frame of a converted demo "
                   "(explore/task_inputs/<scene>, pipeline/run_episode.py), for real-robot training data (use the real --gripper_tex)")
    ap.add_argument('--out', default=None)
    ap.add_argument('--keep', action='store_true', help='keep the per-episode render folders')
    ap.add_argument('--shadow', type=int, default=1, help='1: compose_wrist also draws the shadows the gripper, the active arm and the held object cast, from a light randomized per episode (tilt 0-35 deg, any azimuth, darkening 0.3-0.6, 15 deg penumbra); the per-frame arm joints, the base pose and the light go into task_plan.npz')
    ap.add_argument('--body', type=int, default=1, help="1: also the robot's own body layer (rendering/robot_body.py: idle arm, chest, head, column), hidden by the table top; written to <task_inputs>/body_layer.npz for compose_wrist")
    ap.add_argument('--obj_noise', default=None, help="'pos_sigma_m,yaw_sigma_deg,scale_frac': the grasped object's start pose and size as an "
                    'estimate from the head frame would give them (review W1): per episode (seeded by its index) a Gaussian XY offset, a '
                    'Gaussian yaw about the table normal and a uniform scale in [1-s, 1+s]; logged to <out>/obj_noise_<ep>.json')
    ap.add_argument('--obj_size', type=float, default=None, help='optional measured-size override (m): uniformly scale the grasped mesh about its bottom centre to this median box dimension; unset for the paper floor recipe, which preserves native reconstructed size')
    ap.add_argument('--gripper_tex', default=f'{H2W_DATA_ROOT}/assets/gripper_texture_sim.npz', help='gripper layer colours: the sim-baked texture (gripper_texture_sim.py) - the policy is tested on the sim camera; the real texture (assets/gripper_texture.npz) is for the real robot')
    a = ap.parse_args()
    sname = a.scene.replace('/', '_')
    TW = json.load(open(f'{H2W_OUTPUT_ROOT}/sim_twin/{sname}/twin.json'))
    RB = TW['robot']
    OUT = a.out or f'{H2W_OUTPUT_ROOT}/sim_policy/{sname}/gen/wrist_{a.renderer}'
    os.makedirs(OUT, exist_ok=True)
    LI = f'{H2W_OUTPUT_ROOT}/lyra2'
    pre = 'sim_' if a.inputs == 'sim' else ''
    I0, T0 = f'{LI}/explore_inputs/{pre}{sname}{a.variant}', f'{LI}/task_inputs/{pre}{sname}{a.variant}'
    Th = np.array(TW['human_head_world'])
    Thi = np.linalg.inv(Th)
    Tb = np.array(RB['base_pose'])
    M = np.array(TW['link7_d405'])
    side = TW['side']
    lift = min(max(RB['lift'], -0.5 + 1e-4), -1e-4)
    m = K.reduced(lift, side)
    dm = m.createData()
    f7, ft = m.getFrameId(f'arm_{side[0]}_link7'), m.getFrameId('tcp')
    f_ = 960/np.tan(np.radians(87.0/2))
    K1920 = np.array([[f_, 0, 960], [0, f_, 540], [0, 0, 1.]])
    if a.body:
        import head2wrist.rendering.robot_body as RBD; import head2wrist.rendering.gripper_render as GRD
        _Kw = GRD.intrinsics(424, 240); _v, _u = np.mgrid[0:240, 0:424]; DIRS = np.stack([(_u + 0.5 - _Kw[0, 2])/_Kw[0, 0], (_v + 0.5 - _Kw[1, 2])/_Kw[1, 1], np.ones_like(_u, float)], -1)
    n_c = Thi[:3, :3] @ np.array([0, 0, 1.])
    d_c = float(Th[2, 3] - TW['table_h'])
    g = np.load(f'{H2W_OUTPUT_ROOT}/sim_twin/{sname}/meshes/grasped.npz')
    c0 = np.array(next(o for o in TW['objects'] if o['role'] == 'grasped')['centre'])
    if a.obj_size:                                            # measured size: uniform scale about the bottom centre (it keeps resting on the table)
        import cv2 as _cv2; _V = np.asarray(g['V'], float); (_cx, _cy), (_w, _l), _ = _cv2.minAreaRect(_V[:, :2].astype(np.float32))
        _h = float(_V[:, 2].max() - _V[:, 2].min()); _s = a.obj_size/float(np.median([_w, _l, _h])); _cb = np.array([_cx, _cy, _V[:, 2].min()])
        g = dict(g); g['V'] = _cb + _s*(_V - _cb)
        print(f'object size: box {100*_w:.1f} x {100*_l:.1f} x {100*_h:.1f} cm -> scaled by {_s:.3f} to {100*a.obj_size:.1f} cm', flush=True)
    PL = {p['i']: p for f in glob.glob(f'{H2W_OUTPUT_ROOT}/sim_policy/{sname}/plans*.json') for p in []}
    files = sorted(glob.glob(f'{a.demos}/ep_*.npz'))
    if a.episodes: keep = {int(x) for x in a.episodes.split(',')}; files = [f for f in files if int(os.path.basename(f)[3:7]) in keep]
    for f in files:
        i = int(os.path.basename(f)[3:7])
        try: z = np.load(f); okk = bool(z['success'])
        except Exception as e: print('ep', i, 'unreadable (collector stopped mid-write?), skipped:', e); continue
        if not okk: continue
        if os.path.exists(f'{OUT}/ep_{i:04d}.npy'): print('ep', i, 'done'); continue
        t_ = time.time(); tag = os.path.basename(os.path.dirname(f)); ename = f'sim_{sname}{a.variant}_{tag}_e{i:04d}'; E = f'sim/{sname}{a.variant}_{tag}_e{i:04d}'
        I, T = f'{LI}/explore_inputs/{ename}', f'{LI}/task_inputs/{ename}'
        for d in (I, T): os.makedirs(d, exist_ok=True)
        for fn in ('first_frame.png', 'first_depth.npy'): link(f'{I0}/{fn}', f'{I}/{fn}'); link(f'{T0}/{fn}', f'{T}/{fn}')
        for fn in ('object_mask.npy', 'table_extension.npz'): link(f'{T0}/{fn}', f'{T}/{fn}')
        link(f'{T0}/proxy', f'{T}/proxy')
        # camera path from the recorded arm state (the instant the sim wrist camera captured)
        C2W, TCP = [], []; Tb_e = np.array(z['base_pose'], float) if 'base_pose' in z else Tb     # stance-randomized demos: this episode's base
        for q in z['state'][:, :7]:
            pin.framesForwardKinematics(m, dm, q.astype(float))
            C2W.append(Thi @ Tb_e @ dm.oMf[f7].homogeneous @ M); TCP.append(Thi @ Tb_e @ dm.oMf[ft].homogeneous)
        C2W, TCP = np.stack(C2W), np.stack(TCP); ph = [str(p) for p in z['phase']]; N = len(ph)
        extra = {}
        if a.shadow:                                              # a light per episode (seeded by the episode): direction in the head frame
            rs = np.random.default_rng(1000 + i); tilt, az = np.radians(rs.uniform(0, 35)), rs.uniform(0, 2*np.pi)
            Lw = -np.array([np.sin(tilt)*np.cos(az), np.sin(tilt)*np.sin(az), np.cos(tilt)])                 # travel direction, world (down)
            extra = dict(arm_q=z['state'][:, :7].astype(np.float64), T_head_base=Thi @ Tb_e, lift=lift, side=side,
                         light_dir=Thi[:3, :3] @ Lw, shadow_strength=rs.uniform(0.3, 0.6), penumbra_deg=15.0)
        np.savez(f'{T}/task_plan.npz', c2w=C2W, tcp=TCP, phase=np.array(ph), grasp_frame=ph.index('grasp'), tcp_grasp=TCP[ph.index('grasp')], dive_frames=0, **extra)
        if a.body:
            # the robot's own body as the true wrist camera sees it, hidden where the table top is in front (the planner's
            # table: the twin's edge / far / width / height, which the stance was planned against; the generated background
            # continues the table as an infinite plane, which would hide the chest and arm standing in front of its edge)
            tb = TW['table']; BB, BA, BZ = [], [], []
            for q in z['state'][:, :7]:
                qf = RBD.full_q(side, q.astype(float), lift); dF = K.FULL.createData(); pin.framesForwardKinematics(K.FULL, dF, qf)
                Tc = dF.oMf[K.FULL.getFrameId(f'arm_{side[0]}_link7')].homogeneous @ M; bgr, al, dep = RBD.render(side, qf, Tc)
                Cw = Tb_e @ Tc; rw = DIRS @ Cw[:3, :3].T; sp = (TW['table_h'] - Cw[2, 3])/np.where(np.abs(rw[..., 2]) > 1e-9, rw[..., 2], 1e-9)
                Xw = Cw[:3, 3] + sp[..., None]*rw; on = (sp > 0) & (Xw[..., 0] >= tb['edge_x']) & (Xw[..., 0] <= tb['far_x']) & (np.abs(Xw[..., 1] - tb['y']) <= tb['width']/2)
                al = al*~(on & (sp < dep)); BB.append(bgr); BA.append(np.round(255*al).astype(np.uint8)); BZ.append(np.minimum(dep, 60).astype(np.float16))
            np.savez_compressed(f'{T}/body_layer.npz', bgr=np.stack(BB), alpha=np.stack(BA), depth=np.stack(BZ))
        elif os.path.exists(f'{T}/body_layer.npz'): os.remove(f'{T}/body_layer.npz')
        np.savez(f'{T}/trajectory.npz', w2c=np.linalg.inv(C2W).astype(np.float32), intrinsics=np.tile(K1920[None], (N, 1, 1)).astype(np.float32), image_height=1080, image_width=1920)
        meta = dict(ep=E, t0=0, side=side, n_frames=N, dive_frames=0, sim_episode=f)
        # the head camera of the SOURCE scene: without head_focal headcam.focal() falls back to EgoDex's 736.6 px, and render_proxy /
        
        _m0 = json.load(open(f'{T0}/meta.json')) if os.path.exists(f'{T0}/meta.json') else {}
        meta.update({k_: _m0[k_] for k_ in ('head_focal', 'capture_device') if k_ in _m0})
        for d in (I, T): json.dump(meta, open(f'{d}/meta.json', 'w'))
        # the grasped object at this episode's start pose (the recorded object pose at frame 0), head frame
        q0 = z['object'][0, 3:]; from scipy.spatial.transform import Rotation as R
        Rw = R.from_quat([q0[1], q0[2], q0[3], q0[0]]).as_matrix(); p0 = np.array(z['object'][0, :3], float); sc = 1.0
        if a.obj_noise:                                       # an imperfect object estimate instead of the twin's exact one
            ps, ys, ss = [float(x) for x in a.obj_noise.split(',')]; rn = np.random.default_rng(7919 + i)
            dxy = rn.normal(0, ps, 2); dyaw = rn.normal(0, ys); sc = rn.uniform(1 - ss, 1 + ss)
            Rw = R.from_euler('z', dyaw, degrees=True).as_matrix() @ Rw; p0 = p0 + np.r_[dxy, 0.0]
            json.dump(dict(dx_cm=100*float(dxy[0]), dy_cm=100*float(dxy[1]), dyaw_deg=float(dyaw), scale=float(sc), sigma=a.obj_noise),
                      open(f'{OUT}/obj_noise_{i:04d}.json', 'w'))   # one file per episode: the workers run in parallel
        Vw = sc*(g['V'] - c0) @ Rw.T + p0
        np.savez(f'{T}/object_mesh.npz', V=(Vw - Th[:3, 3]) @ Thi[:3, :3].T, F=g['F'], C=g['rgb'], n=n_c, d=d_c)
        # render + compose
        rv = f'rv5_{tag}_e{i:04d}'; cmds = [[PY, f'{X}/scene/render_proxy.py', '--ep', E, '--out_name', rv]]; bg = f'{T0}/proxy/{rv}/rgb.mp4'
        if a.renderer == 'transfer':                            # Blender v3 + Transfer2.5 T2 (run_transfer.py's T2 spec), two passes
            rb = f'rb3_{tag}_e{i:04d}'; bg = f'{a.transfer_results}/{ename}.mp4'
            if not os.path.exists(bg):
                if os.path.exists(f'{a.transfer_jobs}/{ename}/spec.json'): print(f'ep {i}: Transfer job waiting', flush=True); continue
                for c in (cmds[0], [f'{REPO}/third_party/blender/.venv/bin/python', f'{H2W_REPO_ROOT}/baselines/blender_proxy.py', '--ep', E, '--out_name', rb, '--passes_from', rv]):
                    r = subprocess.run(c, capture_output=True, text=True)
                    if r.returncode: print('ep', i, os.path.basename(c[1]), 'failed:', r.stderr[-1500:]); sys.exit(1)
                J = f'{a.transfer_jobs}/{ename}'; os.makedirs(J, exist_ok=True)
                for fn in ('rgb.mp4', 'depth.mp4', 'seg.mp4', 'guided_mask.mp4'): shutil.copy(os.path.realpath(f'{T0}/proxy/{rb}/{fn}'), f'{J}/{fn}')
                from baselines.prompts import get_prompt
                json.dump(dict(name=ename, prompt=get_prompt(E), video_path='rgb.mp4', guidance=3, seed=1,
                               depth=dict(control_path='depth.mp4', control_weight=1.0), seg=dict(control_path='seg.mp4', control_weight=0.5),
                               edge=dict(control_weight=0.3), vis=dict(control_weight=0.5, preset_blur_strength='medium'),
                               guided_generation_mask='guided_mask.mp4', guided_generation_step_threshold=25), open(f'{J}/spec.json', 'w'), indent=1)
                json.dump(dict(n_frames=N, episode=i, demos=a.demos), open(f'{J}/job.json', 'w'))
                for x in (rv, rb): subprocess.run(['rm', '-rf', f'{T0}/proxy/{x}'])
                print(f'ep {i}: Transfer job ready ({N} frames) -> {J}', flush=True); continue
            cmds = []                                           # pass 2: only compose on the Transfer background
        if a.renderer == 'blender':
            rb = f'rb3_{tag}_e{i:04d}'; bg = f'{T0}/proxy/{rb}/rgb.mp4'
            cmds.append([f'{REPO}/third_party/blender/.venv/bin/python', f'{H2W_REPO_ROOT}/baselines/blender_proxy.py', '--ep', E, '--out_name', rb, '--passes_from', rv])
        cmds.append([PY, f'{X}/rendering/compose_wrist.py', '--ep', E, '--bg', bg, '--tag', f'_gen_{a.renderer}', '--gripper_tex', a.gripper_tex])
        for c in cmds:
            r = subprocess.run(c, capture_output=True, text=True)
            if r.returncode: print('ep', i, os.path.basename(c[1]), 'failed:', r.stderr[-1500:]); sys.exit(1)
        cap = cv2.VideoCapture(f'{H2W_OUTPUT_ROOT}/composite/{ename}_gen_{a.renderer}/wrist_upright.mp4'); fr = []
        while True:
            ok, im = cap.read()
            if not ok: break
            fr.append(cv2.resize(cv2.rotate(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), cv2.ROTATE_90_CLOCKWISE), (424, 240), interpolation=cv2.INTER_AREA))
        assert len(fr) == N, (len(fr), N)
        np.save(f'{OUT}/ep_{i:04d}.npy', np.stack(fr).astype(np.uint8))
        if not a.keep and a.renderer != 'transfer':
            for x in [rv] + ([rb] if a.renderer == 'blender' else []): subprocess.run(['rm', '-rf', f'{T0}/proxy/{x}'])
        print(f'ep {i}: {N} frames in {time.time() - t_:.0f} s', flush=True)


if __name__ == '__main__':
    main()
