from head2wrist.config import H2W_OUTPUT_ROOT
#!/usr/bin/env python3
"""
Generator v5 inputs for a REAL episode (pipeline stage 'bodyshadow'; the sim path does the same in gen_episode.py):
  body layer   the robot's own body as the wrist camera sees it (rendering/robot_body.py), from the planned arm joints
               (robot_plan.npz), the planned stance and lift, hidden where the table top is in front (the planner's
               finite table: the twin export's edge / far / width / height) -> <task_inputs>/body_layer.npz
  shadows      a light randomized per episode (tilt 0-35 deg, any azimuth, darkening 0.3-0.6, 15 deg penumbra), the arm
               joints per frame and the base pose in the head frame -> added to <task_inputs>/task_plan.npz
compose_wrist.py draws both when it finds them. Needs the twin export (stage 'simgate': sim_twin/<name>/twin.json).
  body_shadow_inputs.py --ep <task/idx> [--body 1] [--shadow 1] [--seed 0]
"""
import argparse, json, os, sys, zlib
import numpy as np, pinocchio as pin
HERE = os.path.dirname(os.path.abspath(__file__))
import head2wrist.rendering.robot_body as RBD; import head2wrist.rendering.gripper_render as GR; import head2wrist.robot.sg2_kin as K





def main():
    global BA, BB, BZ, Cw, DIRS, DIVE, Kw, Lw, M, N, P, Q, Qn, RP, T, TW, Tb, Tc, Thi, Thw, Xw, a, al, ap, az, b, bgr, dF, dep, i, k, lift, name, on, qf, rs, rw, side, sp, tb, tilt, u_, v_
    ap = argparse.ArgumentParser()
    ap.add_argument('--ep', required=True)
    ap.add_argument('--body', type=int, default=1)
    ap.add_argument('--shadow', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    name = a.ep.replace('/', '_')
    T = f'{H2W_OUTPUT_ROOT}/lyra2/task_inputs/{name}'
    TW = json.load(open(f'{H2W_OUTPUT_ROOT}/sim_twin/{name}/twin.json'))
    side = TW['side']
    P = dict(np.load(f'{T}/task_plan.npz'))
    N = len(P['phase'])
    DIVE = int(P['dive_frames'])
    RP = np.load(f'{T}/robot_plan.npz')
    Q = RP['arm_q'].astype(float)
    lift = float(RP['lift'][0])
    assert len(Q) == N - DIVE, (len(Q), N, DIVE)
    Tb = np.array(TW['robot']['base_pose'], float)
    Thw = np.array(TW['human_head_world'], float)
    Thi = np.linalg.inv(Thw)
    M = np.array(TW['link7_d405'], float)
    Qn = np.concatenate([np.repeat(Q[:1], DIVE, 0), Q])
    if a.body:
        Kw = GR.intrinsics(424, 240); v_, u_ = np.mgrid[0:240, 0:424]; DIRS = np.stack([(u_ + 0.5 - Kw[0, 2])/Kw[0, 0], (v_ + 0.5 - Kw[1, 2])/Kw[1, 1], np.ones_like(u_, float)], -1)
        tb = TW['table']; BB, BA, BZ = [], [], []
        for i in range(N):
            if i < DIVE:                                                             # compose starts at the dive's end
                BB.append(np.zeros((240, 424, 3), np.uint8)); BA.append(np.zeros((240, 424), np.uint8)); BZ.append(np.full((240, 424), 60, np.float16)); continue
            qf = RBD.full_q(side, Qn[i], lift); dF = K.FULL.createData(); pin.framesForwardKinematics(K.FULL, dF, qf)
            Tc = dF.oMf[K.FULL.getFrameId(f'arm_{side[0]}_link7')].homogeneous @ M; bgr, al, dep = RBD.render(side, qf, Tc)
            Cw = Tb @ Tc; rw = DIRS @ Cw[:3, :3].T; sp = (TW['table_h'] - Cw[2, 3])/np.where(np.abs(rw[..., 2]) > 1e-9, rw[..., 2], 1e-9)
            Xw = Cw[:3, 3] + sp[..., None]*rw; on = (sp > 0) & (Xw[..., 0] >= tb['edge_x']) & (Xw[..., 0] <= tb['far_x']) & (np.abs(Xw[..., 1] - tb['y']) <= tb['width']/2)
            al = al*~(on & (sp < dep)); BB.append(bgr); BA.append(np.round(255*al).astype(np.uint8)); BZ.append(np.minimum(dep, 60).astype(np.float16))
        np.savez_compressed(f'{T}/body_layer.npz', bgr=np.stack(BB), alpha=np.stack(BA), depth=np.stack(BZ))
        print(f'{a.ep}: body layer, {N - DIVE} frames, mean coverage {np.mean([b.mean()/255 for b in BA[DIVE:]]):.3f}')
    elif os.path.exists(f'{T}/body_layer.npz'): os.remove(f'{T}/body_layer.npz')
    for k in ('arm_q', 'T_head_base', 'lift', 'side', 'light_dir', 'shadow_strength', 'penumbra_deg'): P.pop(k, None)
    if a.shadow:
        rs = np.random.default_rng(zlib.crc32(name.encode()) + a.seed); tilt, az = np.radians(rs.uniform(0, 35)), rs.uniform(0, 2*np.pi)
        Lw = -np.array([np.sin(tilt)*np.cos(az), np.sin(tilt)*np.sin(az), np.cos(tilt)])                 # travel direction, world (z up)
        P.update(arm_q=Qn, T_head_base=Thi @ Tb, lift=lift, side=side, light_dir=Thi[:3, :3] @ Lw, shadow_strength=rs.uniform(0.3, 0.6), penumbra_deg=15.0)
        print(f'{a.ep}: light tilt {np.degrees(tilt):.0f} deg, azimuth {np.degrees(az):.0f} deg, strength {P["shadow_strength"]:.2f}')
    np.savez(f'{T}/task_plan.npz', **P)


if __name__ == '__main__':
    main()
