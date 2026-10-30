from head2wrist.config import H2W_OUTPUT_ROOT
#!/usr/bin/env python3
"""
Isaac Sim twin of one demo's scene + the planned robot motion, in the robot world frame (sg2_kin.Scene: z up, table
top at table_h, x = head heading). Everything the sim side (isaac_replay.py) needs, nothing Isaac-specific:
  objects   every static proxy object (proxy/obj<k>.npz, colours = real head pixels where the head saw them, fill
            elsewhere) and the grasped object (object_mesh.npz) at its start pose; decimated to --faces triangles;
            meshes/<name>.obj (+ .npz with world vertices / faces / rgb)
  table     the table top: a box from the table edge to past the farthest object; walls as planes (proxy.json)
  robot     base pose (fixed), lift, per wrist frame the 7 arm joints (robot_plan.npz), the gripper joint (the
            compose_wrist schedule: pre-grasp opening, closing over 7 frames at the grasp, open again at release), phase
  checks    planned TCP and D405 camera poses per frame (world), the object's planned pose at release (rigid with the
            TCP from the grasp on), for the sim side to compare against
Writes $H2W_OUTPUT_ROOT/sim_twin/<name>/{twin.json, meshes/}.
"""
import argparse, json, os, sys
import numpy as np, open3d as o3d
X = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import head2wrist.robot.sg2_kin as K
import head2wrist.rendering.gripper_render as GR
import pinocchio as pin
from head2wrist.wrist_pose import T_TCP_D405



def support_below(Vw, others):
    """top of the other objects' meshes under this object's footprint (its xy box shrunk 20%), else the table"""
    lo, hi = Vw[:, :2].min(0), Vw[:, :2].max(0); c, h = (lo + hi)/2, 0.4*(hi - lo); top = S.table_h
    for P in others:
        ins = (np.abs(P[:, :2] - c) < h).all(1) & (P[:, 2] < Vw[:, 2].min() + 0.01)
        if ins.sum() > 20: top = max(top, float(np.percentile(P[ins, 2], 95)))
    return top


def save_mesh(nm, V, F, rgb, others=()):
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(cw(V)), o3d.utility.Vector3iVector(F.astype(np.int32)))
    m.vertex_colors = o3d.utility.Vector3dVector(rgb[:, :3].astype(np.float64)/255)
    if len(F) > a.faces: m = m.simplify_quadric_decimation(a.faces)
    m.remove_degenerate_triangles(); m.remove_unreferenced_vertices()
    # rest it on what is under it (1 mm above: physics pushes interpenetrating bodies out): the table, or another
    # object under its footprint (pick_place_food/11: a food item on a plate, which a table-only rule pushed 2 cm into
    # the plate); an object below the table always comes up to it. The shift = how far the reconstruction was off
    Vw0 = np.asarray(m.vertices); dz = S.table_h + 0.001 - Vw0[:, 2].min()
    if dz < 0: dz = support_below(Vw0, others) + 0.001 - Vw0[:, 2].min()
    m.translate((0, 0, dz)); m.compute_vertex_normals()
    o3d.io.write_triangle_mesh(f'{OUT}/meshes/{nm}.obj', m, write_vertex_colors=True)
    Vw, Fw, Cw = np.asarray(m.vertices), np.asarray(m.triangles), (np.asarray(m.vertex_colors)*255).astype(np.uint8)
    np.savez(f'{OUT}/meshes/{nm}.npz', V=Vw, F=Fw, rgb=Cw)
    lo, hi = Vw.min(0), Vw.max(0)
    return dict(name=nm, obj=f'meshes/{nm}.obj', centre=((lo + hi)/2).tolist(), size=(hi - lo).tolist(),
                snap_to_table_cm=100*float(dz), faces=int(len(Fw)))


def main():
    global AP_, CAMw, DIVE, FKT, L7, M, OUT, P, PX, R, S, T, TCPw, T_l7_d405, T_rel, Tb, Vo, Vt, a, ap, band, c, c0, c_place, cam_err_mm, cw, dm, fk_err_mm, fl7, ftcp, g, g_close, g_open, i, ig, ir, k, m, mount_spread_mm, name, nm, o, ob, objs, p, phase, q, raw, rgb, setup, side, sn, ss, t, twin, v, w, width, z
    ap = argparse.ArgumentParser()
    ap.add_argument('--ep', required=True)
    ap.add_argument('--faces', type=int, default=20000, help='triangles per object mesh after decimation')
    ap.add_argument('--fps', type=float, default=16.0)
    a = ap.parse_args()
    name = a.ep.replace('/', '_')
    T = f'{H2W_OUTPUT_ROOT}/lyra2/task_inputs/{name}'
    PX = f'{T}/proxy'
    OUT = f'{H2W_OUTPUT_ROOT}/sim_twin/{name}'
    os.makedirs(f'{OUT}/meshes', exist_ok=True)
    S = K.Scene(T)
    setup = json.load(open(f'{T}/robot_setup.json'))['summary']['setup']
    side = json.load(open(f'{T}/meta.json')).get('side', 'right')
    cw = lambda P: (S.cam_to_world(np.eye(4))[:3, :3] @ P.T).T + S.cam_to_world(np.eye(4))[:3, 3]
    objs = []
    raw = {}
    for ob in json.load(open(f'{PX}/proxy.json'))['objects']:
        raw[f"obj{ob['k']}"] = cw(np.load(f"{PX}/obj{ob['k']}.npz")['V'][::20])
    raw['grasped'] = cw(np.load(f'{T}/object_mesh.npz')['V'][::20])
    for ob in json.load(open(f'{PX}/proxy.json'))['objects']:
        z = np.load(f"{PX}/obj{ob['k']}.npz"); sn = f"{PX}/obj{ob['k']}_seen.npz"; nm = f"obj{ob['k']}"
        rgb = np.load(sn)['fill'] if os.path.exists(sn) else z['C']
        objs.append(dict(save_mesh(nm, z['V'], z['F'], rgb, [v for k, v in raw.items() if k != nm]), role='static'))
    z = np.load(f'{T}/object_mesh.npz')
    objs.append(dict(save_mesh('grasped', z['V'], z['F'], z['C'], [v for k, v in raw.items() if k != 'grasped']), role='grasped'))
    P = np.load(f'{T}/task_plan.npz')
    DIVE = int(P['dive_frames'])
    phase = [str(p) for p in P['phase'][DIVE:]]
    TCPw = np.stack([S.cam_to_world(t) for t in P['tcp'][DIVE:]])
    CAMw = np.stack([S.cam_to_world(c) for c in P['c2w'][DIVE:]])
    R = np.load(f'{T}/robot_plan.npz')
    assert len(R['arm_q']) == len(phase), (len(R['arm_q']), len(phase))
    Tb = np.array(setup['base_pose_world'])
    m = K.reduced(float(R['lift'][0]), side)
    dm = m.createData()
    fl7, ftcp = m.getFrameId(f'arm_{side[0]}_link7'), m.getFrameId('tcp')
    L7, FKT = [], []
    for q in R['arm_q']:
        pin.framesForwardKinematics(m, dm, q); L7.append(Tb @ dm.oMf[fl7].homogeneous); FKT.append(Tb @ dm.oMf[ftcp].homogeneous)
    L7, FKT = np.stack(L7), np.stack(FKT)
    fk_err_mm = 1000*np.linalg.norm(FKT[:, :3, 3] - TCPw[:, :3, 3], axis=1)
    M = np.stack([np.linalg.inv(L7[i]) @ TCPw[i] @ T_TCP_D405 for i in range(len(L7))])
    T_l7_d405 = M[len(M)//2]
    mount_spread_mm = 1000*float(np.ptp(M[:, :3, 3], axis=0).max())
    cam_err_mm = 1000*np.linalg.norm(np.stack([t @ T_TCP_D405 for t in TCPw])[:, :3, 3] - CAMw[:, :3, 3], axis=1)
    Vo = np.load(f'{OUT}/meshes/grasped.npz')['V']
    ig, ir = phase.index('grasp'), phase.index('release')
    Vt = (Vo - TCPw[ig][:3, 3]) @ TCPw[ig][:3, :3]
    band = np.abs(Vt[:, 2]) < 0.012
    width = float(np.ptp(Vt[band, 1])) if band.sum() > 20 else float(np.ptp(Vt[:, 1]))
    AP_ = np.load(f'{H2W_OUTPUT_ROOT}/lyra2/explore_inputs/{name}/approach_plan.npz')
    if 'gripper_close' in AP_: width = float(AP_['object_width'])
    g_close = GR.joint_for_gap(width)
    g_open = GR.joint_for_gap(min(width + 0.04, GR.jaw_gap(0.0)))
    ss = lambda x: x*x*(3 - 2*x)
    g = np.full(len(phase), g_open)
    for i, p in enumerate(phase):
        if p == 'grasp': g[i] = g_open + (g_close - g_open)*ss(min(1.0, (i - ig)/7))
        elif p in ('lift-off', 'transport', 'set-down'): g[i] = g_close
        elif p == 'release': g[i] = g_close + (g_open - g_close)*ss(min(1.0, (i - ir)/7))
    T_rel = TCPw[ir] @ np.linalg.inv(TCPw[ig])
    c0 = np.array(objs[-1]['centre']) - [0, 0, objs[-1]['snap_to_table_cm']/100]
    c_place = T_rel[:3, :3] @ c0 + T_rel[:3, 3]
    twin = dict(ep=a.ep, side=side, fps=a.fps, table_h=S.table_h,
                table=dict(edge_x=float((S.origin + setup['table_edge_m']*S.f)[0]), y=float(S.origin[1]),
                           far_x=float(max(o['centre'][0] + o['size'][0]/2 for o in objs) + 0.25), width=1.6, thickness=0.04,
                           rgb=json.load(open(f'{PX}/proxy.json'))['table']['colour']),
                walls=[dict(n=(S.cam_to_world(np.eye(4))[:3, :3] @ np.array(w['n'])).tolist(),
                            point=cw(-w['d']*np.array(w['n'])[None])[0].tolist(), rgb=w['colour'])
                       for w in json.load(open(f'{PX}/proxy.json'))['walls']],
                objects=objs,
                robot=dict(base_pose=setup['base_pose_world'], lift=float(R['lift'][0]), lift_const=bool(np.ptp(R['lift']) < 1e-6),
                           arm_joints=K.ARM(side), arm_q=R['arm_q'].tolist(), gripper_q=g.tolist(), gripper_close=g_close,
                           gripper_open=g_open, gripper_max=GR.G_MAX, object_width_m=width, phase=phase),
                link7_d405=T_l7_d405.tolist(),
                human_head_world=S.cam_to_world(np.eye(4)).tolist(),            # the EgoDex head camera at the input frame (OpenCV axes)
                human_head_K_1920x1080=[[736.6, 0, 960.0], [0, 736.6, 540.0], [0, 0, 1]],
                check=dict(tcp_world=TCPw.tolist(), link7_world=L7.tolist(), fk_tcp_err_mm_max=float(fk_err_mm.max()),
                           d405_mount_spread_mm=mount_spread_mm, d405_vs_tcp_mount_err_mm_max=float(cam_err_mm.max()), d405_world=CAMw.tolist(), d405_K_424x240=GR.intrinsics().tolist(),
                           grasp_frame=ig, release_frame=ir, object_start=objs[-1]['centre'], object_place_planned=c_place.tolist()))
    json.dump(twin, open(f'{OUT}/twin.json', 'w'), indent=1)
    print(f"{a.ep}: {len(objs)} objects ({', '.join(o['name'] for o in objs)}), {len(phase)} frames, grasp {ig}, release {ir}, "
          f"width {100*width:.1f} cm; base {np.array(setup['base_pose_world'])[:2, 3].round(3)}, lift {R['lift'][0]:.2f}; "
          f"object {c0.round(3)} -> planned {c_place.round(3)}; FK vs planned TCP max {fk_err_mm.max():.1f} mm, D405 mount spread "
          f"{mount_spread_mm:.1f} mm, planned camera vs TCP mount {cam_err_mm.max():.1f} mm -> {OUT}")


if __name__ == '__main__':
    main()
