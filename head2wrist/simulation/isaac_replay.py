
def main():
    #!/usr/bin/env python3
    """
    Open-loop replay of one demo's planned FFW-SG2 motion in its Isaac Sim twin (export_twin.py), with physics.
    Runs inside the robotis_lab Isaac Sim 5.1 / Isaac Lab 2.3 image (run_isaac.sh); the robot is robotis_lab's FFW_SG2_CFG
    with the root link fixed at the planned base pose, lift at the planned height.
      scene   table (static box, the measured table colour), walls (visual), every object as a dynamic rigid body from its
              reconstructed mesh (convex decomposition; real head-pixel colours via displayColor)
      motion  the 7 arm joints and gripper joint1 of the planned path, linearly interpolated between the 16 fps plan frames
              at the physics rate; the gripper squeezes past the planned contact opening by --squeeze (rad) so the pads
              press on the object instead of just touching it
      sensors a D405 on the wrist link7 at the planned mount (424x240, 87 deg), an overview camera; rendered per plan frame
    Writes <twin>/isaac/: replay.json (tracking error vs the planned link7, lifted / placed / static objects disturbed),
    frames.npz (wrist + overview RGB, object poses per frame).
    """
    import argparse, json, math, os
    from isaaclab.app import AppLauncher

    ap = argparse.ArgumentParser()
    ap.add_argument('--twin', required=True)
    ap.add_argument('--squeeze', type=float, default=0.15, help='gripper joint past the contact opening (rad)')
    ap.add_argument('--obj_mass', type=float, default=0.05, help='grasped object mass (kg)')
    ap.add_argument('--static_mass', type=float, default=0.3, help='mass of the other objects (kg)')
    ap.add_argument('--friction', type=float, default=1.0)
    ap.add_argument('--substeps', type=int, default=12, help='physics steps per plan frame (16 fps -> 192 Hz)')
    ap.add_argument('--settle_s', type=float, default=3.0, help='objects settle (and overlapping reconstructions separate) before the motion')
    ap.add_argument('--hold_s', type=float, default=1.5)
    AppLauncher.add_app_launcher_args(ap)
    a = ap.parse_args(); a.headless = True; a.enable_cameras = True
    app = AppLauncher(a).app

    try:
        import numpy as np, torch
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
        from isaaclab.sensors import Camera, CameraCfg
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Vt, PhysxSchema
        from scipy.spatial.transform import Rotation
        from robotis_lab.assets.robots.FFW_SG2 import FFW_SG2_CFG

        TW = json.load(open(f'{a.twin}/twin.json')); RB = TW['robot']; CK = TW['check']; side = TW['side'][0]
        OUT = f'{a.twin}/isaac'; os.makedirs(f'{OUT}/usd', exist_ok=True)
        lin = lambda rgb: tuple(((np.array(rgb)/255.)**2.2).tolist())                           # sRGB -> linear (USD colours)
        wxyz = lambda R_: tuple(np.roll(Rotation.from_matrix(R_).as_quat(), 1).tolist())          # scipy xyzw -> wxyz


        def object_usd(nm, mass, approx):
            """the reconstructed mesh as a rigid body USD; origin = mesh centre (the spawn position)"""
            z = np.load(f'{a.twin}/meshes/{nm}.npz'); V, F, C = z['V'], z['F'], z['rgb']; c = (V.min(0) + V.max(0))/2
            path = f'{OUT}/usd/{nm}.usda'; st = Usd.Stage.CreateNew(path)
            UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(st, 1.0)
            root = UsdGeom.Xform.Define(st, '/obj'); st.SetDefaultPrim(root.GetPrim())
            mesh = UsdGeom.Mesh.Define(st, '/obj/geom')
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy((V - c).astype(np.float32)))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(F), 3, np.int32)))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(F.reshape(-1).astype(np.int32)))
            mesh.CreateSubdivisionSchemeAttr('none')
            mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray.FromNumpy(((C/255.)**2.2).astype(np.float32)))
            mat = UsdShade.Material.Define(st, '/obj/look'); pbr = UsdShade.Shader.Define(st, '/obj/look/pbr'); pbr.CreateIdAttr('UsdPreviewSurface')
            rd = UsdShade.Shader.Define(st, '/obj/look/col'); rd.CreateIdAttr('UsdPrimvarReader_float3')
            rd.CreateInput('varname', Sdf.ValueTypeNames.String).Set('displayColor'); rd.CreateOutput('result', Sdf.ValueTypeNames.Float3)
            pbr.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(rd.ConnectableAPI(), 'result')
            pbr.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.7)
            mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), 'surface')
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()); UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(approx)
            if approx == 'convexDecomposition':
                cd = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(mesh.GetPrim())
                cd.CreateMaxConvexHullsAttr(32); cd.CreateVoxelResolutionAttr(500000); cd.CreateShrinkWrapAttr(True)
            pc = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim()); pc.CreateContactOffsetAttr(0.003); pc.CreateRestOffsetAttr(0.0)
            pm = UsdShade.Material.Define(st, '/obj/physmat'); pma = UsdPhysics.MaterialAPI.Apply(pm.GetPrim())
            pma.CreateStaticFrictionAttr(a.friction); pma.CreateDynamicFrictionAttr(a.friction); pma.CreateRestitutionAttr(0.0)
            UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(pm, UsdShade.Tokens.weakerThanDescendants, 'physics')
            UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()); UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(mass)
            PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim()).CreateEnableCCDAttr(True)
            st.Save(); return path, c


        # ---- scene ----
        dt = 1.0/(TW['fps']*a.substeps)
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device='cuda:0', render_interval=a.substeps))
        sim_utils.GroundPlaneCfg().func('/World/ground', sim_utils.GroundPlaneCfg())
        dl = sim_utils.DomeLightCfg(intensity=700.0, color=(1.0, 0.95, 0.88)); dl.func('/World/dome', dl)
        lc = sim_utils.DistantLightCfg(intensity=900.0, angle=20.0); lc.func('/World/sun', lc, orientation=wxyz(Rotation.from_euler('y', 25, degrees=True).as_matrix()))
        tb = TW['table']; th = TW['table_h']; L = tb['far_x'] - tb['edge_x']
        tcfg = sim_utils.CuboidCfg(size=(L, tb['width'], tb['thickness']), collision_props=sim_utils.CollisionPropertiesCfg(),
                                   visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=lin(tb['rgb']), roughness=0.9),
                                   physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=a.friction, dynamic_friction=a.friction))
        tcfg.func('/World/table', tcfg, translation=(tb['edge_x'] + L/2, tb['y'], th - tb['thickness']/2))
        for k, w in enumerate(TW['walls']):                                  # visual only: what the wrist camera sees behind the table
            n = np.array(w['n'])/np.linalg.norm(w['n']); zax = n; xax = np.cross([0, 0, 1.], zax)
            xax = xax/np.linalg.norm(xax) if np.linalg.norm(xax) > 1e-3 else np.array([1., 0, 0]); yax = np.cross(zax, xax)
            wcfg = sim_utils.CuboidCfg(size=(3.0, 3.0, 0.01), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=lin(w['rgb']), roughness=1.0))
            wcfg.func(f'/World/wall{k}', wcfg, translation=tuple(w['point']), orientation=wxyz(np.stack([xax, yax, zax], 1)))
        objs = {}
        for o in TW['objects']:
            g = o['role'] == 'grasped'
            p, c = object_usd(o['name'], a.obj_mass if g else a.static_mass, 'convexHull' if g else 'convexDecomposition')
            objs[o['name']] = RigidObject(RigidObjectCfg(prim_path=f"/World/{o['name']}", spawn=sim_utils.UsdFileCfg(usd_path=p),
                                                         init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(c.tolist()))))
        # robot: fixed root at the planned base pose, planned lift, the arm at the first plan frame, the other arm at zero (as
        # the planner assumed)
        # the plan clamped into the robot USD's joint limits (basic_pick_place/1: arm_r_joint6 = 1.5708 vs 1.570, /12:
        # arm_r_joint1 = 3.1416 vs 3.140; Isaac refuses default positions outside them)
        from robotis_lab.assets.robots import ROBOTIS_LAB_ASSETS_DATA_DIR as _RD
        _st = Usd.Stage.Open(f'{_RD}/robots/FFW/FFW_SG2.usd'); LIM = {}
        for _p in _st.Traverse():
            if _p.IsA(UsdPhysics.RevoluteJoint):
                _j = UsdPhysics.RevoluteJoint(_p); lo_, hi_ = _j.GetLowerLimitAttr().Get(), _j.GetUpperLimitAttr().Get()
                if lo_ is not None and hi_ is not None: LIM[_p.GetName()] = (math.radians(lo_), math.radians(hi_))
            elif _p.IsA(UsdPhysics.PrismaticJoint):
                _j = UsdPhysics.PrismaticJoint(_p); lo_, hi_ = _j.GetLowerLimitAttr().Get(), _j.GetUpperLimitAttr().Get()
                if lo_ is not None and hi_ is not None: LIM[_p.GetName()] = (lo_, hi_)
        print('[replay] USD joint limits read for', len(LIM), 'joints; lift', LIM.get('lift_joint'), flush=True)
        clamped = {}
        def clampj(j, v):
            lo, hi = LIM.get(j, (-1e9, 1e9)); c = min(max(v, lo + 1e-4), hi - 1e-4)
            if abs(c - v) > 1e-6: clamped[j] = max(clamped.get(j, 0.0), abs(c - v))
            return c
        RB['arm_q'] = [[clampj(j, v) for j, v in zip(RB['arm_joints'], q)] for q in RB['arm_q']]; RB['lift'] = clampj('lift_joint', RB['lift'])
        Tb = np.array(RB['base_pose']); rcfg = FFW_SG2_CFG.replace(prim_path='/World/Robot')
        rcfg.spawn.articulation_props.fix_root_link = True
        jp = dict(rcfg.init_state.joint_pos); jp.update({'lift_joint': RB['lift'], f'gripper_{side}_joint1': RB['gripper_q'][0]})
        jp.update({j: float(q) for j, q in zip(RB['arm_joints'], RB['arm_q'][0])})
        rcfg.init_state = rcfg.init_state.replace(pos=tuple(Tb[:3, 3].tolist()), rot=wxyz(Tb[:3, :3]), joint_pos=jp)
        robot = Articulation(rcfg)
        stage = sim_utils.get_current_stage()
        l7 = next(str(p.GetPath()) for p in stage.Traverse() if str(p.GetPath()).startswith('/World/Robot') and p.GetName() == f'arm_{side}_link7')
        M = np.array(TW['link7_d405']); K = np.array(CK['d405_K_424x240'])
        wrist = Camera(CameraCfg(prim_path=f'{l7}/d405', width=424, height=240, data_types=['rgb'], update_period=0,
                                 spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(K.reshape(-1).tolist(), 424, 240, clipping_range=(0.005, 20.0)),
                                 offset=CameraCfg.OffsetCfg(pos=tuple(M[:3, 3].tolist()), rot=wxyz(M[:3, :3]), convention='ros')))
        over = Camera(CameraCfg(prim_path='/World/overview', width=640, height=480, data_types=['rgb'], update_period=0,
                                spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 20.0))))
        sim.reset()
        # overview: above and behind the robot's working shoulder, looking at the object (walls are beyond the table)
        o0 = np.array(CK['object_start']); b0 = Tb[:3, 3]; sgn = 1.0 if side == 'l' else -1.0
        eye = b0 + 0.25*(o0 - b0) + Tb[:3, :3] @ np.array([-0.35, 0.45*sgn, 0.0]); eye[2] = TW['table_h'] + 0.95
        over.set_world_poses_from_view(torch.tensor(np.array([eye]), device='cuda:0', dtype=torch.float32),
                                       torch.tensor(np.array([o0]), device='cuda:0', dtype=torch.float32))
        arm_ids, _ = robot.find_joints(RB['arm_joints'], preserve_order=True); g_ids, _ = robot.find_joints([f'gripper_{side}_joint1'])
        b7 = robot.find_bodies(f'arm_{side}_link7')[0][0]; lid = robot.find_joints('lift_joint')[0][0]
        # start AT the planned configuration (targets alone leave the joints at the USD zero pose)
        robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel))
        robot.set_joint_position_target(robot.data.default_joint_pos.clone())
        lim = robot.data.soft_joint_pos_limits[0, lid].tolist()
        print('[replay] link7 prim', l7, '| joints', len(robot.joint_names), '| bodies', len(robot.body_names),
              f'| lift target {RB["lift"]:.3f}, limits {lim}', flush=True)
        Q = torch.tensor(RB['arm_q'], device='cuda:0', dtype=torch.float32)
        gs, go, gc = np.array(RB['gripper_q']), RB['gripper_open'], RB['gripper_close']
        Gcmd = torch.tensor(np.clip(gs + a.squeeze*np.clip((gs - go)/max(gc - go, 1e-6), 0, 1), 0, RB['gripper_max']), device='cuda:0', dtype=torch.float32)


        def step(q, g, render):
            robot.set_joint_position_target(q[None], joint_ids=arm_ids); robot.set_joint_position_target(g.reshape(1, 1), joint_ids=g_ids)
            robot.write_data_to_sim(); sim.step(render=render); robot.update(dt)
            for ob in objs.values(): ob.update(dt)


        pose = lambda ob: torch.cat([ob.data.root_pos_w[0], ob.data.root_quat_w[0]]).cpu().numpy()
        for _ in range(int(a.settle_s/dt)): step(Q[0], Gcmd[0], False)          # objects settle on the table, arm at the start
        start = {k: pose(ob) for k, ob in objs.items()}
        print(f'[replay] after settle: lift {robot.data.joint_pos[0, lid].item():.4f} (target {RB["lift"]:.4f}), arm err '
              f'{(robot.data.joint_pos[0, arm_ids] - Q[0]).abs().max().item():.4f} rad', flush=True)
        W, O, P, L7, BODY = [], [], [], [], []
        n = len(Q)
        for i in range(n):
            for s in range(a.substeps):
                f = s/a.substeps; j = min(i + 1, n - 1)
                step(Q[i]*(1 - f) + Q[j]*f, Gcmd[i]*(1 - f) + Gcmd[j]*f, s == a.substeps - 1)
            wrist.update(dt*a.substeps); over.update(dt*a.substeps)
            W.append(wrist.data.output['rgb'][0, ..., :3].cpu().numpy()); O.append(over.data.output['rgb'][0, ..., :3].cpu().numpy())
            P.append({k: pose(ob) for k, ob in objs.items()})
            L7.append(torch.cat([robot.data.body_pos_w[0, b7], robot.data.body_quat_w[0, b7]]).cpu().numpy())
            BODY.append(robot.data.body_pos_w[0].cpu().numpy())
        for _ in range(int(a.hold_s/dt)): step(Q[-1], Gcmd[-1], False)
        final = {k: pose(ob) for k, ob in objs.items()}

        # ---- scores ----
        L7p = np.array(CK['link7_world']); L7s = np.array(L7)
        trk_mm = 1000*np.linalg.norm(L7s[:, :3] - L7p[:, :3, 3], axis=1)
        trk_deg = np.degrees([(Rotation.from_quat(np.roll(q[3:], -1)).inv()*Rotation.from_matrix(T[:3, :3])).magnitude() for q, T in zip(L7s, L7p)])
        gname = next(o['name'] for o in TW['objects'] if o['role'] == 'grasped'); ig, ir = CK['grasp_frame'], CK['release_frame']
        gz = np.array([p[gname][2] for p in P]); z0 = start[gname][2]
        place = np.array(CK['object_place_planned']); fin = final[gname][:3]
        moved = {k: 100*float(np.linalg.norm(final[k][:2] - start[k][:2])) for k in objs if k != gname}
        # who moved it: the closest any robot body origin came to the object (its centre, minus its half size) in the frames
        # before it first moved 5 mm; far (> 10 cm) = not the robot (e.g. overlapping reconstructions pushing apart)
        BODY = np.array(BODY); names = list(objs); half = {o['name']: 0.5*max(o['size'][:2]) for o in TW['objects']}
        by_robot = {}
        for k in moved:
            if moved[k] < 1.0: continue
            tr = np.array([p[k][:3] for p in P]); i1 = int(np.argmax(np.linalg.norm(tr[:, :2] - start[k][:2], axis=1) > 0.005))
            dmin = float((np.linalg.norm(BODY[max(0, i1 - 3):i1 + 1] - tr[i1][None, None], axis=-1).min() - half[k])*100)
            by_robot[k] = dict(first_frame=i1, phase=RB['phase'][i1], robot_body_dist_cm=dmin, robot=bool(dmin < 10.0))
        # a container target (a static object whose footprint holds the planned place point, e.g. a bowl): placed = the object ends
        # inside that footprint - it slides down a curved wall (real cube_bowl/1, /2: 3.3-4.8 cm from the release point, all in the bowl)
        hold = [k for k in objs if k != gname and np.linalg.norm(place[:2] - start[k][:2]) < half[k]]
        in_container = any(np.linalg.norm(fin[:2] - start[k][:2]) < half[k] - 0.01 for k in hold)
        res = dict(ep=TW['ep'], clamped_rad=clamped, lift_sim=float(robot.data.joint_pos[0, lid]), lift_plan=RB['lift'], squeeze=a.squeeze, obj_mass=a.obj_mass, friction=a.friction, dt=dt,
                   link7_track_mm=dict(max=float(trk_mm.max()), mean=float(trk_mm.mean())), link7_track_deg_max=float(trk_deg.max()),
                   grasped_pushed_before_grasp_cm=100*float(np.linalg.norm(P[ig][gname][:2] - start[gname][:2])),
                   lift_cm=100*float(gz[ig:ir].max() - z0), lifted=bool(gz[ig:ir].max() - z0 > 0.03),
                   place_err_xy_cm=100*float(np.linalg.norm(fin[:2] - place[:2])), final_height_above_start_cm=100*float(fin[2] - z0),
                   placed=bool(np.linalg.norm(fin[:2] - place[:2]) < 0.03 or in_container), placed_in_container=bool(in_container),
                   static_moved_cm=moved, static_moved_why=by_robot,
                   static_disturbed=bool(any(v['robot'] for v in by_robot.values())))
        json.dump(res, open(f'{OUT}/replay.json', 'w'), indent=1)
        np.savez_compressed(f'{OUT}/frames.npz', wrist=np.stack(W).astype(np.uint8), overview=np.stack(O).astype(np.uint8), link7=L7s,
                            grasped=np.array([p[gname] for p in P]), objects=np.array([[p[k] for k in objs] for p in P]),
                            object_names=np.array(list(objs)))
        print('[replay]', json.dumps(res), flush=True)
        uid = int(os.environ.get('HOST_UID', '-1'))
        if uid >= 0:                                           # files written as root in the container -> the host user
            for dp, dn, fn in os.walk(OUT):
                for x in dn + fn: os.chown(os.path.join(dp, x), uid, uid)
            os.chown(OUT, uid, uid)


    except BaseException:
        import traceback; traceback.print_exc(); os._exit(1)        # app.close() hangs in this headless image
    os._exit(0)


if __name__ == '__main__':
    main()
