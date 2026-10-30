
def main():
    #!/usr/bin/env python3
    """
    Sim demos for the policy test (step 2): the scripted expert's episodes (expert_plan.py) executed with physics in one
    scene's Isaac twin (scene as isaac_replay.py: table, walls, reconstructed objects; robotis_lab FFW_SG2, root fixed at
    the scene's base pose, lift fixed, idle arm at zero, head pitched down at the table as in robotis_lab's recordings).
    Per episode: objects reset (the grasped one at its random start pose), robot at the plan's first joints, 1 s settle,
    then the plan at 16 fps (12 physics steps per frame). Recorded per frame: the human head camera (fixed at the EgoDex head pose of
    the input frame, EgoDex intrinsics, 480x270: the generator's input and the head-view baseline; the robot's own head
    cannot see the workspace in turned stances - head yaw +-20 deg), the robot head camera RGB (robotis_lab ZED cam_head,
    336x188), wrist D405 RGB (424x240, the planned mount on link7), arm joints + gripper joint1 (state), the commanded
    targets (action), the grasped object's pose. Success: lifted >= 3 cm and released within 3 cm of the planned place.
    Writes <out>/ep_<i>.npz and <out>/collect.json. Run in the robotis_lab image (run_isaac.sh-style container).
    """
    import argparse, json, math, os
    from isaaclab.app import AppLauncher

    ap = argparse.ArgumentParser()
    ap.add_argument('--twin', required=True); ap.add_argument('--plans', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--first', type=int, default=0); ap.add_argument('--count', type=int, default=1000)
    ap.add_argument('--squeeze', type=float, default=0.15); ap.add_argument('--obj_mass', type=float, default=0.05)
    ap.add_argument('--static_mass', type=float, default=0.3); ap.add_argument('--friction', type=float, default=1.0)
    ap.add_argument('--substeps', type=int, default=12)
    ap.add_argument('--policy', default=None, help='host:port of policy_server.py: closed-loop evaluation instead of the plan')
    ap.add_argument('--max_frames', type=float, default=1.4, help='policy mode: episode length = this x the plan length')
    ap.add_argument('--clean_hr', default=None, help='only render the human head view at 1920x1080 without the robot and the grasped object (the generator input) to this PNG, then exit')
    ap.add_argument('--overview', action='store_true', help='also record a third-person camera (robot + scene) - for videos')
    ap.add_argument('--head_focal', type=float, default=2.4, help='focal length (USD units) for the --clean_hr camera')
    ap.add_argument('--head_pitch', type=float, default=0.549, help='head_joint1 (rad; robotis_lab records at 0.549)')
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
        from robotis_lab.assets.robots import ROBOTIS_LAB_ASSETS_DATA_DIR as RD

        TW = json.load(open(f'{a.twin}/twin.json')); PL = json.load(open(a.plans)); side = PL['side'][0]
        os.makedirs(a.out, exist_ok=True); os.makedirs(f'{a.out}/usd', exist_ok=True)
        wxyz = lambda R_: tuple(np.roll(Rotation.from_matrix(R_).as_quat(), 1).tolist())
        lin = lambda rgb: tuple(((np.array(rgb)/255.)**2.2).tolist())

        def object_usd(nm, mass, approx):
            z = np.load(f'{a.twin}/meshes/{nm}.npz'); V, F, C = z['V'], z['F'], z['rgb']; c = (V.min(0) + V.max(0))/2
            path = f'{a.out}/usd/{nm}.usda'; st = Usd.Stage.CreateNew(path)
            UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(st, 1.0)
            root = UsdGeom.Xform.Define(st, '/obj'); st.SetDefaultPrim(root.GetPrim()); mesh = UsdGeom.Mesh.Define(st, '/obj/geom')
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy((V - c).astype(np.float32)))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(F), 3, np.int32)))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(F.reshape(-1).astype(np.int32))); mesh.CreateSubdivisionSchemeAttr('none')
            mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray.FromNumpy(((C/255.)**2.2).astype(np.float32)))
            mat = UsdShade.Material.Define(st, '/obj/look'); pbr = UsdShade.Shader.Define(st, '/obj/look/pbr'); pbr.CreateIdAttr('UsdPreviewSurface')
            rd = UsdShade.Shader.Define(st, '/obj/look/col'); rd.CreateIdAttr('UsdPrimvarReader_float3')
            rd.CreateInput('varname', Sdf.ValueTypeNames.String).Set('displayColor'); rd.CreateOutput('result', Sdf.ValueTypeNames.Float3)
            pbr.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(rd.ConnectableAPI(), 'result')
            pbr.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.7); mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), 'surface')
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()); UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(approx)
            if approx == 'convexDecomposition':
                cd = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(mesh.GetPrim()); cd.CreateMaxConvexHullsAttr(32)
                cd.CreateVoxelResolutionAttr(500000); cd.CreateShrinkWrapAttr(True)
            pc = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim()); pc.CreateContactOffsetAttr(0.003); pc.CreateRestOffsetAttr(0.0)
            pm = UsdShade.Material.Define(st, '/obj/physmat'); pma = UsdPhysics.MaterialAPI.Apply(pm.GetPrim())
            pma.CreateStaticFrictionAttr(a.friction); pma.CreateDynamicFrictionAttr(a.friction); pma.CreateRestitutionAttr(0.0)
            UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(pm, UsdShade.Tokens.weakerThanDescendants, 'physics')
            UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()); UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(mass)
            PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim()).CreateEnableCCDAttr(True)
            st.Save(); return path, c

        dt = 1.0/(PL['fps']*a.substeps)
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device='cuda:0', render_interval=a.substeps))
        sim_utils.GroundPlaneCfg().func('/World/ground', sim_utils.GroundPlaneCfg())
        dl = sim_utils.DomeLightCfg(intensity=700.0, color=(1.0, 0.95, 0.88)); dl.func('/World/dome', dl)
        lc = sim_utils.DistantLightCfg(intensity=900.0, angle=20.0); lc.func('/World/sun', lc, orientation=wxyz(Rotation.from_euler('y', 25, degrees=True).as_matrix()))
        tb = TW['table']; th = TW['table_h']; L = tb['far_x'] - tb['edge_x']
        tcfg = sim_utils.CuboidCfg(size=(L, tb['width'], tb['thickness']), collision_props=sim_utils.CollisionPropertiesCfg(),
                                   visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=lin(tb['rgb']), roughness=0.9),
                                   physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=a.friction, dynamic_friction=a.friction))
        tcfg.func('/World/table', tcfg, translation=(tb['edge_x'] + L/2, tb['y'], th - tb['thickness']/2))
        for k, w in enumerate(TW['walls']):
            n = np.array(w['n'])/np.linalg.norm(w['n']); zax = n; xax = np.cross([0, 0, 1.], zax)
            xax = xax/np.linalg.norm(xax) if np.linalg.norm(xax) > 1e-3 else np.array([1., 0, 0]); yax = np.cross(zax, xax)
            wcfg = sim_utils.CuboidCfg(size=(3.0, 3.0, 0.01), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=lin(w['rgb']), roughness=1.0))
            wcfg.func(f'/World/wall{k}', wcfg, translation=tuple(w['point']), orientation=wxyz(np.stack([xax, yax, zax], 1)))
        objs, home = {}, {}
        for o in TW['objects']:
            g = o['role'] == 'grasped'
            p, c = object_usd(o['name'], a.obj_mass if g else a.static_mass, 'convexHull' if g else 'convexDecomposition')
            objs[o['name']] = RigidObject(RigidObjectCfg(prim_path=f"/World/{o['name']}", spawn=sim_utils.UsdFileCfg(usd_path=p),
                                                         init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(c.tolist()))))
            home[o['name']] = c
        gname = next(o['name'] for o in TW['objects'] if o['role'] == 'grasped')
        # joint limits from the robot USD (the plan clamped into them)
        _st = Usd.Stage.Open(f'{RD}/robots/FFW/FFW_SG2.usd'); LIM = {}
        for _p in _st.Traverse():
            for T_, cv in ((UsdPhysics.RevoluteJoint, math.radians), (UsdPhysics.PrismaticJoint, float)):
                if _p.IsA(T_):
                    j = T_(_p); lo_, hi_ = j.GetLowerLimitAttr().Get(), j.GetUpperLimitAttr().Get()
                    if lo_ is not None and hi_ is not None: LIM[_p.GetName()] = (cv(lo_), cv(hi_))
        cl = lambda j, v: min(max(v, LIM.get(j, (-9, 9))[0] + 1e-4), LIM.get(j, (-9, 9))[1] - 1e-4)
        Tb = np.array(PL['base_pose']); rcfg = FFW_SG2_CFG.replace(prim_path='/World/Robot'); rcfg.spawn.articulation_props.fix_root_link = True
        jp = dict(rcfg.init_state.joint_pos); jp.update({'lift_joint': cl('lift_joint', PL['lift']), 'head_joint1': a.head_pitch, 'head_joint2': 0.0,
                                                         f'gripper_{side}_joint1': PL['gripper_open']})
        q0 = PL['plans'][0]['arm_q'][0]; jp.update({j: cl(j, float(q)) for j, q in zip(PL['arm_joints'], q0)})
        rcfg.init_state = rcfg.init_state.replace(pos=tuple(Tb[:3, 3].tolist()), rot=wxyz(Tb[:3, :3]), joint_pos=jp)
        robot = Articulation(rcfg); stage = sim_utils.get_current_stage()
        pth = {}                                    # first (outermost) prim per name: link prims, not their nested meshes
        for p in stage.Traverse():
            if str(p.GetPath()).startswith('/World/Robot'): pth.setdefault(p.GetName(), str(p.GetPath()))
        M = np.array(TW['link7_d405']); K = np.array(TW['check']['d405_K_424x240'])
        wrist = Camera(CameraCfg(prim_path=f"{pth[f'arm_{side}_link7']}/d405", width=424, height=240, data_types=['rgb'], update_period=0,
                                 spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(K.reshape(-1).tolist(), 424, 240, clipping_range=(0.005, 20.0)),
                                 offset=CameraCfg.OffsetCfg(pos=tuple(M[:3, 3].tolist()), rot=wxyz(M[:3, :3]), convention='ros')))
        head = Camera(CameraCfg(prim_path=f"{pth['zed']}/cam_head", width=336, height=188, data_types=['rgb'], update_period=0,
                                spawn=sim_utils.PinholeCameraCfg(focal_length=10.4, focus_distance=200.0, horizontal_aperture=20.955, clipping_range=(0.01, 100.0)),
                                offset=CameraCfg.OffsetCfg(pos=(0.0, 0.03, 0.0), rot=(0.5, 0.5, -0.5, -0.5), convention='opengl')))   # robotis 'isaac' = opengl in Isaac Lab 2.3
        if a.overview:
            ovc = Camera(CameraCfg(prim_path='/World/overview', width=640, height=400, data_types=['rgb'], update_period=0,
                                   spawn=sim_utils.PinholeCameraCfg(focal_length=14.0, clipping_range=(0.05, 20.0))))
        Kh = np.array(TW['human_head_K_1920x1080'])*np.array([[0.25], [0.25], [1]])        # 480x270
        hhead = Camera(CameraCfg(prim_path='/World/human_head', width=480, height=270, data_types=['rgb'], update_period=0,
                                 spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(Kh.reshape(-1).tolist(), 480, 270, clipping_range=(0.01, 20.0))))
        if a.clean_hr:
            Kf = np.array(TW['human_head_K_1920x1080'])
            # explicit focal length: from_intrinsic_matrix's default writes focal = fx/width (0.38 for this camera) with aperture 1,
            # and the renderer drew it ~1.3x zoomed (the reported K stayed 736.6); a physical focal length renders the true K
            wfix = Camera(CameraCfg(prim_path=f"{pth[f'arm_{side}_link7']}/d405_fix", width=424, height=240, data_types=['rgb'], update_period=0,
                                    spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(K.reshape(-1).tolist(), 424, 240, clipping_range=(0.005, 20.0), focal_length=2.4),
                                    offset=CameraCfg.OffsetCfg(pos=tuple(M[:3, 3].tolist()), rot=wxyz(M[:3, :3]), convention='ros')))
            hr = Camera(CameraCfg(prim_path='/World/human_head_hr', width=1920, height=1080, data_types=['rgb'], update_period=0,
                                  spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(Kf.reshape(-1).tolist(), 1920, 1080, clipping_range=(0.01, 20.0), focal_length=a.head_focal)))
        sim.reset()
        Th = np.array(TW['human_head_world'])                                               # the EgoDex head pose (OpenCV axes)
        hhead.set_world_poses(torch.tensor([Th[:3, 3]], device='cuda:0', dtype=torch.float32),
                              torch.tensor([wxyz(Th[:3, :3])], device='cuda:0', dtype=torch.float32), convention='ros')
        if a.clean_hr:                         # the scene as the generator sees it: no robot, no grasped object (EgoDex: hands and object removed)
            hr.set_world_poses(torch.tensor([Th[:3, 3]], device='cuda:0', dtype=torch.float32),
                               torch.tensor([wxyz(Th[:3, :3])], device='cuda:0', dtype=torch.float32), convention='ros')
            for _ in range(int(1.0/dt)): sim.step(render=False)
            for pp in ('/World/Robot', f"/World/{next(o['name'] for o in TW['objects'] if o['role'] == 'grasped')}"):
                UsdGeom.Imageable(sim_utils.get_current_stage().GetPrimAtPath(pp)).MakeInvisible()
            for _ in range(4): sim.render()
            hr.update(dt); im = hr.data.output['rgb'][0, ..., :3].cpu().numpy().astype(np.uint8)
            for nm_, c_ in (('head_hr', hr), ('human_head', hhead), ('wrist', wrist)):
                c_.update(dt)
                print(f'[camera] {nm_}: K', np.round(c_.data.intrinsic_matrices[0].cpu().numpy(), 1).tolist(), 'pos', np.round(c_.data.pos_w[0].cpu().numpy(), 4).tolist(),
                      'quat_ros', np.round(c_.data.quat_w_ros[0].cpu().numpy(), 4).tolist(), flush=True)
            for k_, ob_ in objs.items():
                ob_.update(dt); _pv = np.array(UsdGeom.Mesh(sim_utils.get_current_stage().GetPrimAtPath(f'/World/{k_}/geom')).GetPointsAttr().Get())
                print(f'[camera] object {k_}: root {np.round(ob_.data.root_pos_w[0].cpu().numpy(), 4).tolist()} quat {np.round(ob_.data.root_quat_w[0].cpu().numpy(), 3).tolist()} home {np.round(home[k_], 4).tolist()} | usd mesh extent {np.round(np.ptp(_pv, 0), 4).tolist()}', flush=True)
            _prim = sim_utils.get_current_stage().GetPrimAtPath('/World/human_head_hr')
            print('[camera] USD head_hr focal', _prim.GetAttribute('focalLength').Get(), 'hap', _prim.GetAttribute('horizontalAperture').Get(), 'vap', _prim.GetAttribute('verticalAperture').Get(), flush=True)
            wrist.update(dt); wfix.update(dt); w0 = wrist.data.output['rgb'][0, ..., :3].cpu().numpy().astype(float); w1 = wfix.data.output['rgb'][0, ..., :3].cpu().numpy().astype(float)
            print(f'[camera] wrist default vs explicit focal: mean |diff| {np.abs(w0 - w1).mean():.2f} (0-255)', flush=True)
            from PIL import Image; Image.fromarray(np.concatenate([w0, w1], 1).astype(np.uint8)).save(a.clean_hr.replace('.png', '_wristcmp.png'))
            Image.fromarray(im).save(a.clean_hr); print('[collect] clean head frame ->', a.clean_hr, im.shape, flush=True)
            uid = int(os.environ.get('HOST_UID', '-1'))
            if uid >= 0: os.chown(a.clean_hr, uid, uid)
            os._exit(0)
        if a.overview:                     # third person: in front of and beside the robot, looking back over the table at it
            _f = Tb[:3, 0]; _r = -Tb[:3, 1]; _o = np.array(next(o for o in TW['objects'] if o['role'] == 'grasped')['centre'])
            _tgt = 0.5*(Tb[:3, 3] + _o); _tgt[2] = TW['table_h'] + 0.2
            _eye = Tb[:3, 3] + 0.95*_f + 0.75*_r; _eye[2] = TW['table_h'] + 0.8
            for _w in TW.get('walls', []):                    # keep the camera >= 30 cm inside every wall (bpp/1: 5 cm in front of the far wall -> a wall-filled view)
                _n = np.array(_w['n'], float); _n /= np.linalg.norm(_n); _sd = float(_n @ (_eye - np.array(_w['point'], float)))
                if _sd < 0.3: _eye = _eye + (0.3 - _sd)*_n
            ovc.set_world_poses_from_view(torch.tensor(np.array([_eye]), device='cuda:0', dtype=torch.float32), torch.tensor(np.array([_tgt]), device='cuda:0', dtype=torch.float32))
        arm_ids, _ = robot.find_joints(PL['arm_joints'], preserve_order=True); g_ids, _ = robot.find_joints([f'gripper_{side}_joint1'])
        for _ in range(3): sim.step(render=True)
        head.update(dt); wrist.update(dt)
        _q = head.data.quat_w_world[0].cpu().numpy(); _R = Rotation.from_quat(np.roll(_q, -1)).as_matrix()
        print('[collect] head cam pos', np.round(head.data.pos_w[0].cpu().numpy(), 3), 'forward (world conv +x)', np.round(_R[:, 0], 2),
              '| base', np.round(Tb[:3, 3], 3), 'heading', np.round(Tb[:3, 0], 2), flush=True)
        print('[collect] scene up:', len(robot.joint_names), 'joints; head cam', pth.get('zed'), '; wrist on', pth.get(f'arm_{side}_link7'), flush=True)

        def step(q, g, render):
            robot.set_joint_position_target(q[None], joint_ids=arm_ids); robot.set_joint_position_target(g.reshape(1, 1), joint_ids=g_ids)
            robot.write_data_to_sim(); sim.step(render=render); robot.update(dt)
            for ob in objs.values(): ob.update(dt)

        conn = None
        if a.policy:                          # closed loop: observations to policy_server.py, actions back
            from multiprocessing.connection import Client
            _h, _p = a.policy.split(':'); conn = Client((_h, int(_p)), authkey=b'egoh2w')
        def conn_call(m): conn.send(m); return conn.recv()
        for _ in range(3): sim.step(render=True)
        wrist.update(dt); hhead.update(dt)
        summary = []
        for P in PL['plans'][a.first:a.first + a.count]:
            # reset: objects home, the grasped one at its random start; robot at the plan's first joints
            for k, ob in objs.items():
                pos = np.array(P['object_start']) if k == gname else home[k]
                rot = P['object_quat_wxyz'] if k == gname else [1.0, 0, 0, 0]
                ob.write_root_pose_to_sim(torch.tensor([list(pos) + list(rot)], device='cuda:0', dtype=torch.float32))
                ob.write_root_velocity_to_sim(torch.zeros(1, 6, device='cuda:0'))
            # stance perturbation (plans_eval_pert.json): this episode's robot base pose (fixed root, teleported at the reset)
            Tb_e = np.array(P.get('base_pose', PL['base_pose']))
            robot.write_root_pose_to_sim(torch.tensor([list(Tb_e[:3, 3]) + list(wxyz(Tb_e[:3, :3]))], device='cuda:0', dtype=torch.float32))
            robot.write_root_velocity_to_sim(torch.zeros(1, 6, device='cuda:0'))
            Q = torch.tensor([[cl(j, v) for j, v in zip(PL['arm_joints'], q)] for q in P['arm_q']], device='cuda:0', dtype=torch.float32)
            gs = np.array(P['gripper_q']); go, gc = PL['gripper_open'], PL['gripper_close']
            G = torch.tensor(np.clip(gs + a.squeeze*np.clip((gs - go)/max(gc - go, 1e-6), 0, 1), 0, PL['gripper_max']), device='cuda:0', dtype=torch.float32)
            jq = robot.data.default_joint_pos.clone(); jq[0, arm_ids] = Q[0]; jq[0, g_ids] = G[0]
            robot.write_joint_state_to_sim(jq, torch.zeros_like(jq)); robot.set_joint_position_target(jq)
            for _ in range(int(1.0/dt)): step(Q[0], G[0], False)
            z0 = float(objs[gname].data.root_pos_w[0, 2])
            # the generator's input: the human head view of the scene WITHOUT the robot (EgoDex: the input frame, hands removed)
            rimg = UsdGeom.Imageable(stage.GetPrimAtPath('/World/Robot')); rimg.MakeInvisible()
            for _ in range(2): sim.render()
            hhead.update(dt); clean = hhead.data.output['rgb'][0, ..., :3].cpu().numpy().copy(); rimg.MakeVisible()
            for _ in range(2): sim.render()
            # fresh camera frames before the first observation: the settle steps do not render, so without this the policy's
            # first view was the PREVIOUS episode's last frame (gripper over the box) and, with temporal ensembling, that
            # first chunk dragged the next ~30 frames (closed-loop 0/6 while teacher-forced the policy was accurate)
            wrist.update(dt); head.update(dt); hhead.update(dt)
            HEAD, WR, ST, AC, OB, HH, OV = [], [], [], [], [], [], []
            if a.policy: conn.send({'reset': True}); conn.recv()
            q_cur, g_cur = Q[0].clone(), G[0].clone()
            n_fr = int(a.max_frames*len(Q)) if a.policy else len(Q)
            for i in range(n_fr):
                if a.policy:                  # the policy's next targets from the current observation
                    _w = wrist.data.output['rgb'][0, ..., :3].contiguous().cpu().numpy(); _h = hhead.data.output['rgb'][0, ..., :3].contiguous().cpu().numpy()
                    ob_ = dict(state=[float(v) for v in torch.cat([robot.data.joint_pos[0, arm_ids], robot.data.joint_pos[0, g_ids]]).cpu()],
                               wrist=(_w.astype(np.uint8).tobytes(), list(_w.shape)), head=(_h.astype(np.uint8).tobytes(), list(_h.shape)))   # numpy-free
                    act = torch.tensor(conn_call(ob_)['action'], device='cuda:0', dtype=torch.float32); q_nx, g_nx = act[:7], act[7]
                else:
                    j = min(i + 1, len(Q) - 1); q_nx, g_nx = Q[j], G[j]
                for s in range(a.substeps):
                    f = (s + 1)/a.substeps; step(q_cur*(1 - f) + q_nx*f, g_cur*(1 - f) + g_nx*f, s == a.substeps - 1)
                q_cur, g_cur = q_nx, g_nx
                wrist.update(dt*a.substeps); head.update(dt*a.substeps); hhead.update(dt*a.substeps)
                WR.append(wrist.data.output['rgb'][0, ..., :3].cpu().numpy()); HEAD.append(head.data.output['rgb'][0, ..., :3].cpu().numpy())
                HH.append(hhead.data.output['rgb'][0, ..., :3].cpu().numpy())
                if a.overview: ovc.update(dt*a.substeps); OV.append(ovc.data.output['rgb'][0, ..., :3].cpu().numpy())
                ST.append(torch.cat([robot.data.joint_pos[0, arm_ids], robot.data.joint_pos[0, g_ids]]).cpu().numpy())
                AC.append(torch.cat([q_nx, g_nx.reshape(1)]).cpu().numpy())
                OB.append(torch.cat([objs[gname].data.root_pos_w[0], objs[gname].data.root_quat_w[0]]).cpu().numpy())
            for _ in range(int(1.0/dt)): step(q_cur, g_cur, False)
            OB = np.array(OB); fin = objs[gname].data.root_pos_w[0].cpu().numpy(); fin_q = objs[gname].data.root_quat_w[0].cpu().numpy()
            lifted = float(OB[:, 2].max() - z0); err = float(np.linalg.norm(fin[:2] - np.array(P['place_planned'][:2])))
        
            # held above the target (the post-settle height at most 1 cm above the planned release height); the old test (lift + XY)
            # passed a policy still holding the object over the target
            # the last MEASURED gripper joint, as rescore_evals.py uses (reported results are always scored by rescore_evals.py on the
            # saved rollout: last recorded pose vs the expert rest height; this flag only filters training demos)
            g_last = float(np.asarray(ST[-1])[-1]) if ST else float(PL['gripper_open']); released = g_last < 0.5*(PL['gripper_open'] + PL['gripper_close'])
            resting = float(fin[2]) < float(P['place_planned'][2]) + 0.01; ok_loose = lifted > 0.03 and err < 0.03
            ok = ok_loose and released and resting
            final = dict(object_final=np.r_[fin, fin_q].astype(np.float32), gripper_last=np.float32(g_last), success_loose=ok_loose, released=released, resting=resting,
                         place_planned=np.array(P['place_planned'], np.float32))
            if a.policy:                      # evaluation: keep the camera views as small videos-to-be, not training data
                np.savez_compressed(f"{a.out}/eval_{P['i']:04d}.npz", **({'overview': np.stack(OV).astype(np.uint8)} if OV else {}), wrist=np.stack(WR).astype(np.uint8)[::2], human_head=np.stack(HH).astype(np.uint8)[::2],
                                    state=np.array(ST, np.float32), action=np.array(AC, np.float32), object=OB.astype(np.float32), success=ok, **final)
            else:
              np.savez_compressed(f"{a.out}/ep_{P['i']:04d}.npz", base_pose=Tb_e.astype(np.float32), human_head_clean=clean.astype(np.uint8), human_head=np.stack(HH).astype(np.uint8), head=np.stack(HEAD).astype(np.uint8), wrist=np.stack(WR).astype(np.uint8),
                                state=np.array(ST, np.float32), action=np.array(AC, np.float32), object=OB.astype(np.float32),
                                phase=np.array(P['phase']), success=ok, **final)
            summary.append(dict(i=P['i'], success=bool(ok), success_loose=bool(ok_loose), released=bool(released), resting=bool(resting), lift_cm=100*lifted, place_err_cm=100*err, final_z=float(fin[2]), frames=len(Q)))
            _rp = robot.data.root_pos_w[0].cpu().numpy()
            if 'base_offset' in P: print(f"[collect] ep {P['i']}: base at {np.round(_rp, 3).tolist()} (offset {np.round(P['base_offset'], 3).tolist()})", flush=True)
            print(f"[collect] ep {P['i']}: success={ok} lift {100*lifted:.1f} cm place err {100*err:.1f} cm", flush=True)
        json.dump(summary, open(f'{a.out}/collect_{a.first:04d}.json', 'w'), indent=1)
        uid = int(os.environ.get('HOST_UID', '-1'))
        if uid >= 0:
            for dp, dn, fn in os.walk(a.out):
                for x in dn + fn: os.chown(os.path.join(dp, x), uid, uid)
    except BaseException:
        import traceback; traceback.print_exc(); os._exit(1)
    os._exit(0)


if __name__ == '__main__':
    main()
