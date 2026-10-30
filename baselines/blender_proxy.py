from head2wrist.config import H2W_OUTPUT_ROOT
#!/usr/bin/env python3
"""
The complete scene proxy as a Blender scene (scripted: no manual modelling), rendered with Cycles along the wrist path.
Same geometry as render_proxy.py, rendered with real light transport instead of flat colours:
  background  the empty-scene depth mesh (what the head saw) with the empty head frame projected on it (UVs = head-camera
              projection); the table plane (3 mm under the fitted plane, so the measured mesh wins where both exist)
              with the outpainted table map; the wall planes in their median colour
  shading     v2 (default): every surface is emission of its texture (the light is baked in the real pixels) times
              1 - ao*(1 - local AO within 4 cm): contact shadows near objects, exact real colours elsewhere. v1 mixed in
              diffuse under a white sky, which darkened everything (bowl dE 14.9 vs 6.4, table 13.1 vs 6.3)
  objects     v2: the head frame's pixels on the surface the head saw (object_visibility.py 'seen', blended at the
              border), SAM 3D colours on the unseen side (its texture put an orange blotch inside the bowl)
  table map   v2: the real head frame wherever the head saw that table point, the outpainted map beyond (v1 used the
              map everywhere: a colour seam against the seen mesh)
  camera      keyframed wrist camera (OpenCV -> Blender axes), lens per frame from the task intrinsics; rendered at
              1280x720 square pixels = the per-axis-resized 1280x704 task camera
Writes <task_inputs>/proxy/render_blender/: scene.blend (open in Blender: the scene + the animated camera), rgb.mp4
(1280x704, 16 fps) and links to render_v3's depth / seg / guided_mask / source (same geometry).
Run in third_party/blender/.venv:  python head2wrist/scene/blender_proxy.py --ep basic_pick_place/74 [--frames 0,120,200]
"""
import argparse, json, os, subprocess, sys
import numpy as np, cv2
from scipy.spatial import cKDTree



def mesh_obj(nm, V, F, uv=None, col=None, pass_index=0, fattr=None):
    me = bpy.data.meshes.new(nm); me.vertices.add(len(V)); me.vertices.foreach_set('co', V.astype(np.float32).ravel())
    me.loops.add(3*len(F)); me.loops.foreach_set('vertex_index', F.astype(np.int32).ravel())
    me.polygons.add(len(F)); me.polygons.foreach_set('loop_start', np.arange(0, 3*len(F), 3, dtype=np.int32))
    me.update(); me.validate()
    if uv is not None:
        lay = me.uv_layers.new(name='UV'); lay.data.foreach_set('uv', uv[F.ravel()].astype(np.float32).ravel())
    if col is not None:
        at = me.color_attributes.new(name='Col', type='FLOAT_COLOR', domain='POINT')
        at.data.foreach_set('color', np.concatenate([srgb2lin(col/255.), np.ones((len(col), 1))], 1).astype(np.float32).ravel())
    for an, av in (fattr or {}).items():
        fa = me.attributes.new(name=an, type='FLOAT', domain='POINT'); fa.data.foreach_set('value', av.astype(np.float32))
    ob = bpy.data.objects.new(nm, me); sc.collection.objects.link(ob); ob.pass_index = pass_index
    return ob


def _sock(node, ident, out=False):
    for so in (node.outputs if out else node.inputs):
        if so.identifier == ident: return so
    raise KeyError(ident)


def material_v2(nm, image=None, colour=None, vertex=False, seen_mix=False, use_ao=True, ao_map=None, ao_uv=None, image_uv=None):
    """emission (light baked in the real pixels) x (1 - ao*(1 - local AO)): contact shadows only near objects"""
    m = bpy.data.materials.new(nm); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial'); em = nt.nodes.new('ShaderNodeEmission')
    if image is not None:
        tex = nt.nodes.new('ShaderNodeTexImage'); tex.image = bpy.data.images.get(os.path.basename(image)) or bpy.data.images.load(image)
        tex.image.colorspace_settings.name = 'sRGB'; tex.interpolation = 'Linear'; tex.extension = 'EXTEND'; src = tex.outputs['Color']
        if image_uv:
            uvn = nt.nodes.new('ShaderNodeUVMap'); uvn.uv_map = image_uv; nt.links.new(uvn.outputs['UV'], tex.inputs['Vector'])
    if vertex:
        vc = nt.nodes.new('ShaderNodeVertexColor'); vc.layer_name = 'Col'; vsrc = vc.outputs['Color']
        if seen_mix:
            at = nt.nodes.new('ShaderNodeAttribute'); at.attribute_name = 'seen'
            mx = nt.nodes.new('ShaderNodeMix'); mx.data_type = 'RGBA'; mx.blend_type = 'MIX'
            nt.links.new(at.outputs['Fac'], _sock(mx, 'Factor_Float')); nt.links.new(vsrc, _sock(mx, 'A_Color')); nt.links.new(src, _sock(mx, 'B_Color'))
            src = _sock(mx, 'Result_Color', out=True)
        else:
            src = vsrc
    if image is None and not vertex:
        rgb = nt.nodes.new('ShaderNodeRGB'); rgb.outputs[0].default_value = (*srgb2lin(np.array(colour)/255.), 1); src = rgb.outputs[0]
    if not use_ao:                                     # the table plane lies 3 mm under the seen mesh: AO there would darken it in patches
        nt.links.new(src, em.inputs['Color']); nt.links.new(em.outputs[0], out.inputs['Surface']); return m
    fac = nt.nodes.new('ShaderNodeMath'); fac.operation = 'MULTIPLY_ADD'; fac.inputs[1].default_value = a.ao; fac.inputs[2].default_value = 1 - a.ao
    if ao_map:                                         # contact shadows of the objects only (contact_ao.py), top-down
        at_ = nt.nodes.new('ShaderNodeTexImage'); at_.image = bpy.data.images.get(os.path.basename(ao_map)) or bpy.data.images.load(ao_map)
        at_.image.colorspace_settings.name = 'Non-Color'; at_.extension = 'EXTEND'
        uva = nt.nodes.new('ShaderNodeUVMap'); uva.uv_map = ao_uv; nt.links.new(uva.outputs['UV'], at_.inputs['Vector'])
        nt.links.new(at_.outputs['Color'], fac.inputs[0])
    else:                                              # objects: their own occlusion only (bowl inside), not the table's
        ao = nt.nodes.new('ShaderNodeAmbientOcclusion'); ao.samples = 16; ao.inputs['Distance'].default_value = 0.04; ao.only_local = True
        nt.links.new(ao.outputs['AO'], fac.inputs[0])
    mul = nt.nodes.new('ShaderNodeMix'); mul.data_type = 'RGBA'; mul.blend_type = 'MULTIPLY'; _sock(mul, 'Factor_Float').default_value = 1.0
    nt.links.new(src, _sock(mul, 'A_Color')); nt.links.new(fac.outputs[0], _sock(mul, 'B_Color'))
    nt.links.new(_sock(mul, 'Result_Color', out=True), em.inputs['Color']); nt.links.new(em.outputs[0], out.inputs['Surface'])
    return m


def material(nm, image=None, colour=None, vertex=False, emit=0.5):
    m = bpy.data.materials.new(nm); m.use_nodes = True; nt = m.node_tree; nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial'); dif = nt.nodes.new('ShaderNodeBsdfDiffuse'); em = nt.nodes.new('ShaderNodeEmission')
    mix = nt.nodes.new('ShaderNodeMixShader'); mix.inputs[0].default_value = 1 - emit                          # fac 0 = emission
    if image is not None:
        tex = nt.nodes.new('ShaderNodeTexImage'); tex.image = bpy.data.images.load(image); tex.image.colorspace_settings.name = 'sRGB'
        tex.interpolation = 'Linear'; tex.extension = 'EXTEND'; src = tex.outputs['Color']
    elif vertex:
        vc = nt.nodes.new('ShaderNodeVertexColor'); vc.layer_name = 'Col'; src = vc.outputs['Color']
    else:
        rgb = nt.nodes.new('ShaderNodeRGB'); rgb.outputs[0].default_value = (*srgb2lin(np.array(colour)/255.), 1); src = rgb.outputs[0]
    nt.links.new(src, dif.inputs['Color']); nt.links.new(src, em.inputs['Color'])
    nt.links.new(em.outputs[0], mix.inputs[1]); nt.links.new(dif.outputs[0], mix.inputs[2]); nt.links.new(mix.outputs[0], out.inputs['Surface'])
    return m


def main():
    global bpy
    import bpy
    global AOJ, A_, B_, C2W, De, F0, Ft, H0, Hm, K, Mb, Mcv, OUT, PX, Pd, RH, RW, T, TE, UV, Vt, Vw, W0, Wm, Xg, Xp, _os, _sys, a, ab, ab_, abm, allp, ap, at_, bg, c, cam, cd, cfg, colo, cols, ctr, d, dg, dist, dp_, dst, dv, dz, e1, e2, em_, emp, f, f1, f2, fa_, frames, fx, fy, ga, gb, gp, gx, gy, h, headcam, hi, hl, i, idg, idx, ing, inp_, k, lo, mathutils, mo, mt, mu_, mx_, n, name, nn, nt, nx, ny, o, o_, ob, okm, pa, pref, pts, q, res, s1, s2, sc, seen_p, seen_tab, sg, srgb2lin, sv, t1, t2, ta_, tab, tl, tmap, tmp, tr, tri, u1, u2, ug, up_, uu, uv_map, uvo, vg, vp_, vv, vw, w, wd, wl, wn, wo, world, x, z
    ap = argparse.ArgumentParser()
    ap.add_argument('--ep', required=True)
    ap.add_argument('--samples', type=int, default=64)
    ap.add_argument('--frames', default='all', help="'all' or a comma list (test renders)")
    ap.add_argument('--out_name', default='render_blender_v3')
    ap.add_argument('--passes_from', default='render_v5', help='render_proxy.py folder whose depth / seg / guided mask / source match this geometry')
    ap.add_argument('--ao', type=float, default=0.7, help='contact-shadow strength (local ambient occlusion within 4 cm)')
    a = ap.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:])
    name = a.ep.replace('/', '_')
    T = f'{H2W_OUTPUT_ROOT}/lyra2/task_inputs/{name}'
    PX = f'{T}/proxy'
    OUT = f'{PX}/{a.out_name}'
    os.makedirs(f'{OUT}/frames', exist_ok=True)
    import os as _os, sys as _sys
    pass
    import head2wrist.headcam as headcam
    F0 = headcam.focal(T)
    cfg = json.load(open(f'{PX}/proxy.json'))
    n = np.array(cfg['table']['n'])
    d = float(cfg['table']['d'])
    De = np.load(f'{PX}/empty_depth.npy')
    h, w = De.shape
    fx, fy = F0*w/1920, F0*h/1080
    tr = np.load(f'{T}/trajectory.npz')
    K = tr['intrinsics'].astype(np.float64)
    W0, H0 = float(tr['image_width']), float(tr['image_height'])
    C2W = np.load(f'{T}/task_plan.npz')['c2w']
    RW, RH = 1280, 720
    srgb2lin = lambda c: np.where(c <= 0.04045, c/12.92, ((c + 0.055)/1.055)**2.4)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    pref = bpy.context.preferences.addons['cycles'].preferences
    pref.compute_device_type = 'OPTIX'
    pref.get_devices()
    for dv in pref.devices: dv.use = dv.type == 'OPTIX'
    sc.cycles.device = 'GPU'
    sc.cycles.samples = a.samples
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = 'OPTIX'
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = RW, RH, 100
    sc.view_settings.view_transform = 'Standard'
    sc.view_settings.look = 'None'
    sc.render.image_settings.file_format = 'PNG'
    world = bpy.data.worlds.new('sky')
    sc.world = world
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Color'].default_value = (0, 0, 0, 1)
    world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.0
    vv, uu = np.mgrid[0:h, 0:w]
    Pd = np.stack([(uu - w/2)/fx*De, (vv - h/2)/fy*De, De], -1).reshape(-1, 3)
    UV = np.stack([(uu.ravel() + 0.5)/w, 1 - (vv.ravel() + 0.5)/h], 1)
    idx = np.arange(h*w).reshape(h, w)
    q = [idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]]
    tri = np.concatenate([np.stack([q[0], q[2], q[1]], -1).reshape(-1, 3), np.stack([q[1], q[2], q[3]], -1).reshape(-1, 3)])
    dz = De.reshape(-1)[tri]
    tri = tri[(dz.max(1)/np.maximum(dz.min(1), 1e-3) < 1.03) & (dz.min(1) > 0.05)]
    e1 = np.cross(n, [1, 0, 0.])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    o = -d*n
    TE = np.load(f'{T}/table_extension.npz')
    pts, cols = TE['points'], TE['colors']
    seen_tab = Pd[np.abs(Pd @ n + d) < 0.01]
    allp = np.concatenate([pts, seen_tab])
    ab = np.stack([(allp - o) @ e1, (allp - o) @ e2], 1)
    lo, hi = ab.min(0) - 0.3, ab.max(0) + 0.3
    res = 0.002
    AOJ = json.load(open(f'{PX}/contact_ao.json'))
    lo, hi = np.array(AOJ['lo']), np.array(AOJ['hi'])
    Wm, Hm = (np.ceil((hi - lo)/res)).astype(int)
    gy, gx = np.mgrid[0:Hm, 0:Wm]
    gp = np.stack([lo[0] + (gx.ravel() + 0.5)*res, lo[1] + (gy.ravel() + 0.5)*res], 1)
    pa = np.stack([(pts - o) @ e1, (pts - o) @ e2], 1)
    dist, nn = cKDTree(pa).query(gp, distance_upper_bound=0.006)
    tmap = np.tile(np.array(cfg['table']['colour'], np.uint8), (len(gp), 1))
    okm = np.isfinite(dist)
    tmap[okm] = cols[nn[okm]]
    emp = cv2.cvtColor(cv2.imread(f'{PX}/empty_frame.png'), cv2.COLOR_BGR2RGB)
    Xg = o + gp[:, :1]*e1 + gp[:, 1:]*e2
    ug = F0*Xg[:, 0]/np.maximum(Xg[:, 2], 1e-4) + 960
    vg = F0*Xg[:, 1]/np.maximum(Xg[:, 2], 1e-4) + 540
    ing = (Xg[:, 2] > 0.05) & (ug >= 0) & (ug < 1919) & (vg >= 0) & (vg < 1079)
    dg = np.full(len(Xg), np.inf)
    dg[ing] = De[np.clip((vg[ing]*h/1080).astype(int), 0, h - 1), np.clip((ug[ing]*w/1920).astype(int), 0, w - 1)]
    sg = ing & (np.abs(Xg[:, 2] - dg) < np.maximum(0.02, 0.03*Xg[:, 2]))
    tmap[sg] = emp[vg[sg].astype(int), ug[sg].astype(int)]
    cv2.imwrite(f'{OUT}/table_map.png', cv2.cvtColor(tmap.reshape(Hm, Wm, 3)[::-1], cv2.COLOR_RGB2BGR))
    ga = np.arange(lo[0], hi[0] + 0.01, 0.01)
    gb = np.arange(lo[1], hi[1] + 0.01, 0.01)
    A_, B_ = np.meshgrid(ga, gb)
    ab_ = np.stack([A_.ravel(), B_.ravel()], 1)
    Vt = o - 0.003*n + ab_[:, :1]*e1 + ab_[:, 1:]*e2
    nx, ny = len(ga), len(gb)
    idg = np.arange(nx*ny).reshape(ny, nx)
    Ft = np.concatenate([np.stack([idg[:-1, :-1], idg[:-1, 1:], idg[1:, 1:]], -1).reshape(-1, 3), np.stack([idg[:-1, :-1], idg[1:, 1:], idg[1:, :-1]], -1).reshape(-1, 3)])
    uv_map = np.stack([(ab_[:, 0] - lo[0])/(hi[0] - lo[0]), (ab_[:, 1] - lo[1])/(hi[1] - lo[1])], 1)
    Xp = o + ab_[:, :1]*e1 + ab_[:, 1:]*e2
    up_ = F0*Xp[:, 0]/np.maximum(Xp[:, 2], 1e-4) + 960
    vp_ = F0*Xp[:, 1]/np.maximum(Xp[:, 2], 1e-4) + 540
    inp_ = (Xp[:, 2] > 0.05) & (up_ >= 0) & (up_ < 1919) & (vp_ >= 0) & (vp_ < 1079)
    dp_ = np.full(len(Xp), np.inf)
    dp_[inp_] = De[np.clip((vp_[inp_]*h/1080).astype(int), 0, h - 1), np.clip((up_[inp_]*w/1920).astype(int), 0, w - 1)]
    seen_p = (inp_ & (np.abs(Xp[:, 2] - dp_) < np.maximum(0.02, 0.03*Xp[:, 2]))).astype(np.float32)
    bg = mesh_obj('seen_surface', Pd, tri, uv=UV, pass_index=1)
    abm = np.stack([(Pd - o) @ e1, (Pd - o) @ e2], 1)
    tl = bg.data.uv_layers.new(name='TopUV')
    tl.data.foreach_set('uv', ((abm - lo)/(hi - lo))[tri.ravel()].astype(np.float32).ravel())
    bg.data.materials.append(material_v2('head_projection', image=f'{PX}/empty_frame.png', image_uv='UV', ao_map=f'{PX}/contact_ao.png', ao_uv='TopUV'))
    tab = mesh_obj('table_plane', Vt, Ft, uv=uv_map, pass_index=2, fattr={'seen': seen_p})
    hl = tab.data.uv_layers.new(name='HeadUV')
    hl.data.foreach_set('uv', np.stack([up_/1920, 1 - vp_/1080], 1)[Ft.ravel()].astype(np.float32).ravel())
    mt = bpy.data.materials.new('table_v3')
    mt.use_nodes = True
    nt = mt.node_tree
    nt.nodes.clear()
    o_ = nt.nodes.new('ShaderNodeOutputMaterial')
    em_ = nt.nodes.new('ShaderNodeEmission')
    t1 = nt.nodes.new('ShaderNodeTexImage')
    t1.image = bpy.data.images.load(f'{OUT}/table_map.png')
    t1.extension = 'EXTEND'
    u1 = nt.nodes.new('ShaderNodeUVMap')
    u1.uv_map = 'UV'
    nt.links.new(u1.outputs['UV'], t1.inputs['Vector'])
    t2 = nt.nodes.new('ShaderNodeTexImage')
    t2.image = bpy.data.images.get('empty_frame.png') or bpy.data.images.load(f'{PX}/empty_frame.png')
    t2.extension = 'EXTEND'
    u2 = nt.nodes.new('ShaderNodeUVMap')
    u2.uv_map = 'HeadUV'
    nt.links.new(u2.outputs['UV'], t2.inputs['Vector'])
    at_ = nt.nodes.new('ShaderNodeAttribute')
    at_.attribute_name = 'seen'
    mx_ = nt.nodes.new('ShaderNodeMix')
    mx_.data_type = 'RGBA'
    nt.links.new(at_.outputs['Fac'], _sock(mx_, 'Factor_Float'))
    nt.links.new(t1.outputs['Color'], _sock(mx_, 'A_Color'))
    nt.links.new(t2.outputs['Color'], _sock(mx_, 'B_Color'))
    ta_ = nt.nodes.new('ShaderNodeTexImage')
    ta_.image = bpy.data.images.get('contact_ao.png') or bpy.data.images.load(f'{PX}/contact_ao.png')
    ta_.image.colorspace_settings.name = 'Non-Color'
    ta_.extension = 'EXTEND'
    nt.links.new(u1.outputs['UV'], ta_.inputs['Vector'])
    fa_ = nt.nodes.new('ShaderNodeMath')
    fa_.operation = 'MULTIPLY_ADD'
    fa_.inputs[1].default_value = a.ao
    fa_.inputs[2].default_value = 1 - a.ao
    nt.links.new(ta_.outputs['Color'], fa_.inputs[0])
    mu_ = nt.nodes.new('ShaderNodeMix')
    mu_.data_type = 'RGBA'
    mu_.blend_type = 'MULTIPLY'
    _sock(mu_, 'Factor_Float').default_value = 1.0
    nt.links.new(_sock(mx_, 'Result_Color', out=True), _sock(mu_, 'A_Color'))
    nt.links.new(fa_.outputs[0], _sock(mu_, 'B_Color'))
    nt.links.new(_sock(mu_, 'Result_Color', out=True), em_.inputs['Color'])
    nt.links.new(em_.outputs[0], o_.inputs['Surface'])
    tab.data.materials.append(mt)
    ctr = np.median(Pd[Pd[:, 2] > 0.05], 0)
    for k, wl in enumerate(cfg['walls']):
        wn, wd = np.array(wl['n']), float(wl['d']); c = ctr - (ctr @ wn + wd)*wn
        f1 = np.cross(wn, [0, 1, 0.]) if abs(wn[1]) < 0.9 else np.cross(wn, [1, 0, 0.]); f1 /= np.linalg.norm(f1); f2 = np.cross(wn, f1)
        Vw = np.array([c + s1*3*f1 + s2*3*f2 for s1, s2 in ((-1, -1), (1, -1), (1, 1), (-1, 1))]) - 0.003*wn
        wo = mesh_obj(f'wall{k}', Vw, np.array([[0, 1, 2], [0, 2, 3]]), pass_index=3 + k); wo.data.materials.append(material_v2(f'wall{k}', colour=wl['colour'], use_ao=False))
    for ob in cfg['objects']:
        z = np.load(f"{PX}/obj{ob['k']}.npz")
        sv = np.load(f"{PX}/obj{ob['k']}_seen.npz")                                           # object_visibility.py
        uvo = np.stack([sv['uv'][:, 0], 1 - sv['uv'][:, 1]], 1)                                  # head-image projection (Blender v up)
        colo = sv['fill'].astype(np.float32) if 'fill' in sv.files else z['C'].astype(np.float32)
        mo = mesh_obj(f"object{ob['k']}", z['V'], z['F'], uv=uvo, col=colo, pass_index=10 + ob['k'], fattr={'seen': sv['seen']})
        mo.data.materials.append(material_v2(f"object{ob['k']}", image=f"{H2W_OUTPUT_ROOT}/lyra2/explore_inputs/{name}/first_frame.png", vertex=True, seen_mix=True))
    cd = bpy.data.cameras.new('wrist_cam')
    cd.sensor_fit = 'HORIZONTAL'
    cd.sensor_width = 36.0
    cd.clip_start = 0.005
    cd.clip_end = 20
    cam = bpy.data.objects.new('wrist_cam', cd)
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.rotation_mode = 'QUATERNION'
    import mathutils
    for i, Mcv in enumerate(C2W):
        Mb = Mcv.copy(); Mb[:3, :3] = Mcv[:3, :3] @ np.diag([1, -1, -1.])                 # OpenCV -> Blender camera axes
        cam.matrix_world = mathutils.Matrix(Mb.tolist())
        cd.lens = K[i][0, 0]*RW/W0*36.0/RW                                                   # fx at 1280 px -> mm on a 36 mm sensor
        cam.keyframe_insert('location', frame=i); cam.keyframe_insert('rotation_quaternion', frame=i); cd.keyframe_insert('lens', frame=i)
    sc.frame_start, sc.frame_end = 0, len(C2W) - 1
    sc.render.fps = 16
    bpy.ops.wm.save_as_mainfile(filepath=f'{OUT}/scene.blend')
    frames = range(len(C2W)) if a.frames == 'all' else [int(x) for x in a.frames.split(',')]
    for i in frames:
        sc.frame_set(i); sc.render.filepath = f'{OUT}/frames/rgb_{i:04d}.png'; bpy.ops.render.render(write_still=True)
    if a.frames == 'all':
        tmp = f'{OUT}/rgb.tmp.mp4'; vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), 16, (1280, 704))
        for i in frames: vw.write(cv2.resize(cv2.imread(f'{OUT}/frames/rgb_{i:04d}.png'), (1280, 704), interpolation=cv2.INTER_AREA))
        vw.release(); subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', tmp, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '14', f'{OUT}/rgb.mp4'], check=True); os.remove(tmp)
        for f in ('depth.mp4', 'seg.mp4', 'guided_mask.mp4', 'source.npy'):                  # same geometry as render_v3
            dst = f'{OUT}/{f}'
            if not os.path.exists(dst): __import__("shutil").copy2(f'{PX}/{a.passes_from}/{f}', dst)
    print('wrote', OUT, len(list(frames)), 'frames')


if __name__ == '__main__':
    main()
