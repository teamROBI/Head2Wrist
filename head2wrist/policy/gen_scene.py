from head2wrist.config import H2W_DATA_ROOT, H2W_OUTPUT_ROOT, H2W_PYTHON
#!/usr/bin/env python3
"""
Generated-wrist arm of the sim policy test, scene part: the pipeline's own reconstruction of the sim scene from one
human-head frame, the way it reconstructs an EgoDex scene. Input: the twin rendered from the EgoDex head pose at
1920x1080 with the robot and the grasped object hidden (collect_demos.py --clean_hr; EgoDex: the input frame with the
hands and the grasped object removed; the grasped object is composited per episode by the object layer).
Depth: single-view DA3-Nested of that frame, then D' = a*D + b fitted to the table plane (EgoDex: the plane below the
fingertips; here the twin's table at the known head pose - the same kind of anchor). Then the pipeline stages on the
pseudo-episode 'sim/<scene>': sam3_objects, extend_table, build_proxy (no grasped object in the frame -> empty
object mask), build_static_objects, object_visibility, contact_ao.
Writes explore_inputs/sim_<scene>/ and task_inputs/sim_<scene>/ (proxy/ shared by every episode).
"""
import argparse, json, os, subprocess, sys
import numpy as np, cv2, torch
X = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); REPO = os.path.dirname(X); PY = H2W_PYTHON
def main():
    global D, DepthAnything3, I, T, TW, Th, _, a, a_d, ap, b_d, cand, cmd, d, ep, fx, fy, h, img, meta, model, name, pm, r, rz, sel, sname, steps, th, uu, vv, w, x, zt
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='stack/18')
    ap.add_argument('--da3', default=f'{H2W_DATA_ROOT}/checkpoints/da3/DA3NESTED-GIANT-LARGE-1.1')
    a = ap.parse_args()
    sname = a.scene.replace('/', '_')
    ep = f'sim/{sname}'
    name = f'sim_{sname}'
    TW = json.load(open(f'{H2W_OUTPUT_ROOT}/sim_twin/{sname}/twin.json'))
    I = f'{H2W_OUTPUT_ROOT}/lyra2/explore_inputs/{name}'
    T = f'{H2W_OUTPUT_ROOT}/lyra2/task_inputs/{name}'
    for d in (I, T): os.makedirs(d, exist_ok=True)
    img = cv2.imread(f'{H2W_OUTPUT_ROOT}/sim_policy/{sname}/gen/clean_head_1920.png')
    assert img is not None
    for d in (I, T): cv2.imwrite(f'{d}/first_frame.png', img)
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(a.da3).to('cuda').eval()
    with torch.no_grad(): pm = model.inference([cv2.cvtColor(img, cv2.COLOR_BGR2RGB)], process_res=756)
    D = pm.depth[0].astype(np.float32)
    h, w = D.shape
    del model
    torch.cuda.empty_cache()
    Th = np.array(TW['human_head_world'])
    th = TW['table_h']
    vv, uu = np.mgrid[0:h, 0:w]
    fx, fy = 736.6*w/1920, 736.6*h/1080
    rz = np.stack([(uu - w/2)/fx, (vv - h/2)/fy, np.ones_like(D)], -1) @ Th[2, :3]
    zt = (th - Th[2, 3])/np.minimum(rz, -1e-3)
    cand = (vv > 0.45*h) & (D > 0.05) & (rz < -0.05) & (zt < 3.0)
    a_d, b_d = float(np.median(zt[cand]/D[cand])), 0.0
    for _ in range(6):
        sel = cand & (np.abs(a_d*D + b_d - zt) < 0.02)
        a_d, b_d = [float(x) for x in np.linalg.lstsq(np.stack([D[sel], np.ones(sel.sum())], 1), zt[sel], rcond=None)[0]]
    D = (a_d*D + b_d).astype(np.float32)
    print(f'depth from the table plane: D = {a_d:.3f} D_da3 + {100*b_d:.1f} cm ({sel.sum()/cand.sum():.0%} table inliers)', flush=True)
    for d in (I, T): np.save(f'{d}/first_depth.npy', D)
    np.save(f'{T}/object_mask.npy', np.zeros(img.shape[:2], bool))
    meta = dict(ep=ep, t0=0, side=TW['side'], n_frames=0, dive_frames=0, sim_scene=a.scene, depth_affine=[a_d, b_d])
    for d in (I, T): json.dump(meta, open(f'{d}/meta.json', 'w'), indent=1)
    steps = [[PY, f'{X}/vision/sam3_objects.py', '--inputs', I], [PY, f'{X}/scene/extend_table.py', '--inputs', T],
             [PY, f'{X}/scene/build_proxy.py', '--ep', ep], [PY, f'{X}/scene/build_static_objects.py', '--ep', ep],
             [PY, f'{X}/scene/object_visibility.py', '--ep', ep], [PY, f'{X}/scene/contact_ao.py', '--ep', ep]]
    for cmd in steps:
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(os.path.basename(cmd[1]), 'exit', r.returncode, '|', (r.stdout.strip().splitlines() or [''])[-1][:160], flush=True)
        if r.returncode: print(r.stderr[-2500:]); sys.exit(1)
    print('scene ready:', T)


if __name__ == '__main__':
    main()
