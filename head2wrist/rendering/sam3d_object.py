from head2wrist.config import H2W_SAM3D_ROOT
from head2wrist.config import H2W_OUTPUT_ROOT

def main():
    #!/usr/bin/env python3
    """
    3D model of the grasped object from the input frame (SAM 3D Objects), placed in the head camera frame with OUR metric
    depth: the pipeline takes a point map (here: first_depth.npy of the explore inputs, OpenCV axes -> PyTorch3D
    x left, y up) and returns the object's local->camera rotation / translation / scale in that frame.
    Run in the sam3d-objects env ($H2W_SAM3D_ROOT, checkpoints $H2W_SAM3D_ROOT/checkpoints/hf):
      $H2W_SAM3D_PYTHON head2wrist/rendering/sam3d_object.py --ep basic_pick_place/74
    Writes <out>_sam3d.npz (default <task_inputs>/object_mesh_sam3d.npz; refine_object.py turns it into <out>.npz): vertices (camera frame, OpenCV axes, m), faces, vertex colours (RGB uint8), the
    pose, and object_mesh_check.jpg (the mesh projected over the input frame).
    """
    import argparse, os, sys
    import numpy as np, cv2, torch
    S3D = os.path.expanduser(f'{H2W_SAM3D_ROOT}')
    # The upstream notebook helper imports modules from its own checkout.
    sys.path.insert(0, S3D)
    sys.path.insert(0, os.path.join(S3D, 'notebook'))

    ap = argparse.ArgumentParser()
    ap.add_argument('--ep', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--mask', default=None, help='object mask .npy (default: <task_inputs>/object_mask.npy, the grasped object)')
    ap.add_argument('--out', default=None, help='output prefix (default: <task_inputs>/object_mesh)')
    a = ap.parse_args(); name = a.ep.replace('/', '_')
    I = f'{H2W_OUTPUT_ROOT}/lyra2/explore_inputs/{name}'; T = f'{H2W_OUTPUT_ROOT}/lyra2/task_inputs/{name}'
    img = cv2.cvtColor(cv2.imread(f'{I}/first_frame.png'), cv2.COLOR_BGR2RGB); H0, W0 = img.shape[:2]
    mask = np.load(a.mask or f'{T}/object_mask.npy').astype(np.uint8); OUT = a.out or f'{T}/object_mesh'
    import os as _os, sys as _sys; import head2wrist.headcam as headcam; F0 = headcam.focal(I)
    D = np.load(f'{I}/first_depth.npy').astype(np.float32); h, w = D.shape; fx, fy = F0*w/1920, F0*h/1080
    vv, uu = np.mgrid[0:h, 0:w]; P = np.stack([(uu - w/2)/fx*D, (vv - h/2)/fy*D, D], -1)                  # OpenCV
    P = cv2.resize(P, (W0, H0), interpolation=cv2.INTER_NEAREST)
    Pp = (P*np.array([-1, -1, 1])).astype(np.float32)                                                           # PyTorch3D camera
    cwd = os.getcwd(); os.chdir(S3D)
    # the DINO backbone comes from torch.hub (source github): without an explicit branch torch.hub asks github.com for the
    # default branch on every load, even when cached, so a GitHub outage (HTTP 504) aborted the pipeline. Load the cached
    # repo (and its cached weights) locally when present.
    _hub_load = torch.hub.load
    def _hub_load_cached(repo_or_dir, model, *args, source='github', **kw):
        loc = os.path.expanduser(f"~/.cache/torch/hub/{repo_or_dir.replace('/', '_')}_main")
        if source == 'github' and os.path.isdir(loc): return _hub_load(loc, model, *args, source='local', **kw)
        return _hub_load(repo_or_dir, model, *args, source=source, **kw)
    torch.hub.load = _hub_load_cached
    from inference import Inference
    inf = Inference('checkpoints/hf/pipeline.yaml', compile=False)
    out = inf(img, mask, seed=a.seed, pointmap=torch.from_numpy(Pp))
    os.chdir(cwd)
    print('output keys', list(out.keys()))
    from pytorch3d.transforms import quaternion_to_matrix
    Rq = quaternion_to_matrix(out['rotation'].float()).cpu().numpy().reshape(3, 3)
    t = out['translation'].float().cpu().numpy().reshape(3); s = out['scale'].float().cpu().numpy().reshape(-1)
    glb = out['glb']; import trimesh
    m = glb if isinstance(glb, trimesh.Trimesh) else glb.dump(concatenate=True)
    V = np.asarray(m.vertices, np.float64); F = np.asarray(m.faces, np.int64)
    try: C = np.asarray(m.visual.vertex_colors)[:, :3].astype(np.uint8)
    except Exception: C = np.full((len(V), 3), 160, np.uint8)
    np.savez(f'{OUT}_raw.npz', V=V, F=F, C=C, R=Rq, t=t, s=s,
             gs_xyz=out['gaussian'][0].get_xyz.detach().float().cpu().numpy())
    print('glb verts bbox', np.round(V.min(0), 3), np.round(V.max(0), 3), '| gaussian xyz bbox',
          np.round(out['gaussian'][0].get_xyz.min(0).values.cpu().numpy(), 3), np.round(out['gaussian'][0].get_xyz.max(0).values.cpu().numpy(), 3))
    print('rotation', np.round(Rq, 3).tolist(), 'translation', np.round(t, 3), 'scale', np.round(s, 4))
    # ---- place the mesh in the head camera frame (OpenCV axes) ----
    # glb vertices = local @ [[1,0,0],[0,0,-1],[0,1,0]] (z-up -> y-up, to_glb) -> local = (x, -z, y);
    # pose = PyTorch3D Transform3d scale -> rotate -> translate on row vectors: p3d = (s*local) @ R + t; OpenCV = p3d*(-1,-1,1)
    Vl = np.stack([V[:, 0], -V[:, 2], V[:, 1]], 1)
    Vc = ((s*Vl) @ Rq + t)*np.array([-1, -1, 1.])
    np.savez(f'{OUT}_sam3d.npz', V=Vc.astype(np.float32), F=F.astype(np.int32), C=C, R=Rq, t=t, s=s)
    K0 = np.array([[F0, 0, W0/2], [0, F0, H0/2], [0, 0, 1]])
    uv = (Vc @ K0.T)[:, :2]/Vc[:, 2:3]; chk = img.copy()
    cnt, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(chk, cnt, -1, (0, 255, 0), 2)
    for (x, y), c in zip(uv[::3].astype(int), C[::3]):
        if 0 <= x < W0 and 0 <= y < H0: chk[y, x] = (255, 0, 255)
    ys, xs = np.nonzero(mask); x0, x1, y0, y1 = max(0, xs.min()-80), xs.max()+80, max(0, ys.min()-80), ys.max()+80
    cv2.imwrite(f'{OUT}_sam3d_check.jpg', cv2.cvtColor(cv2.resize(chk[y0:y1, x0:x1], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR))
    ext = Vc.max(0) - Vc.min(0); print('object extents (cm)', np.round(100*ext, 1), 'centre', np.round(Vc.mean(0), 3))


if __name__ == '__main__':
    main()
