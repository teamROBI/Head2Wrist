from head2wrist.config import H2W_DATA_ROOT, H2W_OUTPUT_ROOT
#!/usr/bin/env python3
"""
The gripper texture for the SIM test: gripper_texture.py's procedure (per-vertex colour from wrist frames where the
gripper is still) on Isaac sim wrist frames instead of the real FFW-SG2 recordings, so the gripper layer of the generated
views looks like the gripper of the camera the policy is tested on (the real texture against the sim camera was a
sim-test-only gap: exp-r8 multi-scene diagnosis). The frames come from a scene that is NOT under test (default the
stack_18 demos), so nothing of the test scenes leaks in; on the real robot the real texture plays this role.
Writes $H2W_DATA_ROOT/assets/gripper_texture_sim.npz (same format; gripper_render.render_textured(tex=...)).
"""
import argparse, glob, os, sys
import numpy as np, cv2, trimesh
import head2wrist.rendering.gripper_render as GR
import head2wrist.rendering.gripper_texture as GT
from head2wrist.wrist_pose import T_TCP_D405





def main():
    global D, L, NV, P, S, T, Tc, Tp, W_, a, ap, c, cnt, col, cv_, f, files, fp, free, g, g0, gv, i, idx, j, l, li, m, meshes, off, rng, seen, t, u, used, v, z
    ap = argparse.ArgumentParser()
    ap.add_argument('--demos', default=f'{H2W_OUTPUT_ROOT}/sim_policy/stack_18/demos')
    ap.add_argument('--out', default=f'{H2W_DATA_ROOT}/assets/gripper_texture_sim.npz')
    ap.add_argument('--episodes', type=int, default=80)
    ap.add_argument('--frames_per_episode', type=int, default=30)
    ap.add_argument('--edge', type=float, default=0.0008)
    ap.add_argument('--k', type=int, default=64)
    a = ap.parse_args()
    meshes = []
    for l in GR.LINKS:
        m = GR._mesh(f'gripper_r_{l}'); v, f = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, max_edge=a.edge)
        meshes.append(trimesh.Trimesh(v, f, process=False))
    off = np.cumsum([0] + [len(m.vertices) for m in meshes])
    NV = off[-1]
    Tc = np.linalg.inv(T_TCP_D405)
    seen = [np.zeros(len(m.vertices), bool) for m in meshes]
    for g0 in np.arange(0, GR.G_MAX + 1e-6, 0.025):                       # candidate vertices: visible at some opening
        Tp = GR.link_poses_tcp('right', g0); D, L = GT.rendered(g0)
        for li, m in enumerate(meshes):
            T = Tc @ Tp[GR.LINKS[li]]; P = m.vertices @ T[:3, :3].T + T[:3, 3]
            u = np.round(GT.KC[0, 0]*P[:, 0]/P[:, 2] + GT.KC[0, 2] - 0.5).astype(int); v = np.round(GT.KC[1, 1]*P[:, 1]/P[:, 2] + GT.KC[1, 2] - 0.5).astype(int)
            j = np.nonzero((u >= 0) & (u < GT.W) & (v >= 0) & (v < GT.H) & (P[:, 2] > 0.01))[0]
            seen[li][j[(L[v[j], u[j]] == li) & (np.abs(D[v[j], u[j]] - P[j, 2]) < 0.0015)]] = True
    GT.CAND = [(off[li] + np.nonzero(seen[li])[0], meshes[li].vertices[seen[li]], meshes[li].vertex_normals[seen[li]]) for li in range(len(meshes))]
    S = np.zeros((NV, a.k, 3), np.uint8)
    cnt = np.zeros(NV, np.int32)
    rng = np.random.default_rng(0)
    used = 0
    files = sorted(glob.glob(f'{a.demos}/ep_*.npz'))
    files = [files[i] for i in sorted(rng.choice(len(files), min(a.episodes, len(files)), replace=False))]
    for fp in files:
        z = np.load(fp); W_, g = z['wrist'], z['state'][:, 7].astype(float); idx = np.nonzero(np.abs(np.gradient(g)) < 0.003)[0]
        for t in np.sort(rng.choice(idx, min(a.frames_per_episode, len(idx)), replace=False)):
            gv, col = GT.sample(float(g[t]), cv2.cvtColor(W_[t], cv2.COLOR_RGB2BGR))     # the sampler works in BGR like the videos
            free = cnt[gv] < a.k; gv, col = gv[free], col[free]; S[gv, cnt[gv]] = col; cnt[gv] += 1; used += 1
    cv_ = np.concatenate([c[0] for c in GT.CAND])
    print(f'{len(files)} episodes, {used} frames | candidate vertices with >= 5 samples: {100*(cnt[cv_] >= 5).mean():.1f}%, median samples {np.median(cnt[cv_]):.0f}')
    GT.save(a.out, meshes, off, GT.finalize(meshes, off, S, cnt), cnt)


if __name__ == '__main__':
    main()
