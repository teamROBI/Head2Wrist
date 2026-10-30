from head2wrist.config import H2W_SAM3D_PYTHON
from head2wrist.config import H2W_OUTPUT_ROOT, H2W_PYTHON
#!/usr/bin/env python3
"""SAM 3D model + silhouette refinement for every static object of the proxy (proxy.json from build_proxy.py):
sam3d_object.py in the sam3d-objects env, refine_object.py in the project env. Skips objects already done (mesh newer
than its mask)."""
import argparse, json, os, subprocess, sys




def main():
    global PX, PY, S3D, X, a, ap, cmd, k, name, ob, pre, r
    ap = argparse.ArgumentParser()
    ap.add_argument('--ep', required=True)
    a = ap.parse_args()
    name = a.ep.replace('/', '_')
    PX = f'{H2W_OUTPUT_ROOT}/lyra2/task_inputs/{name}/proxy'
    X = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    S3D = os.path.expanduser(f'{H2W_SAM3D_PYTHON}')
    PY = H2W_PYTHON
    for ob in json.load(open(f'{PX}/proxy.json'))['objects']:
        k = ob['k']; pre = f'{PX}/obj{k}'
        if os.path.exists(f'{pre}.npz') and os.path.getmtime(f'{pre}.npz') > os.path.getmtime(f'{PX}/obj{k}_mask.npy'):
            print(f'obj{k}: done'); continue                           # (a mesh older than its mask is of another object)
        for cmd in ([S3D, f'{X}/rendering/sam3d_object.py', '--ep', a.ep, '--mask', f'{PX}/obj{k}_mask.npy', '--out', pre],
                    [PY, f'{X}/rendering/refine_object.py', '--ep', a.ep, '--mask', f'{PX}/obj{k}_mask.npy', '--prefix', pre]):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode: sys.exit(f'obj{k} failed: {" ".join(cmd[:2])}\n{r.stderr[-2000:]}')
        print(f'obj{k}:', r.stdout.strip().splitlines()[-1][:200])
    open(f'{PX}/static3d.done', 'w').write('ok\n')


if __name__ == '__main__':
    main()
