from head2wrist.config import H2W_DATA_ROOT, H2W_OUTPUT_ROOT, H2W_REPO_ROOT
#!/usr/bin/env python3
"""
Cosmos-Transfer2.5 on the complete scene proxy (render_proxy.py) of one demo, two variants:
  T1  depth (ours) + segmentation (ours) + edge and blurred RGB computed from the proxy render
  T2  T1 + guided generation: pixels whose colour came from the real head frame (guided_mask.mp4) are anchored to the
      proxy render for the first --guided_steps denoising steps; the rest (planes, object back sides) is free
  T3  T2 without the blur control, with the real head frame (hands + grasped object removed) as the image context:
      the output must show THIS scene (identity), not a restyle; use --render render_v3 (guided mask without inpainted
      areas)
Run in the Transfer2.5 env (third_party/cosmos-transfer2.5/.venv, HF_HUB_CACHE=$H2W_DATA_ROOT/checkpoints/hf_hub).
Writes $H2W_OUTPUT_ROOT/transfer25/<ep>_<variant>/ (spec.json, the generated video, log).
"""
import argparse, json, os, subprocess, sys

from baselines.prompts import BLOCKOUT_INTENT, get_prompt




def main():
    global D_, R, REPO, RW_, a, ap, cmd, cv2, f, name, np, out, r, spec, v
    ap = argparse.ArgumentParser()
    ap.add_argument('--ep', required=True)
    ap.add_argument('--variants', default='T1,T2')
    ap.add_argument('--guided_steps', type=int, default=25)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--render', default='render_v2', help='proxy render folder (render_proxy.py --out_name)')
    ap.add_argument('--tag', default='', help='suffix for the output folder (e.g. _blender)')
    ap.add_argument('--prompt', default='any', choices=['any', 'hand', 'auto'], help='scene prompt: hand-written (prompts.py), auto (caption_scene.py), any = hand-written if there is one')
    ap.add_argument('--wrist_only', action='store_true', help='generate only the wrist frames (from the end of the dive): 2 chunks instead of 3')
    a = ap.parse_args()
    name = a.ep.replace('/', '_')
    R = f'{H2W_OUTPUT_ROOT}/lyra2/task_inputs/{name}/proxy/{a.render}'
    REPO = f'{H2W_REPO_ROOT}/third_party/cosmos-transfer2.5'
    if a.wrist_only:                                   # trimmed copies of the controls: the dive is a synthetic transition
        import numpy as np, cv2
        D_ = int(np.load(f'{H2W_OUTPUT_ROOT}/lyra2/task_inputs/{name}/task_plan.npz')['dive_frames']); RW_ = f'{R}/wrist'; os.makedirs(RW_, exist_ok=True)
        for f in ('rgb.mp4', 'depth.mp4', 'seg.mp4', 'guided_mask.mp4'):
            # re-trim when the render is newer than the trimmed copy: a re-planned path re-renders R/ but the old wrist/
            # copies stayed and Transfer ran on the previous camera path (planner v4 regeneration, stack/18 + stack/48)
            if os.path.exists(f'{R}/{f}') and (not os.path.exists(f'{RW_}/{f}') or os.path.getmtime(f'{RW_}/{f}') < os.path.getmtime(f'{R}/{f}')):
                subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', f'{R}/{f}', '-vf', f'select=gte(n\\,{D_})', '-vsync', '0', '-c:v', 'libx264', '-crf', '12', '-pix_fmt', 'yuv420p', f'{RW_}/{f}'], check=True)
        R = RW_
    for v in a.variants.split(','):
        out = f'{H2W_OUTPUT_ROOT}/transfer25/{name}_{v}{a.tag}'; os.makedirs(out, exist_ok=True)
        spec = dict(name=f'{name}_{v}{a.tag}', prompt=get_prompt(a.ep, a.prompt), video_path=f'{R}/rgb.mp4', guidance=3, seed=a.seed,
                    depth=dict(control_path=f'{R}/depth.mp4', control_weight=1.0),
                    seg=dict(control_path=f'{R}/seg.mp4', control_weight=0.5),
                    edge=dict(control_weight=0.3),
                    vis=dict(control_weight=0.5, preset_blur_strength='medium'))
        if v in ('T2', 'T3'):
            spec.update(guided_generation_mask=f'{R}/guided_mask.mp4', guided_generation_step_threshold=a.guided_steps)
        if v == 'TB':                     # blockout: no real pixels to anchor; head frame as the reference, scene-intent prompt
            spec.pop('vis'); spec['image_context_path'] = f'{H2W_OUTPUT_ROOT}/lyra2/task_inputs/{name}/first_frame.png'
            spec['prompt'] = get_prompt(a.ep, a.prompt) + BLOCKOUT_INTENT[a.ep]
        if v == 'T3':                     # scene fidelity: the real head frame as the appearance reference (replaces blur;
            spec.pop('vis')               # Transfer2.5 refuses vis + image_context together)
            spec['image_context_path'] = f'{H2W_OUTPUT_ROOT}/lyra2/task_inputs/{name}/first_frame.png'
        json.dump(spec, open(f'{out}/spec.json', 'w'), indent=1)
        cmd = (f'cd {REPO} && source .venv/bin/activate && export HF_HUB_CACHE={H2W_DATA_ROOT}/checkpoints/hf_hub && '
               f'python examples/inference.py -i {out}/spec.json -o {out} --disable-guardrails > {out}/log.txt 2>&1')
        r = subprocess.run(['bash', '-c', cmd]); print(v, 'exit', r.returncode, '->', out, flush=True)
        if r.returncode: print(open(f'{out}/log.txt').read()[-3000:]); sys.exit(1)


if __name__ == '__main__':
    main()
