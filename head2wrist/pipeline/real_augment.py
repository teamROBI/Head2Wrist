from head2wrist.config import H2W_DATA_ROOT, H2W_OUTPUT_ROOT, H2W_PYTHON, H2W_REPO_ROOT
#!/usr/bin/env python3
"""
The sim test's data recipe on a REAL (converted Quest) demo: one human demonstration -> many training episodes, no robot.
The sim policies were trained on ~250 episodes per scene: the twin's scripted expert re-planned the gate-validated grasp for
randomized object starts and stances (sim_policy/expert_plan.py), and the generator made each episode's wrist views from
one head frame. With only the human demos' own starts the generated-view policy drops (stack/48: 20 demos -> 20/30 in range,
100 -> 25/30, 252 -> 29/30; outputs/sim_policy/stack_48/results_strict_n{20,100}.txt). Here, per converted demo <task>/<k>
(after run_episode.py --to compose_v5, whose sim gate built outputs/sim_twin/<task>_<k>):
  1 plans   expert_plan.py on the demo's twin: --n starts (object +-8 cm / +-30 deg, the training range of the sim test) with a
            randomized stance (--base_pert), each IK-checked at the demo's lift
  2 demos   one npz per plan in the format the generator and to_lerobot_repr.py read: the PLANNED arm + gripper path (state),
            the next state (action), phases, the object's start pose, the base pose. Kinematic: the robot replays planned paths
            on the real robot too, and the generator attaches the object to the gripper by the jaw schedule
  3 views   sim_policy/gen_episode.py --inputs real: the pipeline's reconstruction of the REAL head frame (proxy, object
            mesh from the twin at the start pose, URDF gripper with the REAL gripper texture, robot body, randomized shadows)
-> outputs/real/<task>/augment/<k>/{demos/ep_XXXX.npz, views/ep_XXXX.npy, augment.json}; resumable (done views are kept).
  real_augment.py --ep <task>/<k> [--n 84] [--workers 6] [--seed 1000+k]
Nothing here looks at the deployment scene through the robot: the inputs are the human's head video and the twin made from it.
"""
import argparse, json, os, subprocess, sys, time
import numpy as np

REPO = f'{H2W_REPO_ROOT}'; X = f'{REPO}/head2wrist'; PY = H2W_PYTHON; OUT = f'{H2W_OUTPUT_ROOT}'




def main():
    global A, D, DM, P, S, V, a, ap, f, idx, k, lf, log, name, nv, obj, p, p_, pj, procs, r, rc, seed, sub, t0, task, w, x
    ap = argparse.ArgumentParser()
    ap.add_argument('--ep', required=True)
    ap.add_argument('--n', type=int, default=84)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--gripper_tex', default=f'{H2W_DATA_ROOT}/assets/gripper_texture.npz', help='the REAL D405 gripper texture')
    ap.add_argument('--aug_dir', default='augment', help='output folder under outputs/real/<task>/ (a variant keeps the same plans and demos)')
    ap.add_argument('--obj_size', type=float, default=None, help='optional measured-size override (m); the paper floor recipe leaves this unset and uses the native reconstructed mesh')
    a = ap.parse_args()
    task, k = a.ep.split('/')
    name = a.ep.replace('/', '_')
    seed = a.seed if a.seed is not None else 1000 + int(k)
    D = f'{OUT}/real/{task}/{a.aug_dir}/{k}'
    DM, V = f'{D}/demos', f'{D}/views'
    os.makedirs(DM, exist_ok=True)
    os.makedirs(V, exist_ok=True)
    log = lambda *x: print(f'[{time.strftime("%H:%M")}] {a.ep}:', *x, flush=True)
    assert os.path.exists(f'{OUT}/sim_twin/{name}/twin.json'), f'no twin for {a.ep}: run pipeline/run_episode.py through simgate first'
    assert os.path.exists(f'{OUT}/lyra2/task_inputs/{name}/proxy'), f'no real reconstruction for {a.ep}: run pipeline/run_episode.py through compose_v5 first'
    pj = f'{OUT}/sim_policy/{name}/plans_aug.json'
    if not (os.path.exists(pj) and len(json.load(open(pj))['plans']) >= a.n):
        r = subprocess.run([PY, f'{X}/policy/expert_plan.py', '--ep', a.ep, '--n', str(a.n), '--seed', str(seed), '--base_pert', '--out_name', 'plans_aug'],
                           capture_output=True, text=True)
        if r.returncode: sys.exit(f'{a.ep}: expert_plan failed: {r.stderr[-800:]}')
    P = json.load(open(pj))
    if not P['plans']:      # the planner keeps every start 2 cm clear of all other objects (EgoDex bpp/65: a toy resting on a card)
        sys.exit(f'{a.ep}: no feasible start: is the object on or right next to another object (a mat, a coaster, the target)? '
                 'Record with the object free-standing, >= 5 cm from anything else and ~20 cm from the target')
    if len(P['plans']) < a.n: log(f"only {len(P['plans'])} of {a.n} starts feasible (the object is close to another object)")
    log(f"plans: {len(P['plans'])} ({P['plans'][0]['arm_q'].__len__()} frames in the first)")
    for p in P['plans']:
        f = f'{DM}/ep_{p["i"]:04d}.npz'
        if os.path.exists(f): continue
        S = np.c_[np.asarray(p['arm_q'], float), np.asarray(p['gripper_q'], float)]          # (T, 8): arm joints + gripper joint
        A = np.r_[S[1:], S[-1:]]                                                                # action: the next planned state
        obj = np.tile(np.r_[np.asarray(p['object_start'], float), np.asarray(p['object_quat_wxyz'], float)], (len(S), 1))
        np.savez(f, state=S, action=A, phase=np.asarray(p['phase']), object=obj, base_pose=np.asarray(p['base_pose'], float),
                 success=True, planned=True, source=f'expert_plan on the twin of {a.ep} (kinematic)')
    log(f'demo files: {len(P["plans"])} in {DM}')
    idx = [p['i'] for p in P['plans'] if not os.path.exists(f'{V}/ep_{p["i"]:04d}.npy')]
    if idx:
        t0 = time.time(); procs = []
        for w in range(a.workers):
            sub = idx[w::a.workers]
            if not sub: continue
            lf = open(f'{D}/gen_w{w}.log', 'w')
            procs.append(subprocess.Popen([PY, f'{X}/policy/gen_episode.py', '--scene', a.ep, '--inputs', 'real', '--demos', DM, '--episodes', ','.join(map(str, sub)),
                                           '--out', V, '--gripper_tex', a.gripper_tex] + (['--obj_size', str(a.obj_size)] if a.obj_size else []), stdout=lf, stderr=subprocess.STDOUT))
        rc = [p_.wait() for p_ in procs]
        if any(rc): sys.exit(f'{a.ep}: view generation failed in a worker (see {D}/gen_w*.log)')
        log(f'views: {len(idx)} generated in {(time.time() - t0)/60:.0f} min ({a.workers} workers)')
    nv = len([x for x in os.listdir(V) if x.endswith('.npy')])
    json.dump(dict(ep=a.ep, plans=len(P['plans']), views=nv, seed=seed, twin=f'{OUT}/sim_twin/{name}', gripper_tex=a.gripper_tex,
                   demos=DM, views_dir=V), open(f'{D}/augment.json', 'w'), indent=1)
    log(f'done: {nv} episodes with generated views -> {D}')


if __name__ == '__main__':
    main()
