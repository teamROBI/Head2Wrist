from head2wrist.config import H2W_DATA_ROOT, H2W_LEROBOT_PYTHON, H2W_OUTPUT_ROOT, H2W_PYTHON, H2W_REPO_ROOT
#!/usr/bin/env python3
"""
One task's Quest recordings -> converted demos -> generated wrist views -> LeRobot dataset -> training, resumable.
  real_batch.py --task <name> --sessions <dir,dir,...|glob> --calib <calib.json> --description "<what the hand does>"
                [--objects a,b] [--side right] [--windows "k:s:e,..."] [--lift auto|free|<m>] [--jobs 2]
                [--steps convert,pipeline,augment,dataset,train] [--variants 4] [--min_tolerance 0.75] [--train_steps 80000]
                [--aug_total 252] [--aug_workers 6]
  convert   session k -> $H2W_DATA_ROOT/data/lab/<task>/<k>.hdf5 + .mp4 (capture/egokit_to_egodex.py); --windows
            "k:start_s:end_s" trims session k (one demo out of a longer recording)
  pipeline  pipeline/run_episode.py --to compose_v5 per demo, --jobs at a time (each holds SAM 3 / SAM 3D / Isaac on the GPU
            for a while: keep it small). The robot stands at ONE lift for the task: --lift auto plans demo 0 freely and
            fixes its lift for the others (setup_pose.py --lift); a number fixes it; free lets every demo choose
  augment   the sim test's data recipe (pipeline/real_augment.py): per good demo, the twin's expert re-plans the grasp for
            randomized object starts and stances (--aug_total episodes over the good demos) and the generator makes their
            wrist views from the REAL head frame -> outputs/real/<task>/augment/<k>/. The human demos' own starts alone are too
            few (stack/48 sim: 20 demos -> 20/30 in range vs 29/30 with 252)
  dataset   the demos with a v5 clip and stance placement tolerance >= --min_tolerance (robot_setup.json) ->
            pipeline/real_to_lerobot.py with --variants stance variants -> $H2W_OUTPUT_ROOT/real/<task>/lerobot_v5
            (also the per-demo metadata robot_day.sh reads); with augmented episodes, also the training set
            sim_policy/to_lerobot_repr.py -> $H2W_OUTPUT_ROOT/real/<task>/lerobot_aug_floor
  train     AUG=strong REPR=eef_hybrid scripts/train_repr.sh, in the background, on lerobot_aug_floor
Status per demo (stage reached, sim gate, lift, placement tolerance, clip) -> $H2W_OUTPUT_ROOT/real/<task>/batch.json
"""
import argparse, glob, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

REPO = f'{H2W_REPO_ROOT}'; X = f'{REPO}/head2wrist'; PY = H2W_PYTHON; LP = H2W_LEROBOT_PYTHON
LAB = f'{H2W_DATA_ROOT}/data/lab'; OUT = f'{H2W_OUTPUT_ROOT}'


def status(k):
    ep = f'{a.task}/{k}'; n = ep.replace('/', '_'); T = f'{OUT}/lyra2/task_inputs/{n}'; d = dict(ep=ep, session=sessions[k])
    lp = f'{R}/logs/{k}.log'
    if os.path.exists(lp):
        last = {}                                      # a re-run appends to the log: the last exit code of each stage counts
        for l in open(lp):
            if '] exit ' in l: last[l.split(']')[0].strip(' [')] = l.split('] exit ')[1].split()[0]
        d['last_stage'] = list(last)[-1] if last else None; d['failed_stage'] = next((k for k, v in last.items() if v != '0'), None)
    if os.path.exists(f'{T}/robot_setup.json'):
        s = json.load(open(f'{T}/robot_setup.json'))['summary']; d['lift'] = s['setup']['lift_m']; d['tolerance'] = s['search'].get('placement_tolerance')
    g = f'{OUT}/sim_twin/{n}/gate.json'
    if os.path.exists(g): gg = json.load(open(g)); d['gate'] = gg['variant'] if gg['passed'] else 'rejected'
    d['clip'] = os.path.exists(f'{OUT}/composite/{n}_v5/clip.npz')
    aj = f'{R}/augment/{k}/augment.json'                   # augmented episodes with generated views (real_augment.py)
    if os.path.exists(aj): d['augment'] = json.load(open(aj))['views']
    return d


def save():
    B['demos'] = {str(k): status(k) for k in range(len(sessions))}; json.dump(B, open(BJ, 'w'), indent=1)


def table():
    save(); print(f'\n{"demo":>4s} {"session":28s} {"stage":>11s} {"gate":>8s} {"lift":>6s} {"tol":>5s} clip')
    for k, d in B['demos'].items():
        print(f'{k:>4s} {os.path.basename(d["session"])[:28]:28s} {(d.get("failed_stage") and "FAIL " + d["failed_stage"]) or d.get("last_stage") or "-":>11s} '
              f'{d.get("gate", "-"):>8s} {d.get("lift", float("nan")):6.2f} {d.get("tolerance") if d.get("tolerance") is not None else float("nan"):5.2f} {"yes" if d["clip"] else "no"}')


def good_demos():
    return [d['ep'] for d in B['demos'].values() if d['clip'] and (d.get('tolerance') or 0) >= a.min_tolerance]


def main():
    global B, BJ, R, W, a, aj, ap, aug, c, cmd, d, dm, done_lifts, ds, e, ex, extra, good, k, ks, l, lf, lift, lifts, log, low, lp, n_each, n_ep, old, out, r, rd, run_name, s, sessions, steps, vw, w
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--sessions', required=True)
    ap.add_argument('--calib', required=True)
    ap.add_argument('--description', required=True)
    ap.add_argument('--objects', default='')
    ap.add_argument('--side', default='right', choices=['right', 'left'])
    ap.add_argument('--windows', default='')
    ap.add_argument('--lift', default='auto')
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--steps', default='convert,pipeline,augment,dataset,train')
    ap.add_argument('--variants', type=int, default=4)
    ap.add_argument('--aug_total', type=int, default=252, help='augmented episodes over all good demos (paper: 84 each for three demos; 0: none)')
    ap.add_argument('--aug_workers', type=int, default=6)
    ap.add_argument('--min_tolerance', type=float, default=0.75)
    ap.add_argument('--train_steps', type=int, default=80000)
    a = ap.parse_args()
    steps = a.steps.split(',')
    assert '/' not in a.task and a.task, 'task: a plain name'
    sessions = sorted(glob.glob(a.sessions)) if any(c in a.sessions for c in '*?[') else [s for s in a.sessions.split(',') if s]
    assert sessions and all(os.path.exists(os.path.join(s, 'poses.txt')) for s in sessions), f'sessions without poses.txt: {[s for s in sessions if not os.path.exists(os.path.join(s, "poses.txt"))]}'
    W = {int(k): (float(s), float(e)) for k, s, e in (w.split(':') for w in a.windows.split(',') if w)}
    R = f'{OUT}/real/{a.task}'
    os.makedirs(f'{R}/logs', exist_ok=True)
    BJ = f'{R}/batch.json'
    B = json.load(open(BJ)) if os.path.exists(BJ) else dict(task=a.task, demos={})
    log = lambda *m: print(time.strftime('[%H:%M:%S]'), *m, flush=True)
    if 'convert' in steps:
        for k, s in enumerate(sessions):
            if os.path.exists(f'{LAB}/{a.task}/{k}.hdf5'): continue
            cmd = [PY, f'{X}/capture/egokit_to_egodex.py', '--session', s, '--calib', a.calib, '--out', LAB, '--task', a.task, '--idx', str(k),
                   '--side', a.side, '--description', a.description, '--objects', a.objects]
            if k in W: cmd += ['--start_s', str(W[k][0]), '--end_s', str(W[k][1])]
            r = subprocess.run(cmd, capture_output=True, text=True); out = r.stdout.strip().splitlines()
            for l in out[:-1]:                                     # the converter's diagnostics: dropped / filled pose rows, undecodable frames
                if l.startswith(('poses.txt:', 'WARNING')) or 'outside the pose stream' in l: log(f'convert {k}: {l[:200]}')
            log(f'convert {k} ({os.path.basename(s)}):', (out or ['?'])[-1][:200] if not r.returncode else 'FAILED ' + r.stderr[-300:])
    if 'pipeline' in steps:
        def run(k, extra=''):
            if os.path.exists(f'{OUT}/composite/{a.task}_{k}_v5/clip.npz'): return
            lp = f'{R}/logs/{k}.log'; log(f'pipeline {k} start {extra}')
            with open(lp, 'a') as f:
                subprocess.run([PY, f'{X}/pipeline/run_episode.py', '--root', LAB, '--ep', f'{a.task}/{k}', '--side', a.side, '--to', 'compose_v5'] + (['--stance_args', extra] if extra else []), stdout=f, stderr=subprocess.STDOUT)
            d = status(k); why = ''
            if not d['clip']:                                      # the reason, so the table is not just 'FAILED'
                ls_ = [l.strip() for l in open(lp, errors='ignore') if l.strip()]
                why = next((l for l in reversed(ls_) if any(w in l for w in ('Error', 'assert', 'FAIL', 'REJECTED', 'rejected'))), ls_[-1] if ls_ else '')
            log(f'pipeline {k} end: {"clip" if d["clip"] else "FAILED at " + str(d.get("failed_stage"))}, gate {d.get("gate")}, lift {d.get("lift")}, tolerance {d.get("tolerance")}'
                + (f' | {why[:180]}' if why else ''))
        ks = list(range(len(sessions))); lift = None
        if a.lift == 'auto':
            done_lifts = [status(k).get('lift') for k in ks if status(k).get('lift') is not None]
            if done_lifts: lift = done_lifts[0]
            else: run(0); lift = status(0).get('lift'); ks = ks[1:]
            log(f'lift for the task: {lift} (demo 0)')
        elif a.lift != 'free': lift = float(a.lift)
        extra = f'--lift {lift}' if lift is not None else ''
        with ThreadPoolExecutor(max_workers=a.jobs) as ex: list(ex.map(lambda k: run(k, extra), ks))
        table()
    if 'augment' in steps and a.aug_total > 0:
        save(); good = good_demos()
        if not good: sys.exit('no demo passed: see the table / logs')
        n_each = -(-a.aug_total // len(good)); log(f'augment: {n_each} planned episodes for each of {len(good)} demos ({a.aug_workers} generator workers)')
        for e in good:
            lf = f'{R}/logs/augment_{e.split("/")[1]}.log'
            r = subprocess.run(f'{PY} {X}/pipeline/real_augment.py --ep {e} --n {n_each} --workers {a.aug_workers} > {lf} 2>&1', shell=True)
            aj = f'{R}/augment/{e.split("/")[1]}/augment.json'
            log(f'{e}: augment ->', json.load(open(aj))['views'] if os.path.exists(aj) and r.returncode == 0 else f'FAILED (see {lf})')
        save()
    if 'dataset' in steps:
        save(); good = good_demos()
        low = [d['ep'] for d in B['demos'].values() if d['clip'] and (d.get('tolerance') or 0) < a.min_tolerance]
        if low: log(f'left out (placement tolerance < {a.min_tolerance}): {low}')
        if not good: sys.exit('no demo passed: see the table / logs')
        lifts = sorted({round(B['demos'][e.split('/')[1]]['lift'], 3) for e in good})
        if len(lifts) > 1: log(f'WARNING: the demos plan different lifts {lifts}; the robot runs at one (use --lift auto or a number)')
        r = subprocess.run([LP, f'{X}/pipeline/real_to_lerobot.py', '--eps', ','.join(good), '--root', f'{R}/lerobot_v5', '--repo_id', f'egoh2w/real_{a.task}_v5',
                            '--variants', str(a.variants)], capture_output=True, text=True)
        log('dataset:', ' | '.join(l for l in r.stdout.splitlines() if 'stance variants' in l or 'episodes' in l or 'WARNING' in l)[:800] or r.stderr[-500:])
        B['dataset'] = dict(root=f'{R}/lerobot_v5', demos=good, lifts=lifts)
        aug = [e for e in good if isinstance(B['demos'][e.split('/')[1]].get('augment'), int) and B['demos'][e.split('/')[1]]['augment'] > 0]
        if aug:                                                         # the training set: the augmented episodes (sim recipe)
            dm = ','.join(f'{R}/augment/{e.split("/")[1]}/demos' for e in aug); vw = ','.join(f'{R}/augment/{e.split("/")[1]}/views' for e in aug)
            r = subprocess.run([LP, f'{X}/policy/to_lerobot_repr.py', '--demos', dm, '--wrist_from', vw, '--root', f'{R}/lerobot_aug_floor',
                                '--repo_id', 'head2wrist/lerobot_aug_floor', '--state', 'eef', '--scene', aug[0].replace('/', '_'), '--task', a.description],
                               capture_output=True, text=True)
            n_ep = json.load(open(f'{R}/lerobot_aug_floor/meta/info.json'))['total_episodes'] if os.path.exists(f'{R}/lerobot_aug_floor/meta/info.json') else 0
            log(f'training set (augmented): {n_ep} episodes from {len(aug)} demos' if n_ep else f'augmented dataset FAILED: {r.stderr[-500:]}')
            if n_ep: B['dataset']['train_root'] = f'{R}/lerobot_aug_floor'; B['dataset']['train_episodes'] = n_ep
        json.dump(B, open(BJ, 'w'), indent=1)
    if 'train' in steps:
        ds = f'{R}/lerobot_aug_floor'
        assert os.path.exists(f'{ds}/meta/info.json'), 'build the floor dataset first (--steps augment,dataset)'
        log(f'training on {ds}')
        run_name = f'real_{a.task}_floor'; lp = f'{OUT}/sim_policy/train/{run_name}.log'; rd = f'{OUT}/sim_policy/train/{run_name}'
        os.makedirs(f'{OUT}/sim_policy/train', exist_ok=True)
        if os.path.exists(rd):                                 # LeRobot refuses an existing output dir: keep the old run aside
            old = f'{rd}_prev_{time.strftime("%H%M%S")}'; os.rename(rd, old); log(f'previous run moved to {old}')
        subprocess.Popen(f'AUG=strong REPR=eef_hybrid STEPS={a.train_steps} nohup {REPO}/scripts/train_repr.sh {ds} {run_name} > {lp} 2>&1 &', shell=True)
        B['train'] = dict(run=run_name, log=lp, checkpoint=f'{OUT}/sim_policy/train/{run_name}/checkpoints/{a.train_steps:06d}/pretrained_model', started=time.strftime('%Y-%m-%d %H:%M'))
        json.dump(B, open(BJ, 'w'), indent=1); log(f'training {run_name} started (~2 h for 80k steps): tail -f {lp}')
    log(f'-> {BJ}')


if __name__ == '__main__':
    main()
