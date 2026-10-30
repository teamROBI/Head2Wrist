from head2wrist.config import H2W_OUTPUT_ROOT, H2W_PYTHON
#!/usr/bin/env python3
"""
Sim gate with grasp search: for each episode, grasp variants are tried in order - re-plan the grasp, transport and arm
(the base / lift setting kept: stance is not re-searched), export the Isaac Sim twin, replay the planned motion with
physics - and the first variant that picks and places the object is kept. None passes -> the episode is rejected.
  pass     object lifted >= 3 cm, released within 3 cm of the planned place point, no other object moved > NUDGE_CM by
           the robot (a target container nudged 1-4 cm is flagged, not failed)
  variants A  grasp from the object mesh, closed fingertips at the object's far end, closed-finger clearance 4 mm, tilt
              <= 20 deg on thin objects (planner default)
           B  A, 1 cm shallower
           C  the original planner: human pinch point snapped onto the object footprint, open-finger clearance 3 mm
           D  mesh-centred across the jaws, at the human pinch depth, open-finger clearance 3 mm
Writes <sim_twin>/<name>/gate.json (every attempt) and, on a pass, gate_pass.json (the run_episode 'simgate' stage
output: a rejected episode stops there, before any video generation); exits 3 when an episode is rejected. Usage: sim_gate.py --eps a/1,b/2 [--variants B,C,D]
"""
import argparse, json, os, shutil, subprocess
X = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); REPO = os.path.dirname(X); PY = H2W_PYTHON
VARIANTS = dict(A=dict(name='A'), B=dict(name='B', dz=-0.01),
                C=dict(name='C', mesh=False, clear='open', clearance=0.003, tilt_cap=None),
                D=dict(name='D', depth='human', clear='open', clearance=0.003, tilt_cap=None))
NUDGE_CM = 4.0


def verdict(r):
    robot_moves = [m for k, m in r['static_moved_cm'].items() if r['static_moved_why'].get(k, {}).get('robot')]
    worst = max(robot_moves, default=0.0)
    ok = r['lifted'] and r['placed'] and worst <= NUDGE_CM
    return ok, ('clean' if worst <= 1.0 else 'nudged') if ok else 'fail', worst


def main():
    global D, I, a, ap, ep, kind, n, ok, passed, r, rejected, rp, rr, t, tries, v, worst
    ap = argparse.ArgumentParser()
    ap.add_argument('--eps', required=True)
    ap.add_argument('--variants', default='A,B,C,D')
    a = ap.parse_args()
    for ep in a.eps.split(','):
        n = ep.replace('/', '_'); I = f'{H2W_OUTPUT_ROOT}/lyra2/explore_inputs/{n}'; D = f'{H2W_OUTPUT_ROOT}/sim_twin/{n}'
        tries = []; os.makedirs(D, exist_ok=True)
        for v in a.variants.split(','):
            json.dump(VARIANTS[v], open(f'{I}/grasp_variant.json', 'w'))
            r = subprocess.run([PY, f'{X}/pipeline/run_episode.py', '--ep', ep, '--to', 'arm', '--redo', 'approach,transport,task,table,arm'],
                               capture_output=True, text=True)
            if r.returncode:
                tries.append(dict(variant=v, result='plan failed', log=r.stdout[-400:])); print(ep, v, 'plan failed', flush=True); continue
            subprocess.run(['bash', f'{REPO}/scripts/replay_batch.sh', ep], capture_output=True, text=True)
            rp = f'{D}/isaac/replay.json'
            if not os.path.exists(rp) or os.path.getmtime(rp) < os.path.getmtime(f'{D}/twin.json'):
                tries.append(dict(variant=v, result='replay failed')); print(ep, v, 'replay failed', flush=True); continue
            rr = json.load(open(rp)); shutil.copy(rp, f'{D}/isaac/replay_{v}.json')
            ok, kind, worst = verdict(rr)
            tries.append(dict(variant=v, result=kind, lift_cm=rr['lift_cm'], place_err_cm=rr['place_err_xy_cm'], robot_moved_cm=worst))
            print(f"{ep} {v}: {kind} (lift {rr['lift_cm']:.1f} cm, place err {rr['place_err_xy_cm']:.1f} cm, robot moved others {worst:.1f} cm)", flush=True)
            if ok: break
        passed = next((t for t in tries if t['result'] in ('clean', 'nudged')), None)
        json.dump(dict(ep=ep, passed=passed is not None, variant=passed and passed['variant'], tries=tries), open(f'{D}/gate.json', 'w'), indent=1)
        print(f'GATE {ep}: ' + (f"PASS with {passed['variant']} ({passed['result']})" if passed else 'REJECTED'), flush=True)
        if passed: json.dump(passed, open(f'{D}/gate_pass.json', 'w'), indent=1)
        else: rejected = True
    raise SystemExit(3 if 'rejected' in globals() else 0)


if __name__ == '__main__':
    main()
