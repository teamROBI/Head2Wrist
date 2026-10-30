from head2wrist.config import H2W_OUTPUT_ROOT
#!/usr/bin/env python3
"""
The real-robot trial log, one row per attempt, so the paper's real-robot numbers come from a recorded table:
  trial_log.py add --task <task> --k <demo> --mode <replay|execute> --dump <dir> [--outcome success|fail] [--phase ...] [--notes ...]
      (asks for the missing fields interactively: outcome, failure phase, object start)
  trial_log.py summary [--date YYYYMMDD]
-> $H2W_OUTPUT_ROOT/robot_day/<date>/trials.jsonl (append-only)
Failure phases: approach (never reached the object), grasp (missed / pushed it), lift (dropped while lifting),
transport (dropped while carrying), place (released off target), release (did not let go), other.
"""
import argparse, json, os, sys, time
PHASES = ['approach', 'grasp', 'lift', 'transport', 'place', 'release', 'other']




def main():
    global A, D, F, R, S, a, ap, ask, fails, l, mode, outcome, phase, r, row, rows, s, start, sub, task
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    A = sub.add_parser('add')
    A.add_argument('--task', required=True)
    A.add_argument('--k', type=int, default=0)
    A.add_argument('--mode', required=True, choices=['replay', 'execute'])
    A.add_argument('--dump', default='')
    A.add_argument('--outcome', choices=['success', 'fail'], default=None)
    A.add_argument('--phase', choices=PHASES + ['none'], default=None)
    A.add_argument('--start', default=None, help='object start, e.g. "as demo" or "+3 cm left"')
    A.add_argument('--notes', default='')
    S = sub.add_parser('summary')
    S.add_argument('--date', default=time.strftime('%Y%m%d'))
    a = ap.parse_args()
    D = os.environ.get('OUT_DIR') or f'{H2W_OUTPUT_ROOT}/robot_day/{getattr(a, "date", None) or time.strftime("%Y%m%d")}'
    os.makedirs(D, exist_ok=True)
    F = f'{D}/trials.jsonl'
    ask = lambda q, ok=None: next(x for x in iter(lambda: input(q).strip(), None) if (ok is None or x in ok))
    if a.cmd == 'add':
        outcome = a.outcome or ask('outcome [success/fail]: ', ('success', 'fail'))
        phase = a.phase or ('none' if outcome == 'success' else ask(f'failure phase {PHASES}: ', PHASES))
        start = a.start if a.start is not None else (input('object start (enter = as in the demo): ').strip() or 'as demo')
        row = dict(time=time.strftime('%H:%M:%S'), task=a.task, k=a.k, mode=a.mode, outcome=outcome, failure_phase=phase, object_start=start, notes=a.notes, dump=a.dump)
        open(F, 'a').write(json.dumps(row) + '\n'); print('logged:', row)
    else:
        rows = [json.loads(l) for l in open(F)] if os.path.exists(F) else []
        if not rows: sys.exit(f'no trials in {F}')
        for mode in sorted({r['mode'] for r in rows}):
            for task in sorted({r['task'] for r in rows if r['mode'] == mode}):
                R = [r for r in rows if r['mode'] == mode and r['task'] == task]; s = sum(r['outcome'] == 'success' for r in R)
                fails = {}; [fails.__setitem__(r['failure_phase'], fails.get(r['failure_phase'], 0) + 1) for r in R if r['outcome'] == 'fail']
                print(f'{task:20s} {mode:8s} {s}/{len(R)} success   failures by phase: {fails}')


if __name__ == '__main__':
    main()
