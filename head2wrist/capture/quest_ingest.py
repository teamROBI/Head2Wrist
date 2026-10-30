from head2wrist.config import H2W_DATA_ROOT
#!/usr/bin/env python3
"""
Pull EgoKit recordings from a Meta Quest 3 and summarize them (the first step on a recording day).
  quest_ingest.py [--dest $H2W_DATA_ROOT/data/quest_raw] [--list] [--only <name,...>]
  1. adb (PATH or $H2W_DATA_ROOT/tools/platform-tools/adb): the headset must be in developer mode, connected,
     USB debugging allowed inside it; "no permissions" -> install experiments/capture/51-quest.rules (sudo, once)
  2. sessions = folders under /sdcard/DCIM/Recordings (EgoKit 1.0.0) or /sdcard/Android/data/org.chuange.egokit holding
     a poses.txt; each new one is pulled to
     <dest>/<session folder name>/ (already pulled ones are skipped)
  3. per session: duration, pose rate, video size / fps / frames, hand tracking (fraction of pose samples with each hand),
     the files present -> <dest>/sessions.json and a table
The whole folder is pulled: if EgoKit logs the camera intrinsics / extrinsics somewhere, they come along (look in log.txt).
"""
import argparse, json, os, shutil, subprocess, sys
import numpy as np

PKG_DIR = '/sdcard/DCIM/Recordings /sdcard/Android/data/org.chuange.egokit'  # EgoKit 1.0.0 writes to DCIM/Recordings


def summary(d):
    s = dict(name=os.path.basename(d), path=d, files=sorted(os.listdir(d)))
    try:
        rows = np.loadtxt(os.path.join(d, 'poses.txt'), comments='#', ndmin=2); t = rows[:, 1]
        s.update(pose_samples=len(rows), duration_s=float((t[-1] - t[0])/1000), pose_hz=float((len(t) - 1)/max(1e-6, (t[-1] - t[0])/1000)), columns=int(rows.shape[1]))
        for k, side in ((9, 'left'), (9 + 26*7, 'right')):
            J = rows[:, k:k + 26*7].reshape(len(rows), 26, 7)     # untracked = zero positions (the quaternion stays identity)
            s[f'{side}_hand_tracked'] = float(np.mean(np.abs(J[:, :, :3]).sum((1, 2)) > 0))
    except Exception as e: s['poses_error'] = str(e)
    vid = os.path.join(d, 'internal.mp4')
    if os.path.exists(vid):
        r = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,r_frame_rate,nb_frames,codec_name', '-of', 'json', vid], capture_output=True, text=True)
        try: v = json.loads(r.stdout)['streams'][0]; n, dd = v['r_frame_rate'].split('/'); s.update(video=f"{v['width']}x{v['height']}", video_fps=float(n)/float(dd), video_frames=int(v.get('nb_frames', 0)), codec=v['codec_name'])
        except Exception as e: s['video_error'] = str(e)
    else: s['video_error'] = 'no internal.mp4'
    return s


def main():
    global ADB, S, _, a, ap, dev, dst, f, found, keep, l, name, old, p, r, run, s, want, x
    ap = argparse.ArgumentParser()
    ap.add_argument('--dest', default=f'{H2W_DATA_ROOT}/data/quest_raw')
    ap.add_argument('--list', action='store_true', help='only list the sessions on the headset')
    ap.add_argument('--only', default=None)
    a = ap.parse_args()
    ADB = shutil.which('adb') or f'{H2W_DATA_ROOT}/tools/platform-tools/adb'
    if not os.path.exists(ADB): sys.exit(f'adb not found: put platform-tools under {H2W_DATA_ROOT}/tools/ (see RUNBOOK_real_robot.md)')
    run = lambda *c: subprocess.run([ADB, *c], capture_output=True, text=True)
    dev = [l.split('\t') for l in run('devices').stdout.splitlines()[1:] if l.strip()]
    if not dev: sys.exit('no device: connect the Quest by USB, developer mode on, accept "Allow USB debugging" inside the headset')
    if any(s != 'device' for _, s in dev):
        sys.exit(f'device state {dev}: "no permissions" -> sudo cp experiments/capture/51-quest.rules /etc/udev/rules.d/ && sudo udevadm control '
                 '--reload-rules && sudo udevadm trigger, replug; "unauthorized" -> accept the prompt inside the headset')
    found = [os.path.dirname(p) for p in run('shell', f'find {PKG_DIR} -name poses.txt 2>/dev/null').stdout.split()]
    if not found: sys.exit(f'no EgoKit sessions under {PKG_DIR} (record one first; or the app keeps them elsewhere: adb shell find /sdcard -name poses.txt)')
    print(f'{len(found)} sessions on the headset:')
    [print('  ', f) for f in found]
    if a.list: sys.exit(0)
    os.makedirs(a.dest, exist_ok=True)
    want = set(a.only.split(',')) if a.only else None
    S = []
    for f in found:
        name = os.path.basename(f.rstrip('/'))
        if want and name not in want: continue
        dst = os.path.join(a.dest, name)
        if os.path.exists(os.path.join(dst, 'poses.txt')): print(f'{name}: already pulled')
        else:
            print(f'{name}: pulling ...', flush=True); r = run('pull', f, a.dest)
            if r.returncode: print(f'  pull failed: {r.stderr.strip()[-300:]}'); continue
        S.append(summary(dst))
    old = json.load(open(f'{a.dest}/sessions.json')) if os.path.exists(f'{a.dest}/sessions.json') else []
    keep = {x['name']: x for x in old}
    keep.update({x['name']: x for x in S})
    json.dump(list(keep.values()), open(f'{a.dest}/sessions.json', 'w'), indent=1)
    print(f'\n{"session":32s} {"dur s":>6s} {"pose Hz":>7s} {"video":>10s} {"fps":>5s} {"frames":>6s} {"L hand":>6s} {"R hand":>6s}')
    for s in S:
        print(f'{s["name"][:32]:32s} {s.get("duration_s", 0):6.1f} {s.get("pose_hz", 0):7.1f} {s.get("video", "-"):>10s} {s.get("video_fps", 0):5.1f} {s.get("video_frames", 0):6d} '
              f'{s.get("left_hand_tracked", 0):6.2f} {s.get("right_hand_tracked", 0):6.2f}  {s.get("poses_error", "") or s.get("video_error", "")}')
    print(f'-> {a.dest}/sessions.json; files in the first session: {S[0]["files"] if S else "-"}')


if __name__ == '__main__':
    main()
