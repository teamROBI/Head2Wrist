#!/usr/bin/env python3
"""
Head2Wrist policy on the real AI Worker (FFW-SG2): a ROS 2 node that feeds the active arm's wrist D405 image and joint state to
the policy server (head2wrist/policy/policy_server.py --repr eef_hybrid --lift <lift> --side <side>, on the GPU host,
LeRobot venv) and sends the joint targets it answers to the arm, the way ROBOTIS's own inference does (physical_ai_server,
ffw_sg2_rev1_config.yaml; the follower's arm_<s>_controller is a JointTrajectoryController remapped to the leader topic):
  state     /joint_states                                                 arm_<s>_joint1..7, gripper_<s>_joint1, lift_joint
  image     /camera_<side>/camera_<side>/color/image_rect_raw/compressed  CompressedImage (JPEG, BGR) as ROBOTIS records it;
                                                                          a topic without /compressed is read as sensor_msgs/Image.
                                                                          424x240 RGB to the policy (the D405's raw orientation,
                                                                          as the training views)
  command   /leader/joint_trajectory_command_broadcaster_<side>/joint_trajectory   one JointTrajectoryPoint, 8 joints
  lift      /leader/joystick_controller_right/joint_trajectory           lift_joint (the lift_controller)
Every topic can be overridden (--state_topic, --image_topic, --cmd_topic, --lift_topic: the isolated test against
fake_sg2.py uses its own namespace and ROS domain).
Modes:
  shadow    (default) the policy runs, nothing is sent; per step the target vs the current joints is logged
  execute   targets are sent, each limited to --max_step rad per tick (arm) and --max_grip per tick (gripper), arm joints
            within --joint_limit, the gripper within [0, --grip_max]; no new state or image for --stale_s: hold (send
            nothing; the controller keeps the last target); after --max_s: stop
            --set_lift: first command the lift there and wait until it is within 5 mm; --start_q: then ramp the arm (at
            --max_step per tick) to these 8 joints (the plan's start pose), reset the policy and start it
  snap      one full-resolution head camera frame (--snap_topic) + its camera_info + the joints -> <--dump>/snap.npz
            (robot/placement_view.py: the expected view at the planned stance over the live one)
  record    no policy: both wrist cameras + the joint states for --record_s seconds -> <--dump>/record.npz (the wrist
            camera <-> arm check, robot/check_wrist_mapping.py; about half of the recorded SG2 datasets have the wrist
            streams swapped)
  replay_plan  the PLANNED joint path itself, no policy (--plan npz): lift, ramp to its first pose, then each pose in turn at
            --max_step per tick, held until the arm reaches it; the real wrist frame at every pose -> <--dump>/plan_record.npz.
            Measures the generated views against the real wrist camera (robot/compare_views.py) and the plan's open-loop success.
  --dump <dir>  shadow / execute: the wrist image twice a second (what the policy sees, BGR jpg) and, at the end,
            log.npz: per tick the time, joints, policy target, sent command
  --horizon_s   time_from_start of each command (0.1 s: the controller interpolates over a bit more than one tick;
            0 = ROBOTIS's inference, a step to the target)
Without ROS (--replay <LeRobot dataset dir> --episode k) the same loop runs on a recorded SG2 episode, for an offline test.
Runs inside the jazzy ai_worker container (rmw_zenoh; robot/robot_day.sh sets it up as a zenoh client of the robot).
"""
import argparse, json, os, sys, time
from multiprocessing.connection import Client
import numpy as np, cv2

ARM = lambda s: [f'arm_{s[0]}_joint{i}' for i in range(1, 8)] + [f'gripper_{s[0]}_joint1']


class Policy:
    def __init__(self, addr):
        h, p = addr.split(':'); self.c = Client((h, int(p)), authkey=b'egoh2w')

    def reset(self): self.c.send({'reset': True}); return self.c.recv()

    def act(self, state, rgb):
        rgb = np.ascontiguousarray(rgb, np.uint8); self.c.send({'state': [float(x) for x in state], 'wrist': (rgb.tobytes(), rgb.shape)})
        return np.array(self.c.recv()['action'], float)


class Safety:
    def __init__(self, a): self.a = a

    def limit(self, cur, tgt):
        tgt = np.array(tgt, float)
        if self.a.grip_close is not None and tgt[7] > self.a.grip_thresh:   # a 'close' command closes firmly (see --grip_close)
            tgt[7] = max(tgt[7], self.a.grip_close)
        # the arm step is scaled as a whole (fastest joint at --max_step), never clipped per joint: per-joint clipping bends
        # the motion away from the commanded direction
        d = tgt - cur; d[:7] *= min(1.0, self.a.max_step/max(np.abs(d[:7]).max(), 1e-9)); d[7] = np.clip(d[7], -self.a.max_grip, self.a.max_grip)
        out = cur + d; out[:7] = np.clip(out[:7], -self.a.joint_limit, self.a.joint_limit)
        out[7] = np.clip(out[7], 0.0, self.a.grip_max); return out


def prep(img, enc='rgb8'):
    """decoded camera image -> 424x240 RGB"""
    if enc in ('bgr8', 'bgra8'): img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)
    return cv2.resize(img[..., :3], (424, 240), interpolation=cv2.INTER_AREA) if img.shape[:2] != (240, 424) else img[..., :3]


def prep_full(img, enc):
    """a raw ROS image at its own resolution -> RGB"""
    return cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB) if enc in ('bgr8', 'bgra8') else img[..., :3].copy()


def decode(m, compressed):
    """ROS image message -> 424x240 RGB"""
    if compressed:
        bgr = cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None: raise ValueError('undecodable CompressedImage')
        return prep(bgr, 'bgr8')
    return prep(np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1), m.encoding)


def replay(a):
    """offline: the recorded episode's states + wrist frames through the policy, shadow mode"""
    import pandas as pd
    info = json.load(open(f'{a.replay}/meta/info.json')); names = info['features']['observation.state']['names']; idx = [names.index(n) for n in ARM(a.side)]
    df = pd.read_parquet(f'{a.replay}/data/chunk-000/episode_{a.episode:06d}.parquet'); S = np.stack(df['observation.state'].values)[:, idx]
    cap = cv2.VideoCapture(f'{a.replay}/videos/chunk-000/observation.images.cam_wrist_{a.side}/episode_{a.episode:06d}.mp4')
    pol = Policy(a.policy); pol.reset(); step = max(1, int(round(info['fps']/a.hz))); dts, D = [], []
    for k in range(min(len(S), int(a.max_s*info['fps']))):
        ok, fr = cap.read()
        if not ok: break
        if k % step: continue
        t0 = time.time(); tgt = pol.act(S[k], prep(fr, 'bgr8')); dts.append(time.time() - t0); D.append(tgt - S[k])
    D = np.array(D); print(f'{len(D)} steps, policy latency median {1000*np.median(dts):.0f} ms (max {1000*np.max(dts):.0f}); |target - state| arm median {np.median(np.abs(D[:, :7])):.3f} rad, max {np.abs(D[:, :7]).max():.3f}; gripper target range [{(S[:len(D), 7] + D[:, 7]).min():.2f}, {(S[:len(D), 7] + D[:, 7]).max():.2f}], recorded gripper [{S[:, 7].min():.2f}, {S[:, 7].max():.2f}]')


def ros(a):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState, Image, CompressedImage, CameraInfo
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from builtin_interfaces.msg import Duration

    def traj(names, pos, secs):
        m = JointTrajectory(); m.joint_names = list(names); pt = JointTrajectoryPoint(); pt.positions = [float(x) for x in pos]
        ns = int(1e9*secs); pt.time_from_start = Duration(sec=ns//10**9, nanosec=ns % 10**9); m.points = [pt]; return m

    def image_sub(node, topic, cb):
        comp = topic.endswith('/compressed')
        return node.create_subscription(CompressedImage if comp else Image, topic, lambda m: cb(m, comp), qos_profile_sensor_data)

    class Bridge(Node):
        def __init__(self):
            super().__init__('egoh2w_bridge'); self.names = ARM(a.side); self.q = self.img = self.lift = None; self.tq = self.ti = 0.0
            self.n = 0; self.log = []; self.t_start = None; self.t_lift = 0.0; self.bad_img = 0; self.qs = {}; self.lift_dur = 0.0
            self.dwell_t0 = None; self.dwell_q = None; self.was_closing = False; self.ghist = []   # grasp dwell (--grasp_dwell)
            self.held = None; self.t_held = 0.0; self.nq = 0; self.t_jump = None                       # --step_gate
            self.lift_checked = self.ramp_checked = self.aborted = False                               # motion safety checks
            self.create_subscription(JointState, a.state_topic, self.on_js, 10)
            if a.mode == 'snap':                                        # one full-resolution frame of --snap_topic + its camera_info + the joints
                self.snap = {}; comp = a.snap_topic.endswith('/compressed')
                def on_snap(m, comp=comp):
                    self.snap['image'] = cv2.cvtColor(cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) if comp else\
                        prep_full(np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1), m.encoding)
                self.create_subscription(CompressedImage if comp else Image, a.snap_topic, on_snap, qos_profile_sensor_data)
                ci = a.snap_topic.split('/image')[0] + '/camera_info'
                self.create_subscription(CameraInfo, ci, lambda m: self.snap.update(K=np.array(m.k).reshape(3, 3), size=(m.width, m.height)), qos_profile_sensor_data)
                self.create_timer(0.2, self.snap_tick); self.t_start = time.time(); return
            if a.mode == 'record':
                self.rec = {'t': [], 'q_left': [], 'q_right': [], 'lift': [], 'left': [], 'right': []}; self.cams = {}
                self.info = {}
                for s in ('left', 'right'):
                    image_sub(self, a.record_topic.format(side=s), lambda m, comp, s=s: self.cams.__setitem__(s, decode(m, comp)))
                    it = a.record_topic.format(side=s).split('/image_rect_raw')[0] + '/camera_info'     # the colour stream's intrinsics
                    self.create_subscription(CameraInfo, it, lambda m, s=s: self.info.__setitem__(s, dict(K=list(m.k), D=list(m.d), width=m.width, height=m.height, model=m.distortion_model)), qos_profile_sensor_data)
                self.create_timer(1.0/a.hz, self.rec_tick); self.get_logger().info(f'record {a.record_s} s of both wrist cameras -> {a.dump}/record.npz'); return
            image_sub(self, a.image_topic, self.on_img)
            self.pub = self.create_publisher(JointTrajectory, a.cmd_topic, 10); self.lpub = self.create_publisher(JointTrajectory, a.lift_topic, 10)
            self.safe = Safety(a); self.pol = None if a.mode in ('replay_plan', 'goto', 'preset') else Policy(a.policy)
            if a.mode == 'preset':                                       # both arms + head + lift to the init preset (no keyboard GUI)
                self.pubs = {s_: self.create_publisher(JointTrajectory, a.cmd_topic_fmt.format(side=s_), 10) for s_ in ('left', 'right')}
                self.hpub = self.create_publisher(JointTrajectory, a.head_topic, 10); self.arms_checked = False
            if a.mode == 'replay_plan':                                  # the planned joint path itself, no policy
                P = np.load(a.plan); self.plan = P['q'].astype(float); self.plan_frames = P['frames']; self.k = 0; self.t_k = None; self.prec = []
                a.start_q = ','.join(str(x) for x in self.plan[0])
            self.start = np.array([float(x) for x in a.start_q.split(',')]) if a.start_q else None
            moving = a.mode in ('execute', 'replay_plan', 'goto')            # goto: --set_lift, then ramp to --start_q, then exit (no policy)
            self.stage = 'arms' if a.mode == 'preset' else 'lift' if (moving and a.set_lift is not None) else 'ramp' if (moving and self.start is not None) else 'run'
            if self.stage == 'run' and self.pol: self.pol.reset()
            if a.dump: os.makedirs(a.dump, exist_ok=True)
            self.create_timer(1.0/a.hz, self.tick)
            self.get_logger().info(f'mode {a.mode}, side {a.side}, {a.hz} Hz, policy {a.policy}; state {a.state_topic}, image {a.image_topic}, command {a.cmd_topic}; first stage: {self.stage}')

        def on_js(self, m):
            self.js_all = (list(m.name), [float(x) for x in m.position])
            for s in ('left', 'right'):
                nm = ARM(s)
                if all(n in m.name for n in nm): self.qs[s] = np.array([m.position[m.name.index(n)] for n in nm])
            if a.side in self.qs and all(n in m.name for n in self.names): self.q = self.qs[a.side]; self.tq = time.time()
            if 'lift_joint' in m.name: self.lift = float(m.position[m.name.index('lift_joint')])

        def on_img(self, m, comp):
            try: self.img = decode(m, comp); self.ti = time.time()
            except ValueError: self.bad_img += 1

        def snap_tick(self):
            js = getattr(self, 'js_all', None)
            if 'image' in self.snap and 'K' in self.snap and js is not None:
                os.makedirs(a.dump, exist_ok=True)
                np.savez_compressed(f'{a.dump}/snap.npz', image=self.snap['image'], K=self.snap['K'], size=np.array(self.snap['size']), joint_names=np.array(js[0]),
                                    joint_pos=np.array(js[1]), topic=a.snap_topic, t=time.time())
                self.get_logger().info(f'snap {self.snap["image"].shape} -> {a.dump}/snap.npz'); rclpy.shutdown()
            elif time.time() - self.t_start > 10: self.get_logger().warn(f'waiting: image {"image" in self.snap}, camera_info {"K" in self.snap}, joints {js is not None}'); self.t_start = time.time()

        def rec_tick(self):
            now = time.time(); self.t_start = self.t_start or now
            if len(self.cams) < 2 or len(self.qs) < 2:
                if now - self.t_start > 5: self.get_logger().warn(f'waiting: cameras {sorted(self.cams)}, arms {sorted(self.qs)}'); self.t_start = now
                return
            for s in ('left', 'right'): self.rec[s].append(self.cams[s]); self.rec[f'q_{s}'].append(self.qs[s])
            self.rec['t'].append(now); self.rec['lift'].append(np.nan if self.lift is None else self.lift)
            if len(self.rec['t']) >= a.record_s*a.hz:
                os.makedirs(a.dump, exist_ok=True)
                np.savez_compressed(f'{a.dump}/record.npz', **{k: np.array(v) for k, v in self.rec.items()}, left_topic=a.record_topic.format(side='left'), right_topic=a.record_topic.format(side='right'),
                                    camera_info=json.dumps(self.info))
                self.get_logger().info(f'{len(self.rec["t"])} frames -> {a.dump}/record.npz'); rclpy.shutdown()

        def tick(self):
            now = time.time()
            if self.q is None or self.img is None:                      # say what is missing instead of waiting silently
                self.t_wait = getattr(self, 't_wait', now)
                if now - self.t_wait > 5.0:
                    self.get_logger().warn(f'waiting for {"joint states (" + a.state_topic + ")" if self.q is None else ""} {"image (" + a.image_topic + ")" if self.img is None else ""}'); self.t_wait = now
                return
            if now - self.tq > a.stale_s or now - self.ti > a.stale_s: self.get_logger().warn('stale state / image: holding', throttle_duration_sec=1.0); return
            if self.stage == 'arms':                                   # preset: BOTH arms (and the head) first, the lift untouched
                tgt = {'left': np.array(a.safe_left[:7], float), 'right': np.array(a.safe_right[:7], float)}
                if 'left' not in self.qs or 'right' not in self.qs: return
                if not self.arms_checked:
                    far = max(float(np.abs(self.qs[s_][:7] - tgt[s_]).max()) for s_ in tgt)
                    if far > a.pose_tol and self.lift is not None and self.lift < a.arms_lift_min:
                        self.get_logger().error(f'NOT moving the arms: {far:.2f} rad from the preset while the lift is low ({self.lift:.3f} < '
                                                f'{a.arms_lift_min}): they could hit the table; raise the lift first'); self.aborted = True; rclpy.shutdown(); return
                    self.arms_checked = True
                    self.get_logger().info(f'preset: both arms to the preset (up to {far:.2f} rad away); the lift stays at {self.lift:.3f} until they are there')
                    if a.head is not None: self.hpub.publish(traj(['head_joint1', 'head_joint2'], a.head, 3.0))
                done = True
                for s_ in tgt:
                    q_ = self.qs[s_]; self.pubs[s_].publish(traj(ARM(s_), self.safe.limit(q_, np.r_[tgt[s_], q_[7]]), a.horizon_s))   # the gripper keeps its value
                    done = done and float(np.abs(q_[:7] - tgt[s_]).max()) < 0.02
                if done: self.stage = 'lift'; self.get_logger().info('preset: both arms there')
                return
            if self.stage == 'lift':                                   # lift first, wait until it is there
                if self.lift is not None and abs(self.lift - a.set_lift) < 0.005:
                    self.stage = 'ramp' if self.start is not None else 'run'; self.get_logger().info(f'lift at {self.lift:.3f}')
                    if self.stage == 'run' and self.pol: self.pol.reset()
                elif self.lift is not None and not self.lift_checked:
                    ok, why = self.lift_safe()
                    if not ok:
                        self.get_logger().error(f'NOT moving the lift ({self.lift:.3f} -> {a.set_lift:.3f}): {why}'); self.aborted = True; rclpy.shutdown(); return
                    self.lift_checked = True
                elif self.lift is not None and (self.t_lift == 0.0 or now - self.t_lift > self.lift_dur + 2.0):
                    # one goal at --lift_speed; re-sent only if it has not arrived 2 s after its time (a re-sent goal restarts the
                    # controller's interpolation, so sending it every tick would make the lift creep)
                    self.lift_dur = max(1.0, abs(a.set_lift - self.lift)/a.lift_speed)
                    self.lpub.publish(traj(['lift_joint'], [a.set_lift], self.lift_dur)); self.t_lift = now
                    self.get_logger().info(f'lift {self.lift:.3f} -> {a.set_lift:.3f} in {self.lift_dur:.1f} s')
                return
            if self.stage == 'ramp':                                   # ramp to the plan's start pose first
                if not self.ramp_checked:
                    d = float(np.abs(self.q[:7] - self.start[:7]).max())
                    if d > a.ramp_max:
                        self.get_logger().error(f'NOT ramping the arm: it is {d:.2f} rad from the start pose (> --ramp_max {a.ramp_max}); '
                                                'load the init preset in the keyboard GUI first'); self.aborted = True; rclpy.shutdown(); return
                    self.ramp_checked = True
                self.pub.publish(traj(self.names, self.safe.limit(self.q, self.start), a.horizon_s))
                if np.abs(self.q[:7] - self.start[:7]).max() < 0.02 and (a.mode != 'replay_plan' or abs(self.q[7] - self.start[7]) < 0.05):
                    self.stage = 'run'; self.get_logger().info('at the start pose: ' + ('replaying the plan' if a.mode == 'replay_plan' else 'policy on'))
                    if self.pol: self.pol.reset()
                return
            if a.mode in ('goto', 'preset'): self.get_logger().info('at the target pose' if a.mode == 'goto' else 'at the preset'); rclpy.shutdown(); return
            if a.mode == 'replay_plan': return self.replay_tick(now)
            self.t_start = self.t_start or now
            if now - self.t_start > a.max_s: self.get_logger().info('max duration reached: stopping'); rclpy.shutdown(); return
            qp = self.q.copy()
            if a.grip_state_max is not None: qp[7] = min(qp[7], a.grip_state_max)    # the policy sees the gripper within its training range
            # --step_gate: one policy step per REACHED target (as the plan replay, which grasped). The policy was trained on
            # 16 fps plans whose TCP moves ~0.11 m/s; the rate-limited real arm managed ~0.03 m/s, so a per-tick policy ran
            # ahead of the arm (targets 5-11 cm ahead, the 'grasp' passed while the arm was still 4 cm above the cube)
            gate = (self.held is None or (self.dwell_t0 is None and (np.abs(self.q[:7] - self.held[:7]).max() < a.gate_tol or now - self.t_held > a.gate_timeout)))
            if a.step_gate and not gate: tgt = self.held.copy()
            else:
                tgt = self.pol.act(qp, self.img); self.held, self.t_held = tgt.copy(), now; self.nq += 1
                if a.step_gate and np.abs(tgt[:7] - self.q[:7]).max() > a.jump_max:   # one step is ~0.02-0.09 rad: an IK flip / wild output
                    self.get_logger().warn(f'policy target {np.abs(tgt[:7] - self.q[:7]).max():.2f} rad from the arm (> {a.jump_max}): holding', throttle_duration_sec=0.5)
                    self.t_jump = self.t_jump or now; self.held = None; tgt = np.r_[self.q[:7], self.q[7]]
                    if now - self.t_jump > 2.0: self.get_logger().error('the policy keeps jumping: stopping'); rclpy.shutdown(); return
                else: self.t_jump = None
            if self.dwell(now, tgt[7] > a.grip_thresh, self.q[:7]): tgt = np.r_[self.dwell_q, tgt[7]]   # arm holds, gripper closes
            cmd = self.safe.limit(self.q, tgt); self.n += 1
            if self.n % a.hz == 0 or a.mode == 'shadow':
                self.get_logger().info(f'step {self.n} (policy step {self.nq}): |target - q| max {np.abs(tgt[:7] - self.q[:7]).max():.3f} rad, gripper {self.q[7]:.2f} -> {tgt[7]:.2f}')
            if a.dump:
                self.log.append(np.r_[now, self.q, tgt, cmd])
                if self.n % max(1, a.hz//2) == 1: cv2.imwrite(f'{a.dump}/f{self.n:05d}.jpg', cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR))
            if a.mode == 'execute': self.pub.publish(traj(self.names, cmd, a.horizon_s))

    def dwell(self, now, closing, q_hold):
        """grasp dwell: when a close command starts, hold the arm at q_hold until the real gripper has closed (it stalls on the
        object for 0.3 s, or is nearly shut) or --grasp_dwell s pass. The plans / training close in ~0.2-0.4 s while the arm
        waits; the real gripper is slower, so the lift began with the jaws half open (real cube_bowl replay: a top-edge grab).
        -> True while the arm must hold"""
        if a.grasp_dwell <= 0: return False
        self.ghist = [(t_, g_) for t_, g_ in self.ghist if now - t_ < 0.3] + [(now, float(self.q[7]))]
        if closing and not self.was_closing: self.dwell_t0, self.dwell_q = now, np.array(q_hold, float).copy(); self.get_logger().info('grasp: holding the arm while the gripper closes')
        self.was_closing = closing
        if self.dwell_t0 is None: return False
        gs = [g_ for _, g_ in self.ghist]; stalled = self.q[7] > 0.25 and now - self.dwell_t0 > 0.3 and max(gs) - min(gs) < 0.01
        if stalled or self.q[7] > 0.85 or now - self.dwell_t0 > a.grasp_dwell:
            self.get_logger().info(f'grasp: gripper {self.q[7]:.2f} after {now - self.dwell_t0:.1f} s, moving on'); self.dwell_t0 = None; return False
        return True

    def lift_safe(self):
        """A downward lift move requires both arms at their safe presets."""
        dz = a.set_lift - self.lift
        if dz >= -0.005: return True, ''
        if a.safe_left is None or a.safe_right is None: return False, 'a downward move needs --safe_left and --safe_right (the init preset)'
        if 'left' not in self.qs or 'right' not in self.qs: return False, 'both arms\' joint states are not in /joint_states'
        err = {s_: float(np.abs(self.qs[s_][:7] - np.array(v[:7])).max()) for s_, v in (('left', a.safe_left), ('right', a.safe_right))}
        bad = {s_: round(e, 2) for s_, e in err.items() if e > a.pose_tol}
        if bad: return False, f'the arms are not at the init preset (max joint error {bad} rad > {a.pose_tol}): load the preset in the keyboard GUI first'
        return True, ''

    def replay_tick(self, now):
        """step through the planned joints; each pose is held until the ARM reaches it (the gripper closes on the object and
        cannot reach its command) or --k_timeout passes; the wrist frame at each reached pose is recorded"""
        tgt = self.plan[self.k]; self.pub.publish(traj(self.names, self.safe.limit(self.q, tgt), a.horizon_s)); self.t_k = self.t_k or now
        reached = np.abs(self.q[:7] - tgt[:7]).max() < a.track_tol
        if self.dwell(now, tgt[7] > a.grip_thresh, tgt[:7]): return            # the gripper is still closing: stay at this row
        if reached or now - self.t_k > a.k_timeout:
            self.prec.append(dict(k=self.k, frame=int(self.plan_frames[self.k]), q=self.q.copy(), target=tgt.copy(), img=self.img.copy(), t=now, reached=bool(reached)))
            self.k += 1; self.t_k = None
            if self.k % 16 == 0: self.get_logger().info(f'plan frame {self.k}/{len(self.plan)}' + ('' if reached else ' (timed out)'))
            if self.k >= len(self.plan):
                R = self.prec; os.makedirs(a.dump, exist_ok=True)
                np.savez_compressed(f'{a.dump}/plan_record.npz', k=np.array([r['k'] for r in R]), frame=np.array([r['frame'] for r in R]), q=np.array([r['q'] for r in R]),
                                    target=np.array([r['target'] for r in R]), wrist=np.array([r['img'] for r in R]), t=np.array([r['t'] for r in R]),
                                    reached=np.array([r['reached'] for r in R]), image_topic=a.image_topic, plan=a.plan)
                self.get_logger().info(f'plan replayed: {len(R)} frames ({sum(r["reached"] for r in R)} reached) -> {a.dump}/plan_record.npz'); rclpy.shutdown()

    Bridge.replay_tick = replay_tick
    Bridge.dwell = dwell                                          # (defined next to replay_tick, outside the class)
    Bridge.lift_safe = lift_safe
    # rclpy's own SIGINT handler shuts the context down first, so nothing could be published after Ctrl+C: take SIGINT / SIGTERM
    # here instead and HOLD the arm and the lift where they are (a lift goal already sent is otherwise finished by the controller)
    import signal
    from rclpy.signals import SignalHandlerOptions
    def _stop(*_): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _stop)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO); n = Bridge()
    try: rclpy.spin(n)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        if rclpy.ok() and a.mode in ('execute', 'goto', 'replay_plan', 'preset') and n.q is not None:
            for _ in range(5):
                if a.mode == 'preset':
                    for s_, p_ in n.pubs.items():
                        if s_ in n.qs: p_.publish(traj(ARM(s_), n.qs[s_], 0.3))
                else: n.pub.publish(traj(n.names, n.q, 0.3))
                if n.lift is not None: n.lpub.publish(traj(['lift_joint'], [n.lift], 0.3))
                time.sleep(0.05)
            n.get_logger().warn(f'stopped: holding the arm and the lift at {n.lift}')
    finally:
        if a.mode != 'record' and a.dump and n.log:
            np.savez(f'{a.dump}/log.npz', log=np.array(n.log), cols='t, q[8], target[8], command[8]', mode=a.mode, side=a.side); print(f'{len(n.log)} ticks -> {a.dump}/log.npz')
        if rclpy.ok(): rclpy.shutdown()
        if getattr(n, 'aborted', False): sys.exit(2)                 


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--side', choices=['left', 'right'], required=True); ap.add_argument('--policy', default='localhost:6123')
    ap.add_argument('--mode', choices=['shadow', 'execute', 'record', 'snap', 'replay_plan', 'goto', 'preset'], default='shadow'); ap.add_argument('--hz', type=int, default=16)
    ap.add_argument('--max_step', type=float, default=0.03, help='rad per tick (0.48 rad/s at 16 Hz)'); ap.add_argument('--max_grip', type=float, default=0.06)
    ap.add_argument('--grip_close', type=float, default=None, help='a gripper target above --grip_thresh is raised to this (closes firmly): '
                    'the plans and the training data close only to the object\'s measured width (cube_box: 0.55 = 6.3 cm jaws), '
                    'which on the real gripper does not squeeze the object; 0.9 = 2.5 cm jaws, the gripper stops on the object')
    ap.add_argument('--grasp_dwell', type=float, default=0.0, help='max s the arm holds when a close starts, until the gripper has '
                    'closed (stalls on the object, or > 0.85); 0 = off')
    ap.add_argument('--grip_thresh', type=float, default=0.35, help='see --grip_close (training: open ~0.07, closed ~0.55)')
    ap.add_argument('--grip_state_max', type=float, default=None, help='the gripper joint sent to the policy is capped here '
                    '(its training maximum, 0.56 for cube_box): a firmer real grip then looks like the trained one')
    ap.add_argument('--joint_limit', type=float, default=3.1); ap.add_argument('--grip_max', type=float, default=1.1, help='gripper joint range [0, this] (the real gripper closes at ~1.13)')
    ap.add_argument('--stale_s', type=float, default=0.3); ap.add_argument('--max_s', type=float, default=40.0)
    ap.add_argument('--step_gate', action='store_true', help='execute: query the policy only when the arm has reached its last target '
                    '(within --gate_tol rad, or after --gate_timeout s): one policy step per reached pose, like replay_plan')
    ap.add_argument('--safe_left', type=lambda v: [float(x) for x in v.split(',')], default=None, help='init-preset left arm joints (7+): a downward lift move needs both arms there')
    ap.add_argument('--safe_right', type=lambda v: [float(x) for x in v.split(',')], default=None, help='init-preset right arm joints (7+)')
    ap.add_argument('--head', type=lambda v: [float(x) for x in v.split(',')], default=None, help='preset mode: head_joint1,2')
    ap.add_argument('--head_topic', default='/leader/joystick_controller_left/joint_trajectory')
    ap.add_argument('--cmd_topic_fmt', default='/leader/joint_trajectory_command_broadcaster_{side}/joint_trajectory', help='preset mode: both arms')
    ap.add_argument('--arms_lift_min', type=float, default=-0.05, help='preset mode: arms far from the preset move only with the lift above this (m)')
    ap.add_argument('--pose_tol', type=float, default=0.15, help='max joint error (rad) to count as at the preset')
    ap.add_argument('--ramp_max', type=float, default=0.5, help='refuse to ramp the arm to --start_q from farther than this (rad); the return step passes more')
    ap.add_argument('--gate_tol', type=float, default=0.015); ap.add_argument('--gate_timeout', type=float, default=1.5)
    ap.add_argument('--jump_max', type=float, default=0.35, help='--step_gate: a policy target farther than this (rad, any arm joint) is not followed: hold, ask again; 2 s of it: stop')
    ap.add_argument('--horizon_s', type=float, default=0.1, help='time_from_start of each command (0 = a step, as ROBOTIS inference)')
    ap.add_argument('--state_topic', default='/joint_states'); ap.add_argument('--image_topic', default=None)
    ap.add_argument('--cmd_topic', default=None); ap.add_argument('--lift_topic', default='/leader/joystick_controller_right/joint_trajectory')
    ap.add_argument('--record_topic', default='/camera_{side}/camera_{side}/color/image_rect_raw/compressed', help='record mode: the wrist image topic pattern')
    ap.add_argument('--record_s', type=float, default=10.0)
    ap.add_argument('--plan', default=None, help='replay_plan: npz with q (N, 8 planned joints incl. gripper) and frames (plan frame index)')
    ap.add_argument('--track_tol', type=float, default=0.015, help='replay_plan: arm joints within this of the planned pose -> next pose (rad)')
    ap.add_argument('--k_timeout', type=float, default=3.0, help='replay_plan: s per pose before moving on anyway (logged as not reached)')
    ap.add_argument('--snap_topic', default='/zed/zed_node/left/image_rect_color/compressed', help='snap mode: the head camera (placement check)')
    ap.add_argument('--replay', default=None, help='offline: a LeRobot dataset dir'); ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--start_q', default=None, help='8 comma-separated joints: ramp there first (execute mode)'); ap.add_argument('--set_lift', type=float, default=None)
    ap.add_argument('--lift_speed', type=float, default=0.03, help='m/s for --set_lift')
    ap.add_argument('--dump', default=None, help='save the images seen + a per-tick log here (record mode: required)')
    a = ap.parse_args()
    a.image_topic = a.image_topic or f'/camera_{a.side}/camera_{a.side}/color/image_rect_raw/compressed'
    a.cmd_topic = a.cmd_topic or f'/leader/joint_trajectory_command_broadcaster_{a.side}/joint_trajectory'
    if a.mode in ('record', 'snap', 'replay_plan') and not a.dump: ap.error(f'--mode {a.mode} needs --dump')
    if a.mode == 'replay_plan' and not a.plan: ap.error('--mode replay_plan needs --plan')
    replay(a) if a.replay else ros(a)
