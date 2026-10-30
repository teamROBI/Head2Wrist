
def main():
    #!/usr/bin/env python3
    """
    A fake FFW-SG2 follower for testing egoh2w_bridge.py without the robot (runs in the ai_worker container; numpy, cv2,
    rclpy only). Everything lives under --prefix (default /egoh2w_test) and should run on its own ROS domain, so nothing
    reaches the real robot's topics:
      publishes  <p>/joint_states (100 Hz: both arms + grippers, head, lift)
                 <p>/camera_{left,right}/camera_{left,right}/color/image_rect_raw/compressed (30 Hz JPEG, the feed's frames)
      listens    <p>/leader/joint_trajectory_command_broadcaster_{left,right}/joint_trajectory, <p>/leader/joystick_controller_right/
                 joint_trajectory (lift): like the JointTrajectoryController, each new single-point trajectory is followed
                 linearly from the current position over max(time_from_start, distance / --vmax)
      --states replay   the joints and images follow the recording (the camera check: record mode)
      --states command  the joints start at the recording's first frame and follow the commands; the images loop
    Feed: make_fake_feed.py (a recorded LeRobot episode -> npz). At exit: per arm the commands received, the largest commanded
    step and the command rate.
    """
    import argparse, time
    import numpy as np, cv2
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState, CompressedImage, CameraInfo
    from trajectory_msgs.msg import JointTrajectory

    ap = argparse.ArgumentParser(); ap.add_argument('--feed', required=True); ap.add_argument('--prefix', default='/egoh2w_test')
    ap.add_argument('--states', choices=['replay', 'command'], default='command'); ap.add_argument('--vmax', type=float, default=1.5, help='rad/s (m/s lift)')
    ap.add_argument('--max_s', type=float, default=120.0); a = ap.parse_args()
    F = dict(np.load(a.feed)); names = [str(x) for x in F['names']]; S0 = F['state']; fps = float(F['fps']); N = len(S0)   # dict: decompress once
    JOINTS = [n for n in names if n.startswith(('arm_', 'gripper_', 'head_', 'lift'))]; J = [names.index(n) for n in JOINTS]
    P = a.prefix.rstrip('/')


    class Fake(Node):
        def __init__(self):
            super().__init__('fake_sg2'); self.q = S0[0, J].astype(float).copy(); self.seg = {}; self.t0 = time.time(); self.k = 0
            self.stats = {}
            self.js = self.create_publisher(JointState, f'{P}/joint_states', 10)
            self.cam = {s: self.create_publisher(CompressedImage, f'{P}/camera_{s}/camera_{s}/color/image_rect_raw/compressed', qos_profile_sensor_data) for s in ('left', 'right')}
            self.ci = {s: self.create_publisher(CameraInfo, f'{P}/camera_{s}/camera_{s}/color/camera_info', qos_profile_sensor_data) for s in ('left', 'right')}
            self.zed = self.create_publisher(CompressedImage, f'{P}/zed/zed_node/left/image_rect_color/compressed', qos_profile_sensor_data)
            self.zci = self.create_publisher(CameraInfo, f'{P}/zed/zed_node/left/camera_info', qos_profile_sensor_data)
            for s in ('left', 'right'):
                self.create_subscription(JointTrajectory, f'{P}/leader/joint_trajectory_command_broadcaster_{s}/joint_trajectory', lambda m, s=s: self.on_cmd(m, s), 10)
            self.create_subscription(JointTrajectory, f'{P}/leader/joystick_controller_right/joint_trajectory', lambda m: self.on_cmd(m, 'lift'), 10)
            self.create_timer(0.01, self.step); self.create_timer(1.0/fps, self.frame)
            self.get_logger().info(f'fake SG2 under {P}: {len(JOINTS)} joints, feed {str(F["source"])} ({N} frames), states {a.states}')

        def on_cmd(self, m, who):
            if not m.points: return
            now = time.time(); st = self.stats.setdefault(who, dict(n=0, t0=now, max_step=0.0))
            idx = [JOINTS.index(n) for n in m.joint_names if n in JOINTS]; tgt = np.array(m.points[0].positions, float)[:len(idx)]
            if who != 'lift': st['max_step'] = max(st['max_step'], float(np.abs(tgt[:7] - self.q[idx][:7]).max()))
            st['n'] += 1; st['t1'] = now
            dur = max(m.points[0].time_from_start.sec + 1e-9*m.points[0].time_from_start.nanosec, float(np.abs(tgt - self.q[idx]).max())/a.vmax, 1e-3)
            for i, g in zip(idx, tgt): self.seg[i] = (now, self.q[i], g, dur)

        def step(self):
            if not rclpy.ok(): return
            now = time.time()
            if now - self.t0 > a.max_s: rclpy.shutdown(); return
            if a.states == 'replay': self.q = S0[self.k % N, J].astype(float)
            else:
                for i, (t0, q0, g, dur) in list(self.seg.items()):
                    f = min(1.0, (now - t0)/dur); self.q[i] = q0 + f*(g - q0)
                    if f >= 1.0: del self.seg[i]
            m = JointState(); m.header.stamp = self.get_clock().now().to_msg(); m.name = JOINTS; m.position = [float(x) for x in self.q]; self.js.publish(m)

        def frame(self):
            if not rclpy.ok(): return
            k = self.k % N
            for s in ('left', 'right'):
                m = CompressedImage(); m.header.stamp = self.get_clock().now().to_msg(); m.format = 'bgr8; jpeg compressed bgr8'
                m.data = cv2.imencode('.jpg', F[s][k], [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes(); self.cam[s].publish(m)
                ci = CameraInfo(); ci.width, ci.height = 424, 240; ci.distortion_model = 'plumb_bob'; ci.d = [0.0]*5   # a plausible D405 colour stream
                ci.k = [216.5, 0.0, 213.1, 0.0, 216.5, 119.6, 0.0, 0.0, 1.0]; self.ci[s].publish(ci)
            if 'head' in F and len(F['head']) > 1:                  # the ZED head stream (placement check)
                hi = F['head'][k % len(F['head'])]; m = CompressedImage(); m.format = 'bgr8; jpeg compressed bgr8'; m.data = cv2.imencode('.jpg', hi)[1].tobytes(); self.zed.publish(m)
                ci = CameraInfo(); ci.height, ci.width = hi.shape[:2]; f = 0.5*hi.shape[1]/np.tan(np.radians(55)); ci.k = [f, 0.0, hi.shape[1]/2, 0.0, f, hi.shape[0]/2, 0.0, 0.0, 1.0]; self.zci.publish(ci)
            self.k += 1


    rclpy.init(); n = Fake()
    try: rclpy.spin(n)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException, rclpy._rclpy_pybind11.RCLError): pass   # SIGINT shuts the context down mid-spin
    finally:
        for who, st in n.stats.items():
            dt = st.get('t1', st['t0']) - st['t0']
            print(f'commands {who}: {st["n"]} at {st["n"]/max(dt, 1e-6):.1f} Hz, largest commanded arm step {st["max_step"]:.3f} rad')
        print('final joints', {nm: round(float(v), 3) for nm, v in zip(JOINTS, n.q) if nm.startswith(('arm_r', 'gripper_r', 'lift'))})
        if rclpy.ok(): rclpy.shutdown()


if __name__ == '__main__':
    main()
