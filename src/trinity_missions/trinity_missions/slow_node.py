"""
Slow mission node.

Trigger (always on)
  IDLE    -> ARMED    sign_slow (score >= trigger_conf) seen in trigger_frames consecutive
                      frames (not during the cooldown)
  ARMED   -> IDLE     no sign_slow seen for arm_timeout (each new sign sighting re-arms)
  ARMED   -> REQUEST  stop line down to the reference row: its lowest pixel row is at or
                      below stop_line_row_ratio * image height, with at least
                      stop_line_min_pixels pixels, for stop_line_frames consecutive frames
                      (2026-11-18: the road "slow" text request was removed)
  REQUEST -> IDLE     not granted within request_timeout (stale request)
Control (only in [slow])
  /stanley_cmd_vel -> /slow_cmd_vel with v' = min(v, slow_speed) and w' = w * v'/v, i.e. the
  same steering angle (stanley: w = v tan(delta) / wheelbase), only slower.
  finish() after slow_duration (= slow section length / slow_speed + margin).
"""
import signal

from geometry_msgs.msg import Twist
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from trinity_msgs.msg import MissionState
from vision_msgs.msg import Detection2DArray

from trinity_missions.cmd_utils import limit_speed
from trinity_missions.mission_base import MissionNode


class SlowNode(MissionNode):

    def __init__(self):
        super().__init__('slow_node', 'slow')
        self.declare_parameter('detections_topic', '/yolopx/detections')
        self.declare_parameter('stop_line_mask_topic', '/yolopx/stop_line_mask')
        self.declare_parameter('stanley_cmd_topic', '/stanley_cmd_vel')
        self.declare_parameter('cmd_topic', '/slow_cmd_vel')
        self.declare_parameter('trigger_classes', ['sign_slow'])
        self.declare_parameter('trigger_conf', 0.5)
        self.declare_parameter('trigger_frames', 3)
        self.declare_parameter('arm_timeout', 5.0)
        self.declare_parameter('stop_line_row_ratio', 0.80)
        self.declare_parameter('stop_line_min_pixels', 200)
        self.declare_parameter('stop_line_frames', 2)
        self.declare_parameter('image_height', 480)  # px, detection boxes are in input pixels
        self.declare_parameter('request_timeout', 2.0)
        self.declare_parameter('slow_speed', 0.2)
        self.declare_parameter('slow_duration', 5.0)

        p = self.get_parameter
        self.trigger_classes = set(p('trigger_classes').value)
        self.trigger_conf = float(p('trigger_conf').value)
        self.trigger_frames = int(p('trigger_frames').value)
        self.arm_timeout = float(p('arm_timeout').value)
        self.stop_row_ratio = float(p('stop_line_row_ratio').value)
        self.stop_min_px = int(p('stop_line_min_pixels').value)
        self.stop_frames = int(p('stop_line_frames').value)
        self.image_height = int(p('image_height').value)
        self.request_timeout = float(p('request_timeout').value)
        self.slow_speed = float(p('slow_speed').value)
        self.slow_duration = float(p('slow_duration').value)

        self.trig_count = 0          # consecutive frames with a trigger
        self.trig_class = ''
        self.stop_count = 0          # consecutive frames with the stop line at the reference
        self.armed_at = None         # s, last trigger sighting while ARMED
        self.request_at = None       # s

        self.create_subscription(Detection2DArray, p('detections_topic').value,
                                 self.detections_callback, qos_profile_sensor_data)
        self.create_subscription(Image, p('stop_line_mask_topic').value,
                                 self.stop_line_callback, qos_profile_sensor_data)
        self.create_subscription(Twist, p('stanley_cmd_topic').value, self.stanley_callback, 10)
        self.pub_cmd = self.create_publisher(Twist, p('cmd_topic').value, 10)
        self.get_logger().info(
            f'ready: {sorted(self.trigger_classes)} x{self.trigger_frames} -> ARMED '
            f'({self.arm_timeout:.0f} s) -> stop line at {self.stop_row_ratio:.2f} H -> REQUEST | '
            f'slow {self.slow_speed:.2f} m/s for {self.slow_duration:.1f} s, '
            f'cooldown {self.cooldown:.0f} s')

    # ---------------------------------------------------------------- trigger
    def detections_callback(self, msg):
        if self.state not in (MissionState.IDLE, MissionState.ARMED):
            self.trig_count = 0
            return
        best = None
        for det in msg.detections:
            for hyp in det.results:
                h = hyp.hypothesis
                if h.class_id in self.trigger_classes and h.score >= self.trigger_conf:
                    if best is None or h.score > best[1]:
                        best = (h.class_id, h.score)
        if best is None:
            self.trig_count = 0
            return
        self.trig_count += 1
        self.trig_class = best[0]
        if self.trig_count < self.trigger_frames:
            return
        now = self.now()
        if self.state == MissionState.IDLE:
            if self.in_cooldown(now):
                return
            self.set_state(MissionState.ARMED, f'{self.trig_class} x{self.trig_count}')
        self.armed_at = now  # (re)arm on every sign sighting

    def stop_line_callback(self, msg):
        if self.state != MissionState.ARMED:
            self.stop_count = 0
            return
        mask = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width]
        rows = np.flatnonzero(mask.max(axis=1))
        n_px = int(np.count_nonzero(mask))
        bottom = int(rows[-1]) if len(rows) else -1
        reached = n_px >= self.stop_min_px and bottom >= self.stop_row_ratio * (msg.height - 1)
        self.stop_count = self.stop_count + 1 if reached else 0
        if self.stop_count >= self.stop_frames:
            self.request_at = self.now()
            self.set_state(MissionState.REQUEST,
                           f'stop line at row {bottom}/{msg.height}, {n_px} px')

    def tick(self, now):
        if self.state == MissionState.ARMED and now - self.armed_at > self.arm_timeout:
            self.set_state(MissionState.IDLE,
                           f'no stop line within {self.arm_timeout:.0f} s')
        elif self.state == MissionState.REQUEST and now - self.request_at > self.request_timeout:
            self.set_state(MissionState.IDLE, f'not granted within {self.request_timeout:.0f} s')
        elif (self.state == MissionState.RUNNING and self.active
              and now - self.active_since >= self.slow_duration):
            self.finish(f'{self.slow_duration:.1f} s elapsed')

    # ---------------------------------------------------------------- control
    def on_start(self, now):
        self.trig_count = 0
        self.stop_count = 0

    def stanley_callback(self, msg):
        if self.active:
            self.pub_cmd.publish(limit_speed(msg, self.slow_speed))


def main(args=None):
    rclpy.init(args=args)
    node = SlowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Ctrl+C also reaches us via launch: ignore repeats while cleaning up
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
