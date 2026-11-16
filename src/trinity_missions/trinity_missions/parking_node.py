"""
Parking mission node: open-loop (time based) forward parking into one fixed slot.

Map (LIMO competition map, from the stop line across the parking lane): the target slot
(middle column, upper row) starts 988 mm and ends 1504 mm past the stop line (centre 1246),
on the right side, 260..877 mm from the lane centre (centre 569). LIMO 322 x 220 mm.

Trigger (always on)
  IDLE    -> ARMED    sign_parking (score >= trigger_conf) with a box of at least
                      trigger_min_area px^2 (near enough) in trigger_frames consecutive frames
                      (not during the cooldown)
  ARMED   -> IDLE     no stop line within arm_timeout (each new sign sighting re-arms)
  ARMED   -> REQUEST  stop line down to the reference row: its lowest pixel row at or below
                      stop_line_row_ratio * image height, with at least stop_line_min_pixels
                      pixels, for stop_line_frames consecutive frames
  REQUEST -> IDLE     not granted within request_timeout
Phases in [parking] (also when forced with /master/set_mode parking: handy for tuning)
  straight  straight_sec: lane follow capped at speed (straight_lane_follow) or straight
  turn      turn_sec at speed with the full steering to turn_side (w = v tan(steer) / wheelbase)
  in        in_sec straight (0 = skip)
  stop      zero for hold_sec, then DONE: the master goes back to normal
The times are measured on the track (like the 3road lane change): the real yaw rate is only
about 1/3 of the command, so the bicycle model cannot give them.
"""
import math
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

STRAIGHT, TURN, IN, STOP = 'straight', 'turn', 'in', 'stop'


class ParkingNode(MissionNode):

    def __init__(self):
        super().__init__('parking_node', 'parking')
        d = self.declare_parameter
        d('detections_topic', '/yolopx/detections')
        d('stop_line_mask_topic', '/yolopx/stop_line_mask')
        d('stanley_cmd_topic', '/stanley_cmd_vel')
        d('cmd_topic', '/parking_cmd_vel')
        d('trigger_class', 'sign_parking')
        d('trigger_conf', 0.5)
        d('trigger_frames', 3)
        d('trigger_min_area', 2000.0)
        d('arm_timeout', 10.0)
        d('stop_line_row_ratio', 0.80)
        d('stop_line_min_pixels', 200)
        d('stop_line_frames', 2)
        d('request_timeout', 1.0)
        d('speed', 0.2)
        d('straight_lane_follow', True)
        d('straight_sec', 4.0)
        d('turn_side', 'right')
        d('steer', 0.6)
        d('wheelbase', 0.2)
        d('turn_sec', 5.5)
        d('in_sec', 0.0)
        d('hold_sec', 30.0)

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.trigger_class = p('trigger_class')
        self.trigger_conf = float(p('trigger_conf'))
        self.trigger_frames = int(p('trigger_frames'))
        self.trigger_min_area = float(p('trigger_min_area'))
        self.arm_timeout = float(p('arm_timeout'))
        self.stop_row_ratio = float(p('stop_line_row_ratio'))
        self.stop_min_px = int(p('stop_line_min_pixels'))
        self.stop_frames = int(p('stop_line_frames'))
        self.request_timeout = float(p('request_timeout'))
        self.speed = float(p('speed'))
        self.straight_lf = bool(p('straight_lane_follow'))
        self.sign = 1.0 if str(p('turn_side')).lower() == 'left' else -1.0
        self.w = self.speed * math.tan(float(p('steer'))) / float(p('wheelbase'))
        self.steps = [(STRAIGHT, float(p('straight_sec'))), (TURN, float(p('turn_sec'))),
                      (IN, float(p('in_sec'))), (STOP, float(p('hold_sec')))]

        self.trig_count = 0
        self.sign_area = 0.0         # px^2, largest sign box in the last frame (debug)
        self.stop_count = 0
        self.armed_at = None
        self.request_at = None
        self.step = 0                # index into self.steps
        self.step_until = None       # s

        self.create_subscription(Detection2DArray, p('detections_topic'),
                                 self.detections_callback, qos_profile_sensor_data)
        self.create_subscription(Image, p('stop_line_mask_topic'), self.stop_line_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(Twist, p('stanley_cmd_topic'), self.stanley_callback, 10)
        self.pub_cmd = self.create_publisher(Twist, p('cmd_topic'), 10)
        plan = ', '.join(f'{n} {t:.1f} s' for n, t in self.steps if t > 0)
        self.get_logger().info(
            f'ready: {self.trigger_class} (box >= {self.trigger_min_area:.0f} px^2) '
            f'x{self.trigger_frames} -> stop line at row '
            f'{self.stop_row_ratio:.2f} -> [parking] {plan} '
            f'({self.speed:.2f} m/s, {p("turn_side")} w {self.w:.3f})')

    # --------------------------------------------------------------- trigger
    def detections_callback(self, msg):
        now = self.now()
        area = 0.0   # largest parking sign box (px^2, yolopx input pixels)
        for det in msg.detections:
            if any(h.hypothesis.class_id == self.trigger_class
                   and h.hypothesis.score >= self.trigger_conf for h in det.results):
                area = max(area, det.bbox.size_x * det.bbox.size_y)
        self.sign_area = area
        if self.state not in (MissionState.IDLE, MissionState.ARMED) or self.in_cooldown(now):
            self.trig_count = 0
            return
        if 0.0 < area < self.trigger_min_area and self.state == MissionState.IDLE:
            # seen but still far: show the size on the dashboard for tuning
            self.reason = f'{self.trigger_class} box {area:.0f} < {self.trigger_min_area:.0f} px^2'
        self.trig_count = self.trig_count + 1 if area >= self.trigger_min_area else 0
        if self.trig_count >= self.trigger_frames:
            self.armed_at = now   # every new (close enough) sighting re-arms
            self.set_state(MissionState.ARMED, f'{self.trigger_class} x{self.trigger_frames}, '
                                               f'box {area:.0f} px^2')

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

    # ---------------------------------------------------------------- phases
    def on_start(self, now):
        self.step = -1
        self.next_step(now)

    def next_step(self, now):
        self.step += 1
        while self.step < len(self.steps) and self.steps[self.step][1] <= 0.0:
            self.step += 1   # skipped (0 s)
        if self.step >= len(self.steps):
            self.finish(f'parked, held {self.steps[-1][1]:.0f} s')
            return
        name, sec = self.steps[self.step]
        self.step_until = now + sec
        self.set_phase(name, f'{sec:.1f} s')

    def tick(self, now):
        if self.state == MissionState.ARMED and now - self.armed_at > self.arm_timeout:
            self.set_state(MissionState.IDLE, f'no stop line within {self.arm_timeout:.0f} s')
        elif self.state == MissionState.REQUEST and now - self.request_at > self.request_timeout:
            self.set_state(MissionState.IDLE, f'not granted within {self.request_timeout:.0f} s')
        if self.state != MissionState.RUNNING or not self.active:
            if self.state == MissionState.DONE and self.active:
                self.pub_cmd.publish(Twist())   # stay still until the master switches
            return
        if now >= self.step_until:
            self.next_step(now)
        if self.state != MissionState.RUNNING:
            self.pub_cmd.publish(Twist())
            return
        cmd = Twist()
        if self.phase == STRAIGHT and not self.straight_lf:
            cmd.linear.x = self.speed
        elif self.phase == TURN:
            cmd.linear.x, cmd.angular.z = self.speed, self.sign * self.w
        elif self.phase == IN:
            cmd.linear.x = self.speed
        elif self.phase == STRAIGHT:
            return   # lane follow: published from stanley_callback
        self.pub_cmd.publish(cmd)   # STOP: zero

    def stanley_callback(self, msg):
        if self.active and self.state == MissionState.RUNNING and self.phase == STRAIGHT \
                and self.straight_lf:
            self.pub_cmd.publish(limit_speed(msg, self.speed))


def main(args=None):
    rclpy.init(args=args)
    node = ParkingNode()
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
