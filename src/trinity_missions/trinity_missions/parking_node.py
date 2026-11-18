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
  straight  lane follow capped at speed (straight_lane_follow) or straight, measured on odom
            from the trigger pose (origin = start_to_line before the stop line). Slots
            (2026-11-17): slot_count slots of slot_width side by side on turn_side, the first
            starting slot_first past the stop line, slot_open .. slot_open + slot_depth from
            the lane centre. The turn into slot k starts turn_radius before its centre; there
            the lidar points kept in the start frame decide: slot k occupied (slot_min_points
            in slot_occ_scans scans) -> go on to the next slot (the last one is always taken).
            straight_sec is only the time limit.
  turn      turn_sec at speed with the full steering to turn_side (w = v tan(steer) / wheelbase)
  in        in_sec straight (0 = skip)
  stop      zero for hold_sec, then DONE: the master goes back to normal
The times are measured on the track (like the 3road lane change): the real yaw rate is only
about 1/3 of the command, so the bicycle model cannot give them.
"""
import math
import signal

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan
from trinity_msgs.msg import MissionState
from vision_msgs.msg import Detection2DArray

from trinity_missions.cmd_utils import limit_speed
from trinity_missions.mission_base import MissionNode

STRAIGHT, TURN, IN, BACK, STOP = 'straight', 'turn', 'in', 'back', 'stop'


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
        d('require_sign', True)          # false: the stop line alone triggers (no sign, testing)
        d('speed', 0.2)
        d('straight_lane_follow', True)
        d('straight_sec', 4.0)
        d('turn_side', 'right')
        d('steer', 0.6)
        d('wheelbase', 0.2)
        d('turn_sec', 5.5)
        d('in_sec', 0.0)
        d('hold_sec', 30.0)
        # 2026-11-17: turn ends at turn_deg on the odom yaw (turn_sec stays the time limit),
        # then straight back back_dist on odom (wheels straight: the car is parallel to the
        # slot, so it only moves back towards the lane, never into the next slot)
        d('odom_topic', '/odometry/filtered')
        d('odom_timeout', 0.3)
        d('turn_deg', 0.0)               # deg, 0 = time only (the old behaviour)
        d('back_dist', 0.0)              # m, 0 = no backing
        d('back_speed', 0.1)             # m/s
        d('back_max_sec', 3.0)           # s, backing ends after this anyway
        # 2026-11-17: pick the first free one of slot_count slots (lidar, odom start frame)
        d('scan_topic', '/scan')
        d('laser_x', 0.103)
        d('start_to_line', 0.35)         # m, base_link to the stop line at the trigger
        d('slot_first', 0.46)            # m, stop line to the first slot's near edge
        d('slot_width', 0.50)            # m, along the lane
        d('slot_depth', 0.60)            # m, across
        d('slot_open', 0.25)             # m, lane centre to the slot opening
        d('slot_count', 3)
        d('slot_margin', 0.05)           # m, shrink the slot box (tape, neighbours)
        d('slot_min_points', 3)          # lidar points in the box -> occupied scan
        d('slot_occ_scans', 3)           # occupied scans (any, not in a row) -> occupied
        d('turn_radius', 0.62)           # m, full steering radius (measured 0.2 m/s right)

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.laser_x = float(p('laser_x'))
        self.start_to_line = float(p('start_to_line'))
        self.slot_first = float(p('slot_first'))
        self.slot_width = float(p('slot_width'))
        self.slot_depth = float(p('slot_depth'))
        self.slot_open = float(p('slot_open'))
        self.slot_count = int(p('slot_count'))
        self.slot_margin = float(p('slot_margin'))
        self.slot_min_points = int(p('slot_min_points'))
        self.slot_occ_scans = int(p('slot_occ_scans'))
        self.turn_radius = float(p('turn_radius'))
        self.origin = None           # odom pose at the start of [parking]
        self.slot_hits = [0] * self.slot_count
        self.target = 0              # slot index being approached
        self.trigger_class = p('trigger_class')
        self.trigger_conf = float(p('trigger_conf'))
        self.trigger_frames = int(p('trigger_frames'))
        self.trigger_min_area = float(p('trigger_min_area'))
        self.arm_timeout = float(p('arm_timeout'))
        self.stop_row_ratio = float(p('stop_line_row_ratio'))
        self.stop_min_px = int(p('stop_line_min_pixels'))
        self.stop_frames = int(p('stop_line_frames'))
        self.request_timeout = float(p('request_timeout'))
        self.require_sign = bool(p('require_sign'))
        self.speed = float(p('speed'))
        self.straight_lf = bool(p('straight_lane_follow'))
        self.sign = 1.0 if str(p('turn_side')).lower() == 'left' else -1.0
        self.w = self.speed * math.tan(float(p('steer'))) / float(p('wheelbase'))
        self.odom_timeout = float(p('odom_timeout'))
        self.turn_rad = math.radians(float(p('turn_deg')))
        self.back_dist = float(p('back_dist'))
        self.back_speed = abs(float(p('back_speed')))
        self.steps = [(STRAIGHT, float(p('straight_sec'))), (TURN, float(p('turn_sec'))),
                      (IN, float(p('in_sec'))),
                      (BACK, float(p('back_max_sec')) if self.back_dist > 0.0 else 0.0),
                      (STOP, float(p('hold_sec')))]
        self.odom = None             # (x, y, yaw)
        self.odom_at = None
        self.step_from = None        # odom pose at the start of the step

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
        self.create_subscription(Odometry, p('odom_topic'), self.odom_callback, 10)
        self.create_subscription(LaserScan, p('scan_topic'), self.scan_callback,
                                 qos_profile_sensor_data)
        self.pub_cmd = self.create_publisher(Twist, p('cmd_topic'), 10)
        plan = ', '.join(f'{n} {t:.1f} s' for n, t in self.steps if t > 0)
        self.get_logger().info(
            f'ready: {self.trigger_class} (box >= {self.trigger_min_area:.0f} px^2) '
            f'x{self.trigger_frames} -> stop line at row '
            f'{self.stop_row_ratio:.2f} -> [parking] {plan} '
            f'({self.speed:.2f} m/s, {p("turn_side")} w {self.w:.3f}); slots at '
            + ', '.join(f'{self.turn_at(k):.2f}' for k in range(self.slot_count))
            + ' m on odom (turn start)')

    # ----------------------------------------------------------------- slots
    def slot_x(self, k):
        """Centre of slot k along the lane, from the start pose (m)."""
        return self.start_to_line + self.slot_first + (k + 0.5) * self.slot_width

    def turn_at(self, k):
        return self.slot_x(k) - self.turn_radius

    def to_start(self, x, y):
        """odom -> start frame: (along, to the turn side)."""
        x0, y0, h0 = self.origin
        dx, dy = x - x0, y - y0
        a = math.cos(h0) * dx + math.sin(h0) * dy
        c = -math.sin(h0) * dx + math.cos(h0) * dy     # + left
        return a, c * self.sign                          # + towards turn_side

    def scan_callback(self, msg):
        if not (self.active and self.state == MissionState.RUNNING and self.phase == STRAIGHT) \
                or self.origin is None or not self.odom_fresh(self.now()):
            return
        r = np.asarray(msg.ranges, dtype=np.float64)
        ang = msg.angle_min + np.arange(len(r)) * msg.angle_increment
        ok = np.isfinite(r) & (r > msg.range_min) & (r < msg.range_max)
        bx, by = r[ok] * np.cos(ang[ok]) + self.laser_x, r[ok] * np.sin(ang[ok])
        X, Y, H = self.odom
        a, s = self.to_start(X + math.cos(H) * bx - math.sin(H) * by,
                             Y + math.sin(H) * bx + math.cos(H) * by)
        m = self.slot_margin
        side = (s > self.slot_open + m) & (s < self.slot_open + self.slot_depth - m)
        for k in range(self.slot_count):
            x1 = self.start_to_line + self.slot_first + k * self.slot_width
            n = int(np.count_nonzero(side & (a > x1 + m) & (a < x1 + self.slot_width - m)))
            if n >= self.slot_min_points:
                self.slot_hits[k] += 1

    def occupied(self, k):
        return self.slot_hits[k] >= self.slot_occ_scans

    def slots_text(self):
        return ' '.join(f'{k + 1}:{"X" if self.occupied(k) else "-"}({self.slot_hits[k]})'
                        for k in range(self.slot_count))

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.odom = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)
        self.odom_at = self.now()

    def odom_fresh(self, now):
        return self.odom_at is not None and now - self.odom_at <= self.odom_timeout

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
        armed = self.state == MissionState.ARMED or (
            not self.require_sign and self.state == MissionState.IDLE
            and not self.in_cooldown(self.now()))
        if not armed:
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
        self.origin = self.odom if self.odom_fresh(now) else None
        if self.origin is None:
            self.get_logger().warn('no odom at the start: straight runs straight_sec, slot 1')
        self.slot_hits = [0] * self.slot_count
        self.target = 0
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
        self.step_from = self.odom if self.odom_fresh(now) else None
        extra = ''
        if name == STRAIGHT:
            extra = f' max, slot {self.target + 1} turn at {self.turn_at(self.target):.2f} m'
        elif name == TURN and self.turn_rad > 0.0:
            extra = f', or {math.degrees(self.turn_rad):.0f} deg on odom'
        elif name == BACK:
            extra = f' max, back {self.back_dist * 100:.0f} cm on odom'
        self.set_phase(name, f'{sec:.1f} s{extra}')

    def step_done(self, now):
        """Odom ends: the straight at the chosen slot's turn point, the turn at turn_deg, the
        backing at back_dist (else the time)."""
        if self.phase == STRAIGHT:
            if self.origin is None or not self.odom_fresh(now):
                return False
            along = self.to_start(self.odom[0], self.odom[1])[0]
            while along >= self.turn_at(self.target) and self.target < self.slot_count - 1 \
                    and self.occupied(self.target):
                self.get_logger().info(f'slot {self.target + 1} occupied ({self.slots_text()}) '
                                       f'-> slot {self.target + 2}')
                self.target += 1
                self.reason = f'slot {self.target + 1}, turn at {self.turn_at(self.target):.2f} m'
            if along >= self.turn_at(self.target):
                self.get_logger().info(f'slot {self.target + 1} chosen at {along:.2f} m '
                                       f'({self.slots_text()})')
                return True
            return False
        if self.step_from is None or not self.odom_fresh(now):
            return False
        x0, y0, h0 = self.step_from
        x, y, h = self.odom
        if self.phase == TURN and self.turn_rad > 0.0:
            turned = abs(math.atan2(math.sin(h - h0), math.cos(h - h0)))
            if turned >= self.turn_rad:
                self.get_logger().info(f'turn done: {math.degrees(turned):.0f} deg on odom')
                return True
        if self.phase == BACK:
            gone = math.hypot(x - x0, y - y0)
            if gone >= self.back_dist:
                self.get_logger().info(f'back done: {gone * 100:.0f} cm on odom')
                return True
        return False

    def tick(self, now):
        if self.state == MissionState.ARMED and now - self.armed_at > self.arm_timeout:
            self.set_state(MissionState.IDLE, f'no stop line within {self.arm_timeout:.0f} s')
        elif self.state == MissionState.REQUEST and now - self.request_at > self.request_timeout:
            self.set_state(MissionState.IDLE, f'not granted within {self.request_timeout:.0f} s')
        if self.state != MissionState.RUNNING or not self.active:
            if self.state == MissionState.DONE and self.active:
                self.pub_cmd.publish(Twist())   # stay still until the master switches
            return
        if now >= self.step_until or self.step_done(now):
            if self.phase == BACK and self.step_from is None:
                self.get_logger().warn('back: no odom at the start, skipped')
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
        elif self.phase == BACK:
            if self.step_from is not None and self.odom_fresh(now):
                cmd.linear.x = -self.back_speed   # wheels straight
            # no odom: no blind backing (zero until the time ends)
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
