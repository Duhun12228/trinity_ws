"""
Crosswalk mission node.

Trigger (always on)
  IDLE    -> REQUEST  sign_pedestrian (score >= trigger_conf) in trigger_frames consecutive
                      frames, not during the cooldown. The master gives [crosswalk] right
                      away and interrupts [slow] (preempt rule crosswalk>slow); slow_node
                      drops its own pending state while [crosswalk] lasts (suppress_modes).
  IDLE    -> ARMED    (2026-11-18, crosswalks without the sign) light_arm_class (red) in
                      light_arm_frames consecutive frames. Silent: the master keeps driving
                      (normal / slow, own speed); only the dashboard shows it.
  ARMED   -> REQUEST  a crosswalk box (crosswalk_conf) in cw_arm_frames consecutive frames
                      (the sign also requests right away, as from IDLE)
  ARMED   -> IDLE     no light_arm_keep colour (red / yellow) for arm_timeout, or green in
                      disarm_green_frames consecutive frames (pass on green)
  REQUEST -> IDLE     not granted within request_timeout
Phases in [crosswalk]
  approach    lane follow capped at approach_speed.
              -> closing  crosswalk box bottom edge at or below stop_row_ratio * image_height
                       in stop_frames consecutive frames. When the box flickers out, the
                       last box seen in the last box_hold_sec is used.
  closing     keep approaching for stop_delay_sec more, then -> stop
              -> DONE  approach_timeout without a crosswalk box AND without the sign
                       (counted from the last crosswalk box or sign_pedestrian seen, or
                       from entering the mode): the timeout is extended as long as the
                       crosswalk or the sign stays in view (2026-11-18: the sign is seen
                       early, far before the crosswalk)
  stop        zero speed, the traffic light watched every frame (best of red / yellow /
              green above light_conf):
                green in green_frames consecutive frames, after min_stop_sec -> DONE (go)
                stop_duration over and no red / yellow seen in this stop     -> DONE (go:
                                                         no traffic light, or green only)
                stop_duration over after red / yellow                        -> wait_green
  wait_green  zero speed until green in green_frames consecutive frames -> DONE (go)
DONE: the master switches back to normal right away (lane follow at its normal speed);
the base cooldown then blocks re-entry while the same crosswalk is still in view.
"""
from collections import Counter
import signal

from geometry_msgs.msg import Twist
import rclpy
from rclpy.qos import qos_profile_sensor_data
from trinity_msgs.msg import MissionState
from vision_msgs.msg import Detection2DArray

from trinity_missions.cmd_utils import limit_speed
from trinity_missions.mission_base import MissionNode

APPROACH, CLOSING, STOP, WAIT_GREEN = 'approach', 'closing', 'stop', 'wait_green'


class CrosswalkNode(MissionNode):

    def __init__(self):
        super().__init__('crosswalk_node', 'crosswalk')
        self.declare_parameter('detections_topic', '/yolopx/detections')
        self.declare_parameter('stanley_cmd_topic', '/stanley_cmd_vel')
        self.declare_parameter('cmd_topic', '/crosswalk_cmd_vel')
        self.declare_parameter('trigger_class', 'sign_pedestrian')
        self.declare_parameter('trigger_conf', 0.5)
        self.declare_parameter('trigger_frames', 3)
        self.declare_parameter('request_timeout', 1.0)
        self.declare_parameter('approach_speed', 0.2)
        self.declare_parameter('approach_timeout', 5.0)
        self.declare_parameter('crosswalk_class', 'crosswalk')
        self.declare_parameter('crosswalk_conf', 0.3)
        self.declare_parameter('stop_row_ratio', 0.85)
        self.declare_parameter('stop_frames', 2)
        self.declare_parameter('stop_delay_sec', 1.5)
        self.declare_parameter('box_hold_sec', 0.3)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('stop_duration', 4.0)
        self.declare_parameter('light_red_classes', ['traffic_light_red', 'traffic_light_yellow'])
        self.declare_parameter('light_green_class', 'traffic_light_green')
        self.declare_parameter('light_conf', 0.4)
        self.declare_parameter('light_min_frames', 5)
        self.declare_parameter('green_frames', 3)
        self.declare_parameter('min_stop_sec', 1.0)
        self.declare_parameter('light_arm_class', 'traffic_light_red')
        self.declare_parameter('light_arm_keep', ['traffic_light_red', 'traffic_light_yellow'])
        self.declare_parameter('light_arm_conf', 0.4)
        self.declare_parameter('light_arm_frames', 3)
        self.declare_parameter('arm_timeout', 5.0)
        self.declare_parameter('cw_arm_frames', 2)
        self.declare_parameter('disarm_green_frames', 3)

        p = self.get_parameter
        self.trigger_class = p('trigger_class').value
        self.trigger_conf = float(p('trigger_conf').value)
        self.trigger_frames = int(p('trigger_frames').value)
        self.request_timeout = float(p('request_timeout').value)
        self.approach_speed = float(p('approach_speed').value)
        self.approach_timeout = float(p('approach_timeout').value)
        self.cw_class = p('crosswalk_class').value
        self.cw_conf = float(p('crosswalk_conf').value)
        self.stop_row = float(p('stop_row_ratio').value) * (int(p('image_height').value) - 1)
        self.image_height = int(p('image_height').value)
        self.stop_frames = int(p('stop_frames').value)
        self.stop_delay = float(p('stop_delay_sec').value)
        self.box_hold = float(p('box_hold_sec').value)
        self.stop_duration = float(p('stop_duration').value)
        self.red_classes = set(p('light_red_classes').value)
        self.green_class = p('light_green_class').value
        self.light_conf = float(p('light_conf').value)
        self.light_min_frames = int(p('light_min_frames').value)
        self.green_frames = int(p('green_frames').value)
        self.light_classes = self.red_classes | {self.green_class}
        self.min_stop = float(p('min_stop_sec').value)
        self.arm_class = p('light_arm_class').value
        self.arm_keep = set(p('light_arm_keep').value)
        self.arm_conf = float(p('light_arm_conf').value)
        self.arm_frames = int(p('light_arm_frames').value)
        self.arm_timeout = float(p('arm_timeout').value)
        self.cw_arm_frames = int(p('cw_arm_frames').value)
        self.disarm_green = int(p('disarm_green_frames').value)
        self.arm_count = 0           # IDLE: red frames in a row
        self.armed_light_at = None   # s, last red / yellow while ARMED
        self.cw_arm_count = 0        # ARMED: crosswalk box frames in a row
        self.disarm_count = 0        # ARMED: green frames in a row

        self.trig_count = 0
        self.request_at = None
        self.reset_run()

        self.create_subscription(Detection2DArray, p('detections_topic').value,
                                 self.detections_callback, qos_profile_sensor_data)
        self.create_subscription(Twist, p('stanley_cmd_topic').value, self.stanley_callback, 10)
        self.pub_cmd = self.create_publisher(Twist, p('cmd_topic').value, 10)
        self.get_logger().info(
            f'ready: {self.trigger_class} x{self.trigger_frames} -> [crosswalk] approach '
            f'{self.approach_speed:.2f} m/s (gives up {self.approach_timeout:.0f} s after '
            f'the last {self.cw_class} box or {self.trigger_class}) -> stop when '
            f'{self.cw_class} bottom >= row {self.stop_row:.0f} x{self.stop_frames} '
            f'+ {self.stop_delay:.1f} s -> '
            f'{self.stop_duration:.0f} s + traffic light')

    def reset_run(self):
        self.cw_count = 0            # consecutive frames with the crosswalk at the stop row
        self.cw_bottom = None        # px, last crosswalk box bottom seen
        self.cw_seen_at = None       # s
        self.sign_seen_at = None     # s, last sign_pedestrian in [crosswalk] approach
        self.closing_since = None    # s, reference row reached
        self.stop_since = None       # s
        self.lights = Counter()      # this stop: red / yellow / green / none frames
        self.saw_red = False         # this stop: red / yellow seen
        self.green_count = 0

    # --------------------------------------------------------------- trigger
    def detections_callback(self, msg):
        now = self.now()
        best = self.best_scores(msg)
        if self.state == MissionState.IDLE:
            self.check_trigger(best, now)
            if self.state == MissionState.IDLE:
                self.check_light_arm(best, now)
        elif self.state == MissionState.ARMED:
            self.check_trigger(best, now)          # the sign: request right away
            if self.state == MissionState.ARMED:
                self.check_armed(msg, best, now)
        elif self.state == MissionState.RUNNING:
            if self.phase == APPROACH:
                if best.get(self.trigger_class, 0.0) >= self.trigger_conf or \
                        any(best.get(c, 0.0) >= self.arm_conf for c in self.arm_keep):
                    self.sign_seen_at = now      # still the sign / red light: keep approaching
                self.check_stop(msg, now)
            elif self.phase in (STOP, WAIT_GREEN):
                self.record_light(best)

    def best_scores(self, msg):
        best = {}
        for det in msg.detections:
            for hyp in det.results:
                h = hyp.hypothesis
                best[h.class_id] = max(best.get(h.class_id, 0.0), h.score)
        return best

    def check_trigger(self, best, now):
        if best.get(self.trigger_class, 0.0) < self.trigger_conf or self.in_cooldown(now):
            self.trig_count = 0
            return
        self.trig_count += 1
        if self.trig_count >= self.trigger_frames:
            self.trig_count = 0
            self.request_at = now
            self.set_state(MissionState.REQUEST, f'{self.trigger_class} x{self.trigger_frames}')

    def check_light_arm(self, best, now):
        """IDLE: red light -> ARMED (silent, the master keeps driving)."""
        if best.get(self.arm_class, 0.0) < self.arm_conf or self.in_cooldown(now):
            self.arm_count = 0
            return
        self.arm_count += 1
        if self.arm_count >= self.arm_frames:
            self.arm_count = self.cw_arm_count = self.disarm_count = 0
            self.armed_light_at = now
            self.set_state(MissionState.ARMED, f'{self.arm_class} x{self.arm_frames}, '
                                               f'waiting for the {self.cw_class}')

    def check_armed(self, msg, best, now):
        """ARMED: crosswalk box -> REQUEST; red / yellow keeps it; green -> IDLE."""
        if any(best.get(c, 0.0) >= self.arm_conf for c in self.arm_keep):
            self.armed_light_at = now
        self.disarm_count = self.disarm_count + 1 \
            if best.get(self.green_class, 0.0) >= self.arm_conf else 0
        if self.disarm_count >= self.disarm_green:
            self.set_state(MissionState.IDLE, f'green x{self.disarm_count} while armed: pass')
            return
        self.cw_arm_count = self.cw_arm_count + 1 \
            if best.get(self.cw_class, 0.0) >= self.cw_conf else 0
        if self.cw_arm_count >= self.cw_arm_frames:
            self.request_at = now
            self.set_state(MissionState.REQUEST, f'red light + {self.cw_class} '
                                                 f'x{self.cw_arm_count}')

    # ---------------------------------------------------------------- phases
    def check_stop(self, msg, now):
        bottom = None
        for det in msg.detections:
            for hyp in det.results:
                h = hyp.hypothesis
                if h.class_id == self.cw_class and h.score >= self.cw_conf:
                    b = det.bbox.center.position.y + det.bbox.size_y / 2.0
                    bottom = b if bottom is None else max(bottom, b)
        if bottom is not None:
            self.cw_bottom, self.cw_seen_at = bottom, now
        elif self.cw_seen_at is not None and now - self.cw_seen_at <= self.box_hold:
            bottom = self.cw_bottom  # flickered out: keep the last box for a moment
        self.cw_count = self.cw_count + 1 if bottom is not None and bottom >= self.stop_row else 0
        if self.cw_count >= self.stop_frames:
            self.closing_since = now
            self.set_phase(CLOSING, f'{self.cw_class} bottom at row {bottom:.0f}/'
                           f'{self.image_height}, stop in {self.stop_delay:.1f} s')

    def record_light(self, best):
        lights = {c: s for c, s in best.items()
                  if c in self.light_classes and s >= self.light_conf}
        seen = max(lights, key=lights.get) if lights else 'none'
        if self.phase == STOP:
            self.lights[seen] += 1
            if seen in self.red_classes:
                self.saw_red = True
        self.green_count = self.green_count + 1 if seen == self.green_class else 0
        if self.green_count >= self.green_frames and \
                (self.phase == WAIT_GREEN or self.now() - self.stop_since >= self.min_stop):
            self.finish(f'green x{self.green_count}, go')

    def decide(self):
        green = self.lights[self.green_class]
        red = sum(self.lights[c] for c in self.red_classes)
        summary = f'green {green}, red/yellow {red}, none {self.lights["none"]}'
        if not self.saw_red:
            self.finish(f'no red in {self.stop_duration:.0f} s ({summary}), go')
        else:
            self.set_phase(WAIT_GREEN, f'red ({summary}): wait for green')

    def tick(self, now):
        if self.state == MissionState.REQUEST and now - self.request_at > self.request_timeout:
            self.set_state(MissionState.IDLE, f'not granted within {self.request_timeout:.0f} s')
        elif self.state == MissionState.ARMED and now - self.armed_light_at > self.arm_timeout:
            self.set_state(MissionState.IDLE, f'no red light for {self.arm_timeout:.0f} s')
        if self.state != MissionState.RUNNING or not self.active:
            return
        last_seen = max(t for t in (self.cw_seen_at, self.sign_seen_at, self.active_since)
                        if t is not None)
        if self.phase == APPROACH and now - last_seen > self.approach_timeout:
            self.finish(f'no {self.cw_class} box / {self.trigger_class} for '
                        f'{self.approach_timeout:.0f} s')
        elif self.phase == CLOSING and now - self.closing_since >= self.stop_delay:
            self.stop_since = now
            self.set_phase(STOP, f'{self.stop_delay:.1f} s after the reference row')
        elif self.phase == STOP and now - self.stop_since >= self.stop_duration:
            self.decide()
        if self.phase in (STOP, WAIT_GREEN) and self.state == MissionState.RUNNING:
            self.pub_cmd.publish(Twist())  # hold still, independent of the lane follow rate

    # --------------------------------------------------------------- control
    def on_start(self, now):
        self.reset_run()
        self.set_phase(APPROACH, f'{self.approach_speed:.2f} m/s')

    def stanley_callback(self, msg):
        if not self.active:
            return
        if self.state == MissionState.RUNNING and self.phase in (APPROACH, CLOSING):
            self.pub_cmd.publish(limit_speed(msg, self.approach_speed))
        elif self.state == MissionState.DONE:
            self.pub_cmd.publish(msg)  # going; the master switches to normal right away
        # stop / wait_green: zero published from tick()


def main(args=None):
    rclpy.init(args=args)
    node = CrosswalkNode()
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
