"""
3-lane (3road) mission node: lane changes in the 3-lane section, always back to lane 2.

Mode name '3road'; topics use 'three_road' (ROS names cannot start with a digit).

Lane number (only while not changing lanes)
  /yolopx/drivable_class (0 bg / 1 my lane / 2 left / 3 right), bottom part of the image
  (rows below roi_top_ratio): a side lane counts when its pixel share >= side_min_ratio.
    left + right -> lane 2,  right only -> lane 1,  left only -> lane 3
  The same answer lane_frames frames in a row sets the lane.
Trigger (always on)
  IDLE  -> ARMED    a lane number was set (side lane seen: 3-lane section)
  ARMED -> IDLE     no side lane for zone_exit_sec
  ARMED -> REQUEST  from the lidar (/scan, base_link: x forward, y left):
    lane 2: an object in the front box (bumper .. bumper + obstacle_dist, |y| <
            front_half_width) for obstacle_scans scans -> the side whose lane box is
            empty; both empty: away from the object (y >= 0 -> right), y ~ 0 -> right;
            both blocked: wait
    lane 1: the right lane box (y = -side_offset) empty for return_clear_sec -> right
    lane 3: the left lane box (y = +side_offset) empty for return_clear_sec -> left
    lane boxes: side_x_min < x < side_x_max (the lidar sees +-60 deg only),
                |y -+ side_offset| < side_half_width
  REQUEST -> IDLE   not granted within request_timeout
Guard (inside the section): an object closer than guard_dist to the LIDAR in the front box
  (+-guard_half_width) for guard_scans scans while moving forward -> back off straight at
  backoff_speed for backoff_min..max s, until nothing within backoff_clear_dist. ARMED: requests
  [3road] for it; during a lane change the change is aborted. During a lane change the guard
  box reaches guard_turn_extra further to the side the car moves to (front corner sweep).
Settle (still in [3road]): right after every lane change (VERIFY), after a guard backoff
  (GUARD), after an object ahead during a moving result check (RECHECK). An object closer than
  backoff_clear_dist to the lidar when it starts: back off straight first (room to get past).
    stop    hold still settle_sec, VERIFY verify_stop_sec (camera / lane_follow path settle)
    align   not parallel (near / ahead > align_done_deg, VERIFY: > verify_align_skip_deg):
            back off at align_speed with
            w = align_k * yaw until parallel, at most align_max_sec (unknown yaw: skipped)
    judge   standing: the same lane lane_frames frames in a row (frames older than
            judge_discard_sec after the stop are dropped); none within judge_max_sec: the
            last frame with the weak side threshold weak_side_ratio; none: the lane known
  GUARD / RECHECK, then at once (turn / counter): not in the home lane -> one lane towards
  it; in it -> dodge (home 2: the free side, home 1 / 3: lane 2). Nothing possible -> DONE.
  A recheck that finds the original lane counts as a failed change (retry).
  VERIFY: target lane -> ok, DONE (lane follow), or with an object in the front box -> avoid
  at once; original lane -> the same change again at once (retry, max_retries); nothing
  judged -> DONE and the moving result check below.
Direction signs: inside the section sign_A / sign_B frames (score >= sign_conf) are counted;
  once one has >= sign_decide_frames and leads by >= sign_margin, the home lane becomes 1 (A)
  or 3 (B) instead of 2 for the rest of the section: the car moves there at once (blind, one
  lane at a time), dodges objects in it to lane 2 and comes back when the home lane box has
  been clear for return_clear_sec. While signs are being counted (one seen within
  sign_hold_sec, undecided) the return to lane 2 waits, so it does not go the wrong way.
Alignment: the lane number is judged, new sections entered and lane changes requested only
  while lane_follow's centre path heading (/lane_follow/yaw, near and ahead) is within
  +-align_max_deg (a corner hides one side lane -> wrong lane number). While not aligned the
  lane and its frame count are kept, the section exit timer pauses (at most align_pause_max)
  and the result check waits (at most verify_extend_max longer).
Running [3road]: open-loop lane change at change_speed with steer (bicycle model
  w = v tan(steer) / wheelbase), times measured on the track per direction (the steering
  is not symmetric):
    turn     steer to the target side for <dir>_turn_sec
    counter  steer to the other side for <dir>_counter_sec
  then the standing check (settle, VERIFY); after DONE the master goes back to normal,
  lane_follow resets its tracking (it watches /master/mode) and centres the car in the lane.
Moving result check (fallback when nothing was judged standing; the LIMO remote control can
  override /cmd_vel at the base): after DONE the lane is not assumed. After verify_settle
  the stable lane of the next verify_sec decides:
  target -> ok; the original lane (or none by the end) -> failed: back to ARMED at once (no
  cooldown) so the trigger runs again; after max_retries failures in a row no more lane
  changes (lane follow only) until the section ends.
Speed: inside the section (ARMED / REQUEST / lane known / checking) the MissionState
  carries speed_limit = section_speed, so the master caps lane follow at it in normal mode.
Debug: /three_road/debug (bgr8 image for rqt_image_view, only while subscribed): drivable
  classes with the lane shares and lane number, lidar top view with the front / lane boxes
  and the return timer, mission state and the lane_follow tracking state.
"""
from array import array
import math
import signal

import cv2
from geometry_msgs.msg import Twist
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Float32MultiArray, String
from trinity_msgs.msg import MissionState
from vision_msgs.msg import Detection2DArray

from trinity_missions.mission_base import MissionNode

TURN, COUNTER, BACKOFF = 'turn', 'counter', 'backoff'
STOP, ALIGN, JUDGE = 'stop', 'align', 'judge'   # settle steps before avoiding
GUARD = 'guard'                   # request kind: safety backoff instead of a lane change
RECHECK = 'recheck'               # request kind: stop, align, judge the lane, then avoid
VERIFY = 'verify'                 # settle origin: right after a lane change (result check)
LEFT, RIGHT = 'left', 'right'


def put(img, text, org, color, scale):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


class ThreeRoadNode(MissionNode):

    def __init__(self):
        super().__init__('three_road_node', '3road', topic_name='three_road')
        d = self.declare_parameter
        d('drivable_class_topic', '/yolopx/drivable_class')
        d('scan_topic', '/scan')
        d('cmd_topic', '/three_road_cmd_vel')
        # lane number
        d('roi_top_ratio', 0.5)
        d('side_min_ratio', 0.03)
        d('drivable_min_ratio', 0.03)
        d('lane_frames', 3)
        d('zone_exit_sec', 2.0)
        d('trigger_side_fresh_sec', 1.0)
        # direction signs: the more often seen one (with a margin) sets the home lane (instead
        # of lane 2) for the rest of the section: sign_A (left) -> 1, sign_B (right) -> 3
        d('detections_topic', '/yolopx/detections')
        d('sign_left_class', 'sign_A')
        d('sign_right_class', 'sign_B')
        d('sign_conf', 0.5)
        d('sign_decide_frames', 10)
        d('sign_margin', 5)
        d('sign_hold_sec', 1.0)
        # lidar
        d('laser_x', 0.103)
        d('laser_y', 0.0)
        d('laser_yaw', 0.0)
        d('front_x', 0.16)
        d('obstacle_dist', 0.8)
        d('front_half_width', 0.15)
        d('obstacle_scans', 3)
        d('side_offset', 0.5)
        d('side_half_width', 0.2)
        d('side_x_min', 0.3)
        d('side_x_max', 1.0)
        d('side_min_points', 2)
        d('return_clear_sec', 1.5)
        d('center_tol', 0.03)
        d('request_timeout', 1.0)
        # debug image for rqt_image_view (built only while someone subscribes)
        d('debug_topic', '/three_road/debug')
        d('debug_rate_hz', 10.0)
        d('lane_state_topic', '/lane_follow/state')
        # alignment: lane judgement / triggers only while the car is parallel to the lane
        d('yaw_topic', '/lane_follow/yaw')
        d('align_max_deg', 10.0)
        d('yaw_timeout', 0.5)
        d('align_pause_max', 5.0)
        d('verify_extend_max', 3.0)
        # lane change
        d('change_speed', 0.2)
        d('section_speed', 0.2)
        # guard (inside the section only): an object closer than guard_dist to the LIDAR in
        # the front box while moving forward -> back off, steering the car parallel to the lane
        d('cmd_vel_topic', '/cmd_vel')
        d('guard_dist', 0.2)
        d('guard_half_width', 0.15)
        d('guard_turn_extra', 0.10)
        d('guard_min_points', 2)
        d('guard_scans', 2)
        d('backoff_speed', -0.2)
        d('backoff_min_sec', 0.5)
        d('backoff_max_sec', 1.5)
        d('backoff_clear_dist', 0.35)
        # settle after a backoff / a recheck: stop, align, judge the lane, then avoid
        d('settle_sec', 1.0)
        d('align_done_deg', 5.0)
        d('align_speed', -0.15)
        d('align_k', 1.5)
        d('align_max_sec', 3.0)
        d('judge_max_sec', 1.0)
        d('judge_discard_sec', 0.2)
        d('weak_side_ratio', 0.005)
        # result check right after a lane change: stop and judge standing (shorter hold, align
        # only when not even judge-parallel)
        d('verify_stop_sec', 0.5)
        d('verify_align_skip_deg', 10.0)
        # result check: the remote control can override /cmd_vel at the base, so a lane
        # change is only trusted once the drivable classes show the target lane
        d('verify_settle', 1.0)
        d('verify_sec', 2.0)
        d('max_retries', 5)
        d('steer', 0.6)
        d('wheelbase', 0.2)
        d('left_turn_sec', 3.0)
        d('left_counter_sec', 3.6)
        d('right_turn_sec', 3.7)
        d('right_counter_sec', 3.0)

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.roi_top = float(p('roi_top_ratio'))
        self.side_min = float(p('side_min_ratio'))
        self.drivable_min = float(p('drivable_min_ratio'))
        self.lane_frames = int(p('lane_frames'))
        self.zone_exit = float(p('zone_exit_sec'))
        self.side_fresh = float(p('trigger_side_fresh_sec'))
        self.sign_classes = {p('sign_left_class'): 1, p('sign_right_class'): 3}
        self.sign_conf = float(p('sign_conf'))
        self.sign_decide = int(p('sign_decide_frames'))
        self.sign_margin = int(p('sign_margin'))
        self.sign_hold = float(p('sign_hold_sec'))
        self.laser = (float(p('laser_x')), float(p('laser_y')), float(p('laser_yaw')))
        self.front_x = float(p('front_x'))
        self.obstacle_dist = float(p('obstacle_dist'))
        self.front_half = float(p('front_half_width'))
        self.obstacle_scans = int(p('obstacle_scans'))
        self.side_offset = float(p('side_offset'))
        self.side_half = float(p('side_half_width'))
        self.side_x = (float(p('side_x_min')), float(p('side_x_max')))
        self.side_min_points = int(p('side_min_points'))
        self.return_clear = float(p('return_clear_sec'))
        self.center_tol = float(p('center_tol'))
        self.request_timeout = float(p('request_timeout'))
        self.debug_period = 1.0 / max(float(p('debug_rate_hz')), 0.1)
        self.align_max = float(p('align_max_deg'))
        self.yaw_timeout = float(p('yaw_timeout'))
        self.align_pause_max = float(p('align_pause_max'))
        self.verify_extend_max = float(p('verify_extend_max'))
        self.guard_dist = float(p('guard_dist'))
        self.guard_half = float(p('guard_half_width'))
        self.guard_turn_extra = float(p('guard_turn_extra'))
        self.guard_y = (-self.guard_half, self.guard_half)  # current guard box y range
        self.guard_min_points = int(p('guard_min_points'))
        self.guard_scans = int(p('guard_scans'))
        self.backoff_speed = float(p('backoff_speed'))
        self.backoff_min = float(p('backoff_min_sec'))
        self.backoff_max = float(p('backoff_max_sec'))
        self.backoff_clear = float(p('backoff_clear_dist'))
        self.settle_sec = float(p('settle_sec'))
        self.align_done = float(p('align_done_deg'))
        self.align_speed = float(p('align_speed'))
        self.align_k = float(p('align_k'))
        self.align_max_sec = float(p('align_max_sec'))
        self.judge_max = float(p('judge_max_sec'))
        self.judge_discard = float(p('judge_discard_sec'))
        self.weak_side = float(p('weak_side_ratio'))
        self.verify_stop = float(p('verify_stop_sec'))
        self.verify_align_skip = float(p('verify_align_skip_deg'))
        self.align_w_max = abs(self.align_speed) * math.tan(float(p('steer'))) / \
            float(p('wheelbase'))
        self.speed = float(p('change_speed'))
        self.section_speed = float(p('section_speed'))
        self.verify_settle = float(p('verify_settle'))
        self.verify_sec = float(p('verify_sec'))
        self.max_retries = int(p('max_retries'))
        self.w = self.speed * math.tan(float(p('steer'))) / float(p('wheelbase'))
        self.times = {LEFT: (float(p('left_turn_sec')), float(p('left_counter_sec'))),
                      RIGHT: (float(p('right_turn_sec')), float(p('right_counter_sec')))}

        self.lane = None             # current lane number 1 / 2 / 3, None outside the section
        self.cand, self.cand_count = None, 0
        self.side_seen_at = None     # s, last frame with a side lane (only when really seen)
        # s, end of the last lane change: the section exit timer starts no earlier than this
        self.zone_grace_from = None
        self.obstacle_count = 0
        self.side_clear_since = {LEFT: None, RIGHT: None}
        self.side_blocked = {LEFT: False, RIGHT: False}
        self.request = None          # (direction, target lane) of the pending / running change
        self.request_at = None
        self.phase_until = None      # s, end of the current turn / counter step
        self.verify = None           # pending result check: dict(orig, target, side, done_at)
        self.verify_seen = None      # stable lane judged during the check
        self.retries = 0             # failed lane changes in a row (this section)
        self.gave_up = False         # retries exhausted: lane follow only until the section ends
        self.home = 2                # lane to keep: 2, or 1 / 3 once a direction sign is decided
        self.home_pending = False    # sign just decided: go to the home lane at once (blind)
        self.sign_counts = {1: 0, 3: 0}  # frames with sign_A (-> 1) / sign_B (-> 3)
        self.sign_seen_at = None     # s, last frame with a direction sign (undecided)
        # newest inputs for the debug image
        self.dbg_cls = None          # drivable classes (downscaled)
        self.dbg_ratio = (0.0, 0.0, 0.0)  # left / mine / right pixel share in the ROI
        self.dbg_pts = (np.zeros(0), np.zeros(0))  # lidar points in base_link
        self.dbg_front = False       # something in the front box (this scan)
        self.front_oy = 0.0          # m, mean y of the front box points (last time seen)
        self.guard_count = 0         # scans in a row with an object inside guard_dist
        self.guard_near = None       # m, nearest object ahead (lidar x) in the guard width
        self.cmd_v = 0.0             # newest /cmd_vel linear.x (moving forward?)
        self.backoff = None          # running backoff: dict(start, change=(orig, target, side))
        # running settle: dict(origin, change=(orig, target, side) or None, since, ...)
        self.settle = None
        self.recheck = None          # (orig, target, side) of the change a RECHECK interrupts
        self.front_scans = 0         # scans in a row with an object in the front box (any state)
        self.dbg_box = {LEFT: False, RIGHT: False}  # lane boxes blocked (this scan)
        self.lane_state = ''         # /lane_follow/state
        self.yaw = (float('nan'), float('nan'))  # /lane_follow/yaw [near, ahead] deg
        self.yaw_at = None           # s
        self.unaligned_since = None  # s, start of the current not-aligned stretch
        self.unaligned_at = None     # s, last not-aligned frame (pauses the section exit)
        self.last_debug = 0.0

        self.create_subscription(Image, p('drivable_class_topic'), self.drivable_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(LaserScan, p('scan_topic'), self.scan_callback,
                                 qos_profile_sensor_data)
        self.pub_cmd = self.create_publisher(Twist, p('cmd_topic'), 10)
        self.create_subscription(Twist, p('cmd_vel_topic'),
                                 lambda m: setattr(self, 'cmd_v', m.linear.x), 10)
        self.pub_debug = self.create_publisher(Image, p('debug_topic'), 1)
        self.create_subscription(String, p('lane_state_topic'),
                                 lambda m: setattr(self, 'lane_state', m.data), 10)
        self.create_subscription(Float32MultiArray, p('yaw_topic'), self.yaw_callback, 10)
        self.create_subscription(Detection2DArray, p('detections_topic'), self.signs_callback,
                                 qos_profile_sensor_data)
        self.get_logger().info(
            f'ready: lane from {p("drivable_class_topic")} | lane 2: object < '
            f'{self.obstacle_dist:.1f} m -> free side, lane 1/3: lane 2 box clear '
            f'{self.return_clear:.1f} s -> back | change {self.speed:.2f} m/s w {self.w:.3f}, '
            f'left {self.times[LEFT][0]:.1f}/{self.times[LEFT][1]:.1f} s, '
            f'right {self.times[RIGHT][0]:.1f}/{self.times[RIGHT][1]:.1f} s')

    # ------------------------------------------------------------ lane number
    def signs_callback(self, msg):
        """Count sign_A / sign_B frames inside the section; decide the home lane once."""
        in_section = (self.state != MissionState.IDLE or self.lane is not None
                      or self.verify is not None)
        if not in_section or self.home != 2:
            return
        best = None
        for det in msg.detections:
            for hyp in det.results:
                h = hyp.hypothesis
                lane = self.sign_classes.get(h.class_id)
                if lane is not None and h.score >= self.sign_conf \
                        and (best is None or h.score > best[1]):
                    best = (lane, h.score)
        if best is None:
            return
        self.sign_seen_at = self.now()
        self.sign_counts[best[0]] += 1
        lead = max(self.sign_counts, key=self.sign_counts.get)
        other = 3 if lead == 1 else 1
        if self.sign_counts[lead] >= self.sign_decide and \
                self.sign_counts[lead] - self.sign_counts[other] >= self.sign_margin:
            self.home = lead
            self.home_pending = self.lane != lead
            names = {v: k for k, v in self.sign_classes.items()}
            why = (f'{names[lead]} x{self.sign_counts[lead]} vs {names[other]} '
                   f'x{self.sign_counts[other]} -> home lane {lead}')
            self.get_logger().info(why)
            if self.state == MissionState.ARMED:
                self.reason = why

    def step_toward_home(self):
        """(side, target) one lane towards the home lane, None when already there."""
        if self.lane is None or self.lane == self.home:
            return None
        return (LEFT, self.lane - 1) if self.home < self.lane else (RIGHT, self.lane + 1)

    def avoid_from_home(self, oy):
        """(side, target) to dodge an object ahead in the home lane, None when no side is free.
        Home 2: the free side (both free: away from the object, y ~ 0 -> right); home 1 / 3:
        the only neighbour, lane 2."""
        if self.home == 2:
            free = [sd for sd in (LEFT, RIGHT) if not self.dbg_box[sd]]
            if not free:
                return None
            side = free[0] if len(free) == 1 else (LEFT if oy < -self.center_tol else RIGHT)
            return side, (1 if side == LEFT else 3)
        side = RIGHT if self.home == 1 else LEFT
        return (side, 2) if not self.dbg_box[side] else None

    def yaw_callback(self, msg):
        if len(msg.data) >= 2:
            self.yaw, self.yaw_at = (float(msg.data[0]), float(msg.data[1])), self.now()

    def aligned(self, now):
        """Car parallel to the lane: lane_follow near and ahead heading within align_max_deg
        (a corner, a lost lane or a stale value is 'not aligned')."""
        if self.yaw_at is None or now - self.yaw_at > self.yaw_timeout:
            return False
        return all(math.isfinite(a) and abs(a) <= self.align_max for a in self.yaw)

    def drivable_callback(self, msg):
        cls = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width]
        roi = cls[int(msg.height * self.roi_top):]
        area = float(roi.size)
        ratio = tuple(np.count_nonzero(roi == c) / area for c in (2, 1, 3))
        self.dbg_cls, self.dbg_ratio = cls[::2, ::2], ratio
        now = self.now()
        mine = ratio[1]
        left = ratio[0] >= self.side_min
        right = ratio[2] >= self.side_min
        if left or right:
            self.side_seen_at = now
        if self.aligned(now):
            self.unaligned_since = None
        else:
            if self.unaligned_since is None:
                self.unaligned_since = now
            self.unaligned_at = now
        if self.settle is not None and self.phase == JUDGE:
            self.judge_frame(now, ratio)
            return
        if self.state in (MissionState.REQUEST, MissionState.RUNNING, MissionState.DONE):
            return  # the lane is not judged while changing lanes
        if self.unaligned_since is not None:
            return  # not parallel (corner): keep the lane and the frame count as they are
        lane = None
        if mine >= self.drivable_min:
            lane = 2 if left and right else 1 if right else 3 if left else None
        if lane is None:
            self.cand, self.cand_count = None, 0
            return
        self.cand_count = self.cand_count + 1 if lane == self.cand else 1
        self.cand = lane
        if self.cand_count < self.lane_frames:
            return
        if self.verify is not None:
            self.verify_seen = lane       # judged by step_state()
        elif lane != self.lane:
            self.set_lane(lane)

    def section_lost(self, now):
        """No side lane for zone_exit, counted from the last sighting or the last change."""
        pause = None   # a corner hides the side lanes: not-aligned time does not count
        if self.unaligned_at is not None and self.unaligned_since is not None \
                and now - self.unaligned_since <= self.align_pause_max:
            pause = self.unaligned_at
        ref = max(t for t in (self.side_seen_at, self.zone_grace_from, pause, float('-inf'))
                  if t is not None)
        return now - ref > self.zone_exit

    def leave_section(self, why):
        """Out of the 3-lane section: forget the lane, the retries and the speed limit."""
        self.lane = None
        self.verify = None
        self.cand, self.cand_count = None, 0
        self.retries, self.gave_up = 0, False
        self.zone_grace_from = None
        self.home, self.home_pending, self.sign_counts = 2, False, {1: 0, 3: 0}
        self.sign_seen_at = None
        if self.state != MissionState.IDLE:
            self.set_state(MissionState.IDLE, why)
        else:
            self.get_logger().info(f'section over: {why}')
            self.reason = why

    def reset_lidar_timers(self):
        """The lane 2 clear time and the obstacle count restart (lane changed / re-armed)."""
        self.side_clear_since = {LEFT: None, RIGHT: None}
        self.obstacle_count = 0

    def set_lane(self, lane):
        old, self.lane = self.lane, lane
        self.reset_lidar_timers()
        if self.state == MissionState.IDLE and not self.in_cooldown(self.now()):
            self.set_state(MissionState.ARMED, f'3-lane section, lane {lane}')
        elif self.state == MissionState.ARMED:
            self.get_logger().info(f'lane {old} -> {lane}')
            self.reason = f'lane {lane}'

    # ------------------------------------------------------------------ lidar
    def scan_callback(self, msg):
        x, y = self.scan_points(msg)
        lane_box = {}
        for side, sign in ((LEFT, 1.0), (RIGHT, -1.0)):
            sel = ((x > self.side_x[0]) & (x < self.side_x[1])
                   & (np.abs(y - sign * self.side_offset) < self.side_half))
            lane_box[side] = int(np.count_nonzero(sel)) >= self.side_min_points
        front = (x > self.front_x) & (x < self.front_x + self.obstacle_dist) & \
            (np.abs(y) < self.front_half)
        self.dbg_pts, self.dbg_box, self.dbg_front = (x, y), lane_box, bool(front.any())
        if front.any():
            self.front_oy = float(np.mean(y[front]))
        self.front_scans = self.front_scans + 1 if front.any() else 0
        self.check_guard(x, y)
        checking = self.verify is not None and self.state == MissionState.IDLE
        if (self.state != MissionState.ARMED or self.lane is None) and not checking:
            self.obstacle_count = 0
            return
        now = self.now()
        for side in (LEFT, RIGHT):
            if lane_box[side]:
                self.side_clear_since[side] = None
            elif self.side_clear_since[side] is None:
                self.side_clear_since[side] = now
        self.side_blocked = lane_box
        if checking:
            # an object ahead before the check is over: take over, stop, align, judge the
            # lane standing, then avoid (no guessing which lane the change ended in)
            self.obstacle_count = self.obstacle_count + 1 if front.any() else 0
            if self.obstacle_count >= self.obstacle_scans:
                v = self.verify
                self.verify = None
                self.cand, self.cand_count = None, 0
                self.recheck = (v['orig'], v['target'], v['side'])
                self.request = (RECHECK, None)
                self.request_at = now
                self.set_state(MissionState.REQUEST,
                               f'checking: object ahead -> stop and judge the lane '
                               f'(expected {v["target"]})')
            return

        step = self.step_toward_home()
        if step is not None:
            side, target = step
            if self.home_pending:   # sign just decided: blind, no clear wait
                self.request_now(side, target, f'home lane {self.home} (sign): lane '
                                               f'{self.lane} -> {target} at once')
                return
            if self.home == 2 and self.sign_seen_at is not None \
                    and now - self.sign_seen_at <= self.sign_hold:
                # a direction sign is being counted: the home lane may change, do not go yet
                sc = self.sign_counts
                self.reason = f'waiting for the sign decision (A {sc[1]} / B {sc[3]})'
                return
            # back to the home lane once its box stays empty (passed the dodged object)
            since = self.side_clear_since[side]
            if since is not None and now - since >= self.return_clear:
                self.make_request(side, target, f'lane {self.lane}: lane {target} clear '
                                                f'{self.return_clear:.1f} s (home {self.home})')
            return
        self.home_pending = False

        self.obstacle_count = self.obstacle_count + 1 if front.any() else 0
        if self.obstacle_count < self.obstacle_scans:
            return
        oy = float(np.mean(y[front]))
        dodge = self.avoid_from_home(oy)
        if dodge is None:
            self.reason = f'lane {self.lane}: object ahead, no free side, waiting'
            return
        side, target = dodge
        self.make_request(side, target, f'lane {self.lane}: object at x '
                                        f'{float(np.min(x[front])):.2f} y {oy:+.2f} '
                                        f'(home {self.home})')

    def check_guard(self, x, y):
        """
        Safety inside the section: an object closer than guard_dist to the lidar (x measured
        from the lidar) within +-guard_half_width, guard_min_points points for guard_scans
        scans, while moving forward -> back off. ARMED / IDLE inside the section (cooldown, result
        check): ask the master for [3road] (the master drives lane follow there); RUNNING:
        abort the lane change right away.
        """
        lx = self.laser[0]
        y0, y1 = -self.guard_half, self.guard_half
        if self.state == MissionState.RUNNING and self.phase in (TURN, COUNTER) \
                and self.request is not None and self.request[0] in (LEFT, RIGHT):
            # the front corner sweeps out to the side the car moves to: cover it too
            if self.request[0] == LEFT:
                y1 += self.guard_turn_extra
            else:
                y0 -= self.guard_turn_extra
        self.guard_y = (y0, y1)
        sel = (x > self.front_x - 0.02) & (y > y0) & (y < y1)
        self.guard_near = float(np.min(x[sel]) - lx) if sel.any() else None
        close = sel & (x - lx < self.guard_dist)
        self.guard_count = self.guard_count + 1 \
            if int(np.count_nonzero(close)) >= self.guard_min_points else 0
        if self.guard_count < self.guard_scans or self.backoff is not None:
            return
        why = f'guard: object {self.guard_near:.2f} m from the lidar'
        now = self.now()
        in_section = self.lane is not None or self.verify is not None
        idle_in_section = self.state == MissionState.IDLE and in_section  # e.g. cooldown
        if (self.state == MissionState.ARMED or idle_in_section) and self.cmd_v > 0.0:
            self.request = (GUARD, None)
            self.request_at = now
            self.set_state(MissionState.REQUEST, f'{why} -> back off')
        elif self.state == MissionState.RUNNING and self.phase in (TURN, COUNTER) \
                and self.request is not None and self.request[0] != GUARD:
            side, target = self.request
            self.start_backoff(now, (self.lane, target, side), f'{why}, lane change aborted')

    def avoid_move(self):
        """
        (side, target) to avoid from self.lane, None when nothing is possible (no waiting:
        the obstacles are fixed, waiting does not help; the guard protects the change).
          not in the home lane -> one lane towards it (home = 2, or 1 / 3 from the signs)
          in the home lane     -> dodge: home 2 -> the free side (both free: away from the
                                  object, y ~ 0 -> right); home 1 / 3 -> lane 2 (even if
                                  its box looks blocked)
        """
        if self.gave_up or self.lane not in (1, 2, 3):
            return None
        if self.lane != self.home:
            return self.step_toward_home()
        move = self.avoid_from_home(self.front_oy)
        if move is None and self.home != 2:
            move = (RIGHT, 2) if self.home == 1 else (LEFT, 2)
        return move

    def start_backoff(self, now, change, why, origin=GUARD):
        self.backoff = dict(start=now, change=change, origin=origin)
        self.settle = None
        self.set_phase(BACKOFF, why)

    # ------------------------------------------------------------------ settle
    def start_settle(self, now, origin, change, why, backed=False):
        """
        stop -> align -> judge -> decide, all inside [3road]. origin: GUARD (after a guard
        backoff), RECHECK (object ahead during a moving result check), VERIFY (right after a
        lane change). An object closer than backoff_clear_dist to the lidar and no backoff yet:
        back off straight first (a lane change from standstill needs room to get past it).
        """
        if not backed and self.guard_near is not None and self.guard_near < self.backoff_clear:
            self.start_backoff(now, change, f'{why}: object {self.guard_near:.2f} m '
                                            f'< {self.backoff_clear:.2f} m, back off first',
                               origin=origin)
            return
        self.backoff = None
        hold = self.verify_stop if origin == VERIFY else self.settle_sec
        skip = self.verify_align_skip if origin == VERIFY else self.align_done
        self.settle = dict(origin=origin, change=change, since=now, hold=hold, skip=skip)
        self.publish(0.0, 0.0)
        self.set_phase(STOP, f'{why}, hold {hold:.1f} s')

    def yaw_known(self, now):
        return self.yaw_at is not None and now - self.yaw_at <= self.yaw_timeout \
            and all(math.isfinite(a) for a in self.yaw)

    def step_settle(self, now):
        st = self.settle
        age = now - st['since']
        if self.phase == STOP:
            self.publish(0.0, 0.0)
            if age < st['hold']:
                return
            if not self.yaw_known(now):
                self.start_judge(now, 'no lane heading, align skipped')
            elif all(abs(a) <= st['skip'] for a in self.yaw):
                self.start_judge(now, f'parallel (yaw {self.yaw_text()})')
            else:
                st['since'] = now
                self.set_phase(ALIGN, f'yaw {self.yaw_text()} -> within '
                                      f'{self.align_done:.0f} deg')
            return
        if self.phase == ALIGN:
            if not self.yaw_known(now):
                self.publish(0.0, 0.0)
                self.start_judge(now, 'lane heading lost while aligning')
            elif all(abs(a) <= self.align_done for a in self.yaw):
                self.publish(0.0, 0.0)
                self.start_judge(now, f'aligned in {age:.1f} s (yaw {self.yaw_text()})')
            elif age >= self.align_max_sec:
                self.publish(0.0, 0.0)
                self.start_judge(now, f'align timeout {age:.1f} s (yaw {self.yaw_text()})')
            else:
                # backing off: w = k * yaw turns the car parallel (yaw rate sign is the same
                # forward and backward)
                w = self.align_k * math.radians(self.yaw[0])
                self.publish(self.align_speed,
                             max(-self.align_w_max, min(self.align_w_max, w)))
            return
        if self.phase == JUDGE:
            self.publish(0.0, 0.0)
            if st.get('lane') is not None:
                self.decide_after_settle(now, st['lane'], f'lane {st["lane"]} '
                                                         f'x{self.lane_frames}')
            elif age >= self.judge_max:
                weak = st.get('weak')
                if weak is not None:
                    self.decide_after_settle(now, weak, f'lane {weak} (weak, last frame)')
                else:
                    self.decide_after_settle(now, None, 'no lane judged')

    def start_judge(self, now, why):
        st = self.settle
        st.update(since=now, cand=None, count=0, lane=None, weak=None)
        self.set_phase(JUDGE, why)

    def judge_frame(self, now, ratio):
        """One drivable-class frame while standing in JUDGE (no alignment gate: already
        aligned, or as good as it gets)."""
        st = self.settle
        if now - st['since'] < self.judge_discard or st.get('lane') is not None:
            return   # frames from before the stop (pipeline delay)
        left, mine, right = ratio
        if mine < self.drivable_min:
            st['cand'], st['count'] = None, 0
            return
        wl, wr = left >= self.weak_side, right >= self.weak_side
        weak = 2 if wl and wr else 1 if wr else 3 if wl else None
        if weak is not None:
            st['weak'] = weak
        sl, sr = left >= self.side_min, right >= self.side_min
        lane = 2 if sl and sr else 1 if sr else 3 if sl else None
        if lane is None:
            st['cand'], st['count'] = None, 0
            return
        st['count'] = st['count'] + 1 if lane == st['cand'] else 1
        st['cand'] = lane
        if st['count'] >= self.lane_frames:
            st['lane'] = lane

    def decide_after_settle(self, now, lane, how):
        st = self.settle
        if st['origin'] == VERIFY:
            self.decide_verify(now, lane, how)
            return
        self.settle = None
        change = st['change']
        if lane is None:
            lane = self.lane if self.lane is not None else (change[0] if change else None)
        self.lane = lane
        self.cand, self.cand_count = None, 0
        self.verify = None
        self.zone_grace_from = now
        self.reset_lidar_timers()
        note = ''
        if st['origin'] == RECHECK and change is not None:
            orig, target, _ = change
            if lane == target:
                self.retries = 0
            elif lane == orig:
                self.retries += 1
                note = f', change {orig} -> {target} fell short: retry {self.retries}/' \
                       f'{self.max_retries}'
                if self.retries > self.max_retries:
                    self.gave_up = True
                    note += ' (giving up)'
        why = f'judged {how}{note}'
        self.get_logger().info(f'settle: {why}')
        self.start_change(now, self.avoid_move(), why)

    def start_change(self, now, move, why):
        """Next lane change right away (still [3road]); None: DONE, lane follow takes over."""
        if move is None or self.gave_up:
            self.request = None
            self.finish(f'{why}: no lane change -> lane follow')
            return
        self.request = move
        side, target = move
        self.phase_until = now + self.times[side][0]
        self.set_phase(TURN, f'{why} -> {side} to lane {target}, {self.times[side][0]:.1f} s')

    def decide_verify(self, now, lane, how):
        """
        Standing result check right after a lane change (orig, target, side):
          target lane, front clear   -> ok, DONE (lane follow)
          target lane, object ahead  -> avoid from it at once
          original lane (fell short) -> retry the same change at once (max_retries)
          another lane               -> take it; object ahead: avoid, else DONE
          nothing judged             -> DONE with the old moving result check
        """
        st = self.settle
        self.settle = None
        orig, target, side = st['change']
        self.cand, self.cand_count = None, 0
        self.zone_grace_from = now
        self.reset_lidar_timers()
        if lane is None:
            self.verify = dict(orig=orig, target=target, side=side, done_at=now)
            self.verify_seen = None
            self.get_logger().info('verify: no lane judged, checking while driving')
            self.finish(f'lane {orig} -> {target} ({side}), no lane judged standing, checking')
            return
        self.verify = None
        self.lane = lane
        blocked = self.front_scans >= self.obstacle_scans
        if lane == orig and lane != target:
            self.retries += 1
            why = f'still lane {lane} ({how}), retry {self.retries}/{self.max_retries}'
            if self.retries > self.max_retries:
                self.gave_up = True
                why += ': giving up'
            self.get_logger().info(f'verify: {why}')
            self.start_change(now, (side, target), why)
            return
        if lane == target:
            self.retries = 0
            self.get_logger().info(f'lane change ok: lane {orig} -> {lane} ({how})')
        else:
            self.get_logger().info(f'verify: lane {lane} ({how}), expected {target}')
        if blocked:
            self.start_change(now, self.avoid_move(), f'lane {lane}: object ahead after the '
                                                      f'change')
            return
        self.request = None
        self.finish(f'lane change ok: lane {lane}' if lane == target else
                    f'lane {lane} after the change (expected {target})')

    def scan_points(self, msg):
        r = np.asarray(msg.ranges, dtype=np.float64)
        a = msg.angle_min + np.arange(len(r)) * msg.angle_increment
        ok = np.isfinite(r) & (r > msg.range_min) & (r < msg.range_max)
        lx, ly, lyaw = self.laser
        a, r = a[ok] + lyaw, r[ok]
        return r * np.cos(a) + lx, r * np.sin(a) + ly

    def request_now(self, side, target, why):
        """Lane change request without the alignment / side-lane checks (blind moves)."""
        if self.gave_up:
            self.reason = f'gave up after {self.max_retries} retries: {why} (ignored)'
            return
        self.request = (side, target)
        self.request_at = self.now()
        self.set_state(MissionState.REQUEST, why)

    def make_request(self, side, target, why):
        now = self.now()
        if not self.aligned(now):
            self.reason = f'not aligned (yaw {self.yaw_text()}): {why} (held)'
            return
        if self.side_seen_at is None or now - self.side_seen_at > self.side_fresh:
            self.reason = f'no side lane in the last {self.side_fresh:.1f} s: {why} (held)'
            return  # the camera does not confirm the section right now
        if self.gave_up:
            self.reason = f'gave up after {self.max_retries} retries: {why} (ignored)'
            return
        self.request = (side, target)
        self.request_at = self.now()
        self.set_state(MissionState.REQUEST, f'{why} -> {side} to lane {target}')

    # ------------------------------------------------------------ lane change
    def on_start(self, now):
        if self.request is not None and self.request[0] == GUARD:
            self.start_backoff(now, None, 'guard backoff')
            return
        if self.request is not None and self.request[0] == RECHECK:
            change, self.recheck = self.recheck, None
            self.start_settle(now, RECHECK, change, 'object ahead while checking')
            return
        if self.request is None:  # forced by /master/set_mode: towards lane 2 if known
            side = RIGHT if self.lane == 1 else LEFT if self.lane == 3 else RIGHT
            self.request = (side, {1: 2, 3: 2}.get(self.lane, 3))
        side, target = self.request
        self.phase_until = now + self.times[side][0]
        self.set_phase(TURN, f'{side} to lane {target}, {self.times[side][0]:.1f} s')

    def on_stop(self, now):
        self.request = None
        self.settle = None
        self.phase_until = None
        self.backoff = None
        self.guard_count = 0

    def tick(self, now):
        if now - self.last_debug >= self.debug_period and self.pub_debug.get_subscription_count():
            self.last_debug = now
            self.publish_debug(now)
        self.step_state(now)
        # inside the 3-lane section the master drives lane follow at section_speed
        in_section = (self.state in (MissionState.ARMED, MissionState.REQUEST)
                      or self.lane is not None or self.verify is not None)
        self.speed_limit = self.section_speed if in_section else 0.0

    def step_state(self, now):
        if self.verify is not None and self.state == MissionState.IDLE:
            self.check_result(now)
            return
        if self.state == MissionState.IDLE and self.lane is not None:
            # back from a lane change (lane already set): re-arm once the cooldown is over
            if not self.section_lost(now):
                if not self.in_cooldown(now):
                    self.set_state(MissionState.ARMED, f'3-lane section, lane {self.lane}')
            else:
                self.leave_section(f'no side lane for {self.zone_exit:.0f} s')
            return
        if self.state == MissionState.ARMED:
            if self.section_lost(now):
                self.leave_section(f'no side lane for {self.zone_exit:.0f} s')
            return
        if self.state == MissionState.REQUEST:
            if now - self.request_at > self.request_timeout:
                self.request = None
                self.set_state(MissionState.ARMED,
                               f'not granted within {self.request_timeout:.0f} s')
            return
        if not self.active or self.request is None:
            return
        if self.backoff is not None:
            self.step_backoff(now)
            return
        if self.settle is not None:
            self.step_settle(now)
            return
        side, target = self.request
        sign = 1.0 if side == LEFT else -1.0
        if self.state == MissionState.RUNNING:
            if now >= self.phase_until and self.phase == TURN:
                self.phase_until = now + self.times[side][1]
                self.set_phase(COUNTER, f'{self.times[side][1]:.1f} s')
            elif now >= self.phase_until and self.phase == COUNTER:
                # stop and judge standing (still [3road]); the lane is set by the check
                self.zone_grace_from = now
                self.start_settle(now, VERIFY, (self.lane, target, side),
                                  f'lane {self.lane} -> {target} ({side}) done, checking')
                return
        if self.state == MissionState.RUNNING:
            w = sign * self.w if self.phase == TURN else -sign * self.w
            self.publish(self.speed, w)
        else:
            self.publish(self.speed, 0.0)  # DONE: straight until the master takes over

    def step_backoff(self, now):
        """Back off straight for backoff_min..backoff_max s, until nothing within
        backoff_clear_dist, then settle (stop, align, judge) and avoid - still in [3road]."""
        b = self.backoff
        age = now - b['start']
        clear = self.guard_near is None or self.guard_near >= self.backoff_clear
        if (age >= self.backoff_min and clear) or age >= self.backoff_max:
            self.guard_count = 0
            why = f'backoff {age:.1f} s'
            if b['change'] is not None and b['origin'] == GUARD:
                why += f', aborted change {b["change"][0]} -> {b["change"][1]}'
            self.start_settle(now, b['origin'], b['change'], why, backed=True)
            return
        self.publish(self.backoff_speed, 0.0)

    def check_result(self, now):
        """
        After a lane change: wait verify_settle, then the stable lane of the next verify_sec.

        target lane -> success. Original lane (or no answer by the end) -> failed, e.g. the
        remote control held the car: back to ARMED at once (no cooldown) so the trigger is
        evaluated again; after max_retries failures in a row only lane follow until the
        section ends.
        """
        v = self.verify
        age = now - v['done_at']
        if age < self.verify_settle:
            v['last_t'] = now
            return
        if self.aligned(now):   # only aligned frames judge the lane: count that time
            v['aligned_time'] = v.get('aligned_time', 0.0) + now - v.get('last_t', now)
        v['last_t'] = now
        seen = self.verify_seen
        if seen == v['target']:
            self.verify = None
            self.retries = 0
            self.lane = v['target']
            self.cand, self.cand_count = seen, self.lane_frames
            self.get_logger().info(f'lane change ok: lane {v["orig"]} -> {seen}')
            self.reason = f'lane change ok: lane {seen}'
            return
        window_open = (v.get('aligned_time', 0.0) < self.verify_sec and
                       age < self.verify_settle + self.verify_sec + self.verify_extend_max)
        if seen != v['orig'] and window_open:
            return  # no clear answer yet (on the line / not aligned): keep looking
        self.verify = None
        if seen is None and (self.side_seen_at is None or self.side_seen_at < v['done_at']):
            # no side lane at all since the change: out of the 3-lane section, not a failure
            self.leave_section(f'no side lane since the change (expected lane {v["target"]})')
            return
        self.lane = seen if seen is not None else v['orig']
        self.retries += 1
        self.reset_lidar_timers()
        why = (f'lane {self.lane} after the change (expected {v["target"]})'
               if seen is not None else f'no lane judged (expected {v["target"]})')
        self.zone_grace_from = now
        self.cooldown_until = now   # retry right away
        if self.retries > self.max_retries:
            self.gave_up = True
            self.set_state(MissionState.ARMED, f'{why}: failed {self.retries}x, giving up '
                                               f'(lane follow only in this section)')
        else:
            self.set_state(MissionState.ARMED, f'{why}: retry {self.retries}/'
                                               f'{self.max_retries}')

    # ------------------------------------------------------------------ debug
    def publish_debug(self, now):
        """
        /three_road/debug (bgr8, 640x300) for rqt_image_view.

        left : drivable classes (my lane green, left sky, right purple), lane ROI line,
               pixel shares vs side_min_ratio, lane number / candidate
        right: lidar top view (car at the bottom, 1 grid = 0.2 m), front box and lane
               boxes (red = object, green = empty) with the lane 2 clear timer
        bottom: mission state / phase / reason, lane_follow tracking state
        """
        img = np.zeros((300, 640, 3), np.uint8)
        self.draw_classes(img[:240, :320])
        self.draw_lidar(img[:240, 320:], now)
        names = {0: 'IDLE', 1: 'ARMED', 2: 'REQUEST', 3: 'RUNNING', 4: 'DONE'}
        state = names.get(self.state, '?') + (f'/{self.phase}' if self.phase else '')
        cool = ' (cooldown)' if self.in_cooldown(now) else ''
        put(img, f'{state}{cool}  lane follow: {self.lane_state or "-"}', (8, 262),
            (0, 255, 255), 0.55)
        put(img, self.reason[:90], (8, 288), (220, 220, 220), 0.42)
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height, msg.width, msg.encoding, msg.step = 300, 640, 'bgr8', 640 * 3
        msg.data = array('B', img.tobytes())  # array: skips rclpy per-byte check
        self.pub_debug.publish(msg)

    def draw_classes(self, view):
        if self.dbg_cls is None:
            put(view, 'no /yolopx/drivable_class', (10, 120), (0, 0, 255), 0.5)
            return
        lut = np.zeros((256, 3), np.uint8)
        lut[1], lut[2], lut[3] = (0, 160, 0), (235, 206, 135), (200, 50, 160)
        view[:] = cv2.resize(lut[self.dbg_cls], (320, 240), interpolation=cv2.INTER_NEAREST)
        y_roi = int(240 * self.roi_top)
        cv2.line(view, (0, y_roi), (319, y_roi), (255, 255, 255), 1)
        lr, mr, rr = self.dbg_ratio
        for i, (name, r, col) in enumerate((('L', lr, (235, 206, 135)), ('M', mr, (0, 255, 0)),
                                            ('R', rr, (200, 50, 160)))):
            mark = '' if name == 'M' else (' ok' if r >= self.side_min else ' -')
            put(view, f'{name} {r:.2f}{mark}', (8 + 104 * i, 20), col, 0.5)
        lane = f'lane {self.lane}' if self.lane else 'lane -'
        put(view, f'{lane}  cand {self.cand or "-"} x{self.cand_count}  '
                  f'(min {self.side_min:.2f})', (8, 44), (255, 255, 255), 0.45)
        sc = self.sign_counts
        sign_txt = (f'A {sc[1]} / B {sc[3]} -> home {self.home}' if self.home != 2 else
                    f'A {sc[1]} / B {sc[3]} (need {self.sign_decide}, +{self.sign_margin})')
        put(view, sign_txt, (8, 154), (255, 200, 0), 0.45)
        al = self.aligned(self.now())
        put(view, f'{"aligned" if al else "NOT aligned"}  yaw {self.yaw_text()}', (8, 132),
            (0, 255, 0) if al else (0, 0, 255), 0.45)
        if self.verify is not None:
            put(view, f'checking: expect lane {self.verify["target"]}, seen '
                      f'{self.verify_seen or "-"}', (8, 66), (0, 255, 255), 0.45)
        st = self.settle
        if st is not None and self.phase == JUDGE:
            put(view, f'judge: cand {st.get("cand") or "-"} x{st.get("count", 0)}/'
                      f'{self.lane_frames}  weak {st.get("weak") or "-"}', (8, 66),
                (0, 255, 255), 0.45)
        elif st is not None:
            put(view, f'{self.phase} {self.now() - st["since"]:.1f}s  yaw {self.yaw_text()}'
                      f' (done {self.align_done:.0f})', (8, 66), (0, 255, 255), 0.45)
        if self.retries or self.gave_up:
            put(view, 'GAVE UP (lane follow only)' if self.gave_up else
                f'retry {self.retries}/{self.max_retries}', (8, 88), (0, 0, 255), 0.5)

    def draw_lidar(self, view, now):
        scale, ox, oy = 200.0, 160, 228               # px / m, car origin in the view

        def px(x, y):
            return int(ox - y * scale), int(oy - x * scale)

        for g in np.arange(0.2, 1.21, 0.2):           # 0.2 m grid
            cv2.line(view, px(g, -0.8), px(g, 0.8), (50, 50, 50), 1)
        for g in np.arange(-0.8, 0.81, 0.2):
            cv2.line(view, px(0.0, g), px(1.2, g), (50, 50, 50), 1)
        boxes = [('front', self.front_x, self.front_x + self.obstacle_dist, -self.front_half,
                  self.front_half, self.dbg_front)]
        for side, sign in ((LEFT, 1.0), (RIGHT, -1.0)):
            c = sign * self.side_offset
            boxes.append((side, self.side_x[0], self.side_x[1], c - self.side_half,
                          c + self.side_half, self.dbg_box[side]))
        for name, x0, x1, y0, y1, hit in boxes:
            col = (0, 0, 255) if hit else (0, 200, 0)
            cv2.rectangle(view, px(x1, y1), px(x0, y0), col, 2 if hit else 1)
            label = name
            if (name != 'front' and self.state == MissionState.ARMED
                    and self.side_clear_since[name] is not None):  # clear timer: ARMED only
                label += f' {now - self.side_clear_since[name]:.1f}s'
            put(view, label, (px(x1, y1)[0] + 3, px(x1, y1)[1] + 14), col, 0.4)
        gx = self.laser[0] + self.guard_dist          # guard box (orange)
        gcol = (0, 0, 255) if self.guard_count else (0, 140, 255)
        cv2.rectangle(view, px(gx, self.guard_y[1]), px(self.front_x, self.guard_y[0]), gcol,
                      2 if self.guard_count else 1)
        near = f'{self.guard_near:.2f} m' if self.guard_near is not None else '-'
        put(view, f'guard {near}', (200, 236), gcol, 0.45)
        x, y = self.dbg_pts
        sel = (x > -0.1) & (x < 1.25) & (np.abs(y) < 0.85)
        for xi, yi in zip(x[sel], y[sel]):
            cv2.circle(view, px(xi, yi), 2, (255, 255, 255), -1)
        cv2.rectangle(view, px(self.front_x, 0.1), px(-0.16, -0.1), (0, 255, 255), 1)  # car
        put(view, f'obstacle x{self.obstacle_count}/{self.obstacle_scans}', (6, 236),
            (0, 0, 255) if self.dbg_front else (180, 180, 180), 0.45)

    def yaw_text(self):
        n, a = self.yaw
        if not (math.isfinite(n) and math.isfinite(a)):
            return '-'
        return f'{n:+.0f}/{a:+.0f} deg'

    def publish(self, v, w):
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = float(v), float(w)
        self.pub_cmd.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = ThreeRoadNode()
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
