"""
3-lane (3road) mission node, v2 (simple loop). Mode '3road', topics 'three_road'.

1. Lane number: only while the car is parallel to the lane (lane_follow /lane_follow/yaw
   near AND ahead within align_max_deg) and not changing lanes. /yolopx/drivable_class
   pixel counts (rows below roi_top_ratio): a side lane counts with >= side_min_pixels.
     left + right -> 2,  right only -> 1,  left only -> 3;  lane_frames frames in a row.
   Inside the section a jump 1 <-> 3 without a lane change is ignored (misread).
2. Section: entered when lane 2 is judged (all three lanes seen) -> ARMED; the master caps
   lane follow at section_speed (speed_limit). Over after no side lane for zone_exit_sec.
3. Obstacles per lane (lidar, base_link x forward / y left):
     my lane     : front box (|y| < front_half_width, up to dodge_dist2 from the lidar),
                   obstacle_scans scans in a row; the nearest point decides: a dodge that
                   ends one lane away needs it closer than dodge_dist, two lanes away
                   closer than dodge_dist2 (else my lane counts as free)
     next lanes  : lane boxes side_x_min..side_x_max along the lane (rotated by the lane
                   heading /lane_follow/yaw[0] when <= align_trust_deg), centred on the lane
                   centre (/lane_follow/yaw[2]) +- side_offset, >= side_min_points points
     lanes not seen (two lanes away) count as free (three blocked lanes do not happen)
4. Decision loop (ARMED, parallel, lane known), home lane = 2 (or 1 / 3 from the signs):
     my lane blocked         -> the free lane nearest to home (lane 2 with both free and home
                                2: away from the object, y ~ 0 -> right); two lanes at once
                                if needed
     my lane free, not home  -> home when the lanes on the way and home are free (1 -> 3 at
                                once when lane 2 is free), else stay
     else                    -> stay
5. Signs: sign_A / sign_B frames inside the section; once both together reach
   sign_decide_frames, the one seen more often sets home (A -> 1, B -> 3) for the section.
6. Lane change ([3road], the same for every change):
     pre_stop  hold pre_stop_sec, decide again (step 4; DONE when nothing to do), then the
               object ahead: closer to the lidar than backoff_clear_dist (one lane) /
               backoff_clear_dist2 (two lanes) -> back off straight until it is not, hold
               again, decide again; else start. The checks are shorter than the dodge
               distances, so a backed-off object still counts as ahead (no loop)
     turn / (mid) / counter   open loop, measured times (one or two lanes); turn AND counter
               get turn_bias_k x the sideways offset from the lane centre averaged over
               pre_stop (toward the target side: shorter, away: longer; <= turn_bias_max),
               the same time on both so the car still ends parallel; a two-lane change
               biases only mid instead (mid_bias_k, <= mid_bias_max)
     post_stop hold post_stop_sec, then judge standing
     judge     parallel frames only: lane_frames in a row -> lane set, decide again at once
               (next change, or DONE); my lane only -> lane unknown, DONE; nothing within
               judge_max_sec -> lane unknown, DONE (lane follow straightens the car and the
               normal judgement sets the lane)
7. Guard (last resort): an object closer than guard_dist to the lidar while moving forward.
     ARMED / in the section -> [3road]: back off straight, then pre_stop (decide again)
     during turn / mid / counter -> abort, back off straight, then align (back off with
               w = align_k * yaw while |yaw| <= align_trust_deg, until align_done_deg or
               align_max_sec), judge standing; not judged -> the lane before the change
"""
from array import array
import math
import signal

import cv2
from geometry_msgs.msg import Twist
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Float32MultiArray, String
from trinity_msgs.msg import MissionState
from vision_msgs.msg import Detection2DArray

from trinity_missions.mission_base import MissionNode

PRE_STOP, BACKOFF, TURN, MID, COUNTER = 'pre_stop', 'backoff', 'turn', 'mid', 'counter'
POST_STOP, ALIGN, JUDGE = 'post_stop', 'align', 'judge'
GUARD = 'guard'                   # request kind: guard backoff
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
        d('cmd_vel_topic', '/cmd_vel')
        d('detections_topic', '/yolopx/detections')
        d('yaw_topic', '/lane_follow/yaw')
        d('lane_state_topic', '/lane_follow/state')
        d('debug_topic', '/three_road/debug')
        d('debug_rate_hz', 10.0)
        # 1./2. lane number, section
        d('roi_top_ratio', 0.0)
        d('side_min_pixels', 1500)
        d('drivable_min_pixels', 9000)
        d('lane_frames', 3)
        d('align_max_deg', 10.0)
        d('yaw_timeout', 0.5)
        d('entry_lane2_only', True)
        d('reenter_grace_sec', 0.0)
        d('zone_exit_sec', 4.0)
        d('align_pause_max', 5.0)
        d('section_speed', 0.2)
        # 3. lidar
        d('laser_x', 0.103)
        d('laser_y', 0.0)
        d('laser_yaw', 0.0)
        d('front_x', 0.16)
        d('dodge_dist', 0.7)             # m from the lidar: my lane blocked -> one-lane dodge
        d('dodge_dist2', 0.8)            # ... -> two-lane dodge
        d('front_half_width', 0.15)
        d('obstacle_scans', 3)
        d('side_offset', 0.5)
        d('side_half_width', 0.2)
        d('side_x_min', 0.3)
        d('side_x_max', 1.5)
        d('side_min_points', 2)
        d('box_follow_lane', True)
        d('box_shift_max', 0.25)
        d('center_tol', 0.03)
        d('request_timeout', 1.0)
        # 5. signs
        d('sign_left_class', 'sign_A')
        d('sign_right_class', 'sign_B')
        d('sign_conf', 0.5)
        d('sign_decide_frames', 20)
        # 6. lane change
        d('change_speed', 0.2)
        d('steer', 0.983)
        d('wheelbase', 0.2)
        d('left_turn_sec', 2.4)
        d('left_counter_sec', 3.0)
        d('right_turn_sec', 3.1)
        d('right_counter_sec', 2.9)
        d('left2_turn_sec', 2.4)
        d('left2_mid_sec', 3.0)
        d('left2_counter_sec', 3.0)
        d('right2_turn_sec', 3.1)
        d('right2_mid_sec', 2.7)
        d('right2_counter_sec', 3.0)
        d('pre_stop_sec', 0.5)
        # turn time bias from the lane centre offset measured during pre_stop
        d('turn_bias_k', 2.0)            # s per m of extra sideways travel
        d('turn_bias_max', 0.5)          # s, |bias| limit
        d('mid_bias_k', 6.5)             # two-lane change: s of mid per m (0.2 m/s x sin 50 deg)
        d('mid_bias_max', 0.8)           # s
        d('post_stop_sec', 0.5)
        d('judge_max_sec', 1.0)
        d('judge_discard_sec', 0.2)
        d('backoff_speed', -0.2)
        d('backoff_min_sec', 0.5)
        d('backoff_max_sec', 5.0)
        d('backoff_clear_dist', 0.6)     # one-lane change: closer -> back off to this
        d('backoff_clear_dist2', 0.75)   # two-lane change
        # 7. guard / align
        d('guard_dist', 0.13)
        d('guard_half_width', 0.15)
        d('guard_turn_extra', 0.10)
        d('guard_min_points', 2)
        d('guard_scans', 2)
        d('align_done_deg', 5.0)
        d('align_trust_deg', 30.0)
        d('align_speed', -0.15)
        d('align_k', 1.5)
        d('align_max_sec', 3.0)

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.debug_period = 1.0 / max(float(p('debug_rate_hz')), 0.1)
        self.roi_top = float(p('roi_top_ratio'))
        self.side_min = int(p('side_min_pixels'))
        self.drivable_min = int(p('drivable_min_pixels'))
        self.lane_frames = int(p('lane_frames'))
        self.align_max = float(p('align_max_deg'))
        self.yaw_timeout = float(p('yaw_timeout'))
        self.entry_lane2_only = bool(p('entry_lane2_only'))
        self.reenter_grace = float(p('reenter_grace_sec'))
        self.zone_exit = float(p('zone_exit_sec'))
        self.align_pause_max = float(p('align_pause_max'))
        self.section_speed = float(p('section_speed'))
        self.laser = (float(p('laser_x')), float(p('laser_y')), float(p('laser_yaw')))
        self.front_x = float(p('front_x'))
        self.dodge = float(p('dodge_dist'))
        self.dodge2 = float(p('dodge_dist2'))
        self.front_half = float(p('front_half_width'))
        self.obstacle_scans = int(p('obstacle_scans'))
        self.side_offset = float(p('side_offset'))
        self.side_half = float(p('side_half_width'))
        self.side_x = (float(p('side_x_min')), float(p('side_x_max')))
        self.side_min_points = int(p('side_min_points'))
        self.box_follow_lane = bool(p('box_follow_lane'))
        self.box_shift_max = float(p('box_shift_max'))
        self.center_tol = float(p('center_tol'))
        self.request_timeout = float(p('request_timeout'))
        self.sign_classes = {p('sign_left_class'): 1, p('sign_right_class'): 3}
        self.sign_conf = float(p('sign_conf'))
        self.sign_decide = int(p('sign_decide_frames'))
        self.speed = float(p('change_speed'))
        self.w = self.speed * math.tan(float(p('steer'))) / float(p('wheelbase'))
        self.times = {LEFT: (float(p('left_turn_sec')), float(p('left_counter_sec'))),
                      RIGHT: (float(p('right_turn_sec')), float(p('right_counter_sec')))}
        self.times2 = {sd: tuple(float(p(f'{sd}2_{k}_sec')) for k in ('turn', 'mid', 'counter'))
                       for sd in (LEFT, RIGHT)}
        self.pre_stop = float(p('pre_stop_sec'))
        self.turn_bias_k = float(p('turn_bias_k'))
        self.turn_bias_max = float(p('turn_bias_max'))
        self.mid_bias_k = float(p('mid_bias_k'))
        self.mid_bias_max = float(p('mid_bias_max'))
        self.centre_samples = []     # lane centre y during pre_stop (turn bias)
        self.post_stop = float(p('post_stop_sec'))
        self.judge_max = float(p('judge_max_sec'))
        self.judge_discard = float(p('judge_discard_sec'))
        self.backoff_speed = float(p('backoff_speed'))
        self.backoff_min = float(p('backoff_min_sec'))
        self.backoff_max = float(p('backoff_max_sec'))
        self.backoff_clear = float(p('backoff_clear_dist'))
        self.backoff_clear2 = float(p('backoff_clear_dist2'))
        self.backoff_need = self.backoff_clear   # clearance the running backoff aims for
        self.guard_dist = float(p('guard_dist'))
        self.guard_half = float(p('guard_half_width'))
        self.guard_turn_extra = float(p('guard_turn_extra'))
        self.guard_min_points = int(p('guard_min_points'))
        self.guard_scans = int(p('guard_scans'))
        self.align_done = float(p('align_done_deg'))
        self.align_trust = float(p('align_trust_deg'))
        self.align_speed = float(p('align_speed'))
        self.align_k = float(p('align_k'))
        self.align_max_sec = float(p('align_max_sec'))
        self.align_w_max = abs(self.align_speed) * math.tan(float(p('steer'))) / \
            float(p('wheelbase'))

        # section / lane
        self.lane = None             # 1 / 2 / 3, None outside the section or while unknown
        self.lane_unknown = False    # inside the section, lane lost: wait for a judgement
        self.cand, self.cand_count = None, 0
        self.side_seen_at = None     # s, last frame with a side lane
        self.zone_grace_from = None  # s, section exit timer starts no earlier than this
        self.unaligned_since = None
        self.unaligned_at = None
        self.left_section_at = None
        self.home = 2
        self.sign_counts = {1: 0, 3: 0}
        # lidar
        self.box = {LEFT: False, RIGHT: False}   # lane boxes blocked (this scan)
        self.front_scans = 0         # scans in a row with an object in the front box
        self.front_near = None       # m from the lidar, nearest point in the front box
        self.front_oy = 0.0          # m, mean y of the front box points (last time seen)
        self.guard_count = 0
        self.guard_near = None       # m, nearest object ahead (from the lidar) in the guard width
        self.guard_y = (-self.guard_half, self.guard_half)
        self.cmd_v = 0.0
        # lane change
        self.request = None          # (side, target) or (GUARD, None)
        self.request_at = None
        self.plan = []               # remaining (phase, sec) steps of the change
        self.phase_until = None
        self.phase_since = None
        self.after_backoff = None    # PRE_STOP or ALIGN: what follows the running backoff
        self.orig_lane = None        # lane before the running change (guard fallback)
        self.judge = None            # standing judgement: dict(cand, count, single, origin)
        self.decision = '-'          # last decision text (debug)
        # inputs / debug
        self.yaw = (float('nan'), float('nan'))
        self.yaw_at = None
        self.centre_y = float('nan')
        self.lane_state = ''
        self.dbg_cls = None
        self.dbg_px = (0, 0, 0)
        self.dbg_pts = (np.zeros(0), np.zeros(0))
        self.dbg_shift = 0.0
        self.dbg_angle = 0.0
        self.last_debug = 0.0

        self.create_subscription(Image, p('drivable_class_topic'), self.drivable_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(LaserScan, p('scan_topic'), self.scan_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(Detection2DArray, p('detections_topic'), self.signs_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(Float32MultiArray, p('yaw_topic'), self.yaw_callback, 10)
        self.create_subscription(String, p('lane_state_topic'),
                                 lambda m: setattr(self, 'lane_state', m.data), 10)
        self.create_subscription(Twist, p('cmd_vel_topic'),
                                 lambda m: setattr(self, 'cmd_v', m.linear.x), 10)
        self.pub_cmd = self.create_publisher(Twist, p('cmd_topic'), 10)
        self.pub_debug = self.create_publisher(Image, p('debug_topic'), 1)
        self.get_logger().info(
            f'ready (v2): enter on lane 2, home 2 / signs x{self.sign_decide} | change '
            f'{self.speed:.2f} m/s w {self.w:.2f}, left {self.times[LEFT]}, right '
            f'{self.times[RIGHT]} | stop {self.pre_stop:.1f} s + ahead >= '
            f'{self.backoff_clear:.2f} / {self.backoff_clear2:.2f} m (1 / 2 lanes) before '
            f'every change')

    # ============================================================== inputs
    def yaw_callback(self, msg):
        if len(msg.data) >= 2:
            self.yaw, self.yaw_at = (float(msg.data[0]), float(msg.data[1])), self.now()
            self.centre_y = float(msg.data[2]) if len(msg.data) >= 3 else float('nan')

    def yaw_known(self, now):
        return self.yaw_at is not None and now - self.yaw_at <= self.yaw_timeout \
            and all(math.isfinite(a) for a in self.yaw)

    def aligned(self, now, limit=None):
        lim = self.align_max if limit is None else limit
        return self.yaw_known(now) and all(abs(a) <= lim for a in self.yaw)

    def in_section(self):
        return self.lane is not None or self.lane_unknown or \
            self.state in (MissionState.ARMED, MissionState.REQUEST, MissionState.RUNNING)

    def signs_callback(self, msg):
        if not self.in_section() or self.home != 2:
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
        self.sign_counts[best[0]] += 1
        a, b = self.sign_counts[1], self.sign_counts[3]
        if a + b >= self.sign_decide and a != b:
            self.home = 1 if a > b else 3
            self.get_logger().info(f'signs A {a} / B {b} -> home lane {self.home}')

    # --------------------------------------------------------- lane number
    def drivable_callback(self, msg):
        cls = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width]
        roi = cls[int(msg.height * self.roi_top):]
        left, mine, right = (int(np.count_nonzero(roi == c)) for c in (2, 1, 3))
        self.dbg_cls, self.dbg_px = cls[::2, ::2], (left, mine, right)
        now = self.now()
        sl, sr = left >= self.side_min, right >= self.side_min
        if sl or sr:
            self.side_seen_at = now
        al = self.aligned(now)
        if al:
            self.unaligned_since = None
        else:
            self.unaligned_since = self.unaligned_since or now
            self.unaligned_at = now
        lane = (2 if sl and sr else 1 if sr else 3 if sl else None) \
            if mine >= self.drivable_min else None
        if self.judge is not None:              # standing judgement (RUNNING / judge)
            self.judge_frame(now, al, lane, mine)
            return
        if self.state in (MissionState.REQUEST, MissionState.RUNNING, MissionState.DONE):
            return
        if not al:
            return                              # not parallel: keep lane and frame count
        if lane is None:
            self.cand, self.cand_count = None, 0
            return
        self.cand_count = self.cand_count + 1 if lane == self.cand else 1
        self.cand = lane
        if self.cand_count < self.lane_frames or lane == self.lane:
            return
        if self.lane is not None and abs(lane - self.lane) == 2:
            self.reason = f'lane {self.lane}: lane {lane} judged, ignored (two lanes away)'
        elif self.lane is None and not self.lane_unknown and lane != 2 \
                and self.entry_lane2_only and not self.in_reenter_grace(now):
            self.reason = f'lane {lane} judged, entry needs lane 2'
        else:
            self.set_lane(lane)

    def set_lane(self, lane):
        old, self.lane = self.lane, lane
        self.lane_unknown = False
        if self.state == MissionState.IDLE and not self.in_cooldown(self.now()):
            self.set_state(MissionState.ARMED, f'3-lane section, lane {lane}')
        elif old is not None:
            self.get_logger().info(f'lane {old} -> {lane}')

    def in_reenter_grace(self, now):
        return self.left_section_at is not None and \
            now - self.left_section_at <= self.reenter_grace

    def section_lost(self, now):
        pause = None   # a corner hides the side lanes: not-aligned time does not count
        if self.unaligned_at is not None and self.unaligned_since is not None \
                and now - self.unaligned_since <= self.align_pause_max:
            pause = self.unaligned_at
        ref = max(t for t in (self.side_seen_at, self.zone_grace_from, pause, float('-inf'))
                  if t is not None)
        return now - ref > self.zone_exit

    def leave_section(self, why):
        self.lane, self.lane_unknown = None, False
        self.cand, self.cand_count = None, 0
        self.zone_grace_from = None
        self.home, self.sign_counts = 2, {1: 0, 3: 0}
        self.left_section_at = self.now()
        if self.state != MissionState.IDLE:
            self.set_state(MissionState.IDLE, why)
        else:
            self.get_logger().info(f'section over: {why}')
            self.reason = why

    # =============================================================== lidar
    def scan_points(self, msg):
        r = np.asarray(msg.ranges, dtype=np.float64)
        a = msg.angle_min + np.arange(len(r)) * msg.angle_increment
        ok = np.isfinite(r) & (r > msg.range_min) & (r < msg.range_max)
        lx, ly, lyaw = self.laser
        a, r = a[ok] + lyaw, r[ok]
        return r * np.cos(a) + lx, r * np.sin(a) + ly

    def box_shift(self):
        """Lane centre y from the car (+ left) the lane boxes are centred on."""
        c = self.centre_y
        if not self.box_follow_lane or self.yaw_at is None or \
                self.now() - self.yaw_at > self.yaw_timeout or not math.isfinite(c) or \
                abs(c) > self.box_shift_max:
            return 0.0
        return c

    def box_angle(self):
        """Lane direction from the car (rad, + left): lane_follow's near heading when fresh
        and trusted (<= align_trust_deg), else 0. The lane boxes are laid along it, so a
        car a few degrees off does not shift the far end of a box (10 deg = 35 cm at 2 m)."""
        if not self.box_follow_lane or not self.yaw_known(self.now()) or \
                abs(self.yaw[0]) > self.align_trust:
            return 0.0
        return math.radians(self.yaw[0])

    def scan_callback(self, msg):
        x, y = self.scan_points(msg)
        self.dbg_pts = (x, y)
        self.dbg_shift = shift = self.box_shift()
        self.dbg_angle = th = self.box_angle()
        u = x * math.cos(th) + y * math.sin(th)       # along the lane
        v = -x * math.sin(th) + y * math.cos(th)      # across the lane (+ left)
        for side, sign in ((LEFT, 1.0), (RIGHT, -1.0)):
            sel = ((u > self.side_x[0]) & (u < self.side_x[1])
                   & (np.abs(v - (shift + sign * self.side_offset)) < self.side_half))
            self.box[side] = int(np.count_nonzero(sel)) >= self.side_min_points
        reach = self.laser[0] + max(self.dodge, self.dodge2)
        front = (x > self.front_x) & (x < reach) & \
            (np.abs(y) < self.front_half)
        if front.any():
            self.front_oy = float(np.mean(y[front]))
            self.front_near = float(np.min(x[front]) - self.laser[0])
        else:
            self.front_near = None
        self.front_scans = self.front_scans + 1 if front.any() else 0
        self.check_guard(x, y)
        if self.state == MissionState.ARMED and self.lane is not None:
            self.loop()

    def side_blocked(self):
        """Next lanes with an object in their lane box."""
        out = set()
        if self.box[LEFT] and self.lane > 1:
            out.add(self.lane - 1)
        if self.box[RIGHT] and self.lane < 3:
            out.add(self.lane + 1)
        return out

    def dodge_target(self, side):
        """Lane to dodge to if my lane is blocked: the free lane nearest to home (lane 2,
        both free, home 2: away from the object, y ~ 0 -> right); None if none."""
        lane, home = self.lane, self.home
        free = [ln for ln in (1, 2, 3) if ln != lane and ln not in side]
        if not free:
            return None
        free.sort(key=lambda ln: abs(ln - home))
        if len(free) == 2 and abs(free[0] - home) == abs(free[1] - home):
            return 1 if self.front_oy < -self.center_tol else 3
        return free[0]

    def my_lane_blocked(self, side):
        """Object ahead close enough for the dodge it would cause (one / two lanes)."""
        if self.front_scans < self.obstacle_scans or self.front_near is None:
            return False
        return self.front_near < self.dodge_limit(side)

    def dodge_limit(self, side):
        """Front distance that starts the dodge: dodge2 for a two-lane change, but dodge
        when the middle lane is blocked (go on straight, maybe it frees -> one lane)."""
        tgt = self.dodge_target(side)
        if tgt is None or abs(tgt - self.lane) != 2:
            return self.dodge
        return self.dodge if 2 in side else self.dodge2

    def blocked_lanes(self):
        if self.lane is None:
            return set()
        side = self.side_blocked()
        return side | ({self.lane} if self.my_lane_blocked(side) else set())

    def decide(self):
        """(side, target) or None, from self.lane / home / blocked lanes (step 4)."""
        lane, home = self.lane, self.home
        if lane is None:
            return None
        side = self.side_blocked()
        if self.my_lane_blocked(side):
            target = self.dodge_target(side)
            if target is None:
                return None
        elif lane != home:
            step = 1 if home > lane else -1
            way = range(lane + step, home + step, step)
            if any(ln in side for ln in way):
                return None
            target = home
        else:
            return None
        return (LEFT if target < lane else RIGHT), target

    def loop(self):
        """Decision loop while ARMED (on every scan)."""
        now = self.now()
        if not self.aligned(now):
            return
        move = self.decide()
        bl = sorted(self.blocked_lanes())
        wait = ''
        if move is None and self.lane in (1, 3) and 2 in self.side_blocked() \
                and self.front_near is not None and self.front_near < self.dodge2:
            wait = f' (lane 2 blocked: straight to {self.dodge:.2f})'
        self.decision = (f'lane {self.lane} home {self.home} blocked {bl or "-"} -> '
                         + (f'{move[0]} to {move[1]}' if move else 'stay') + wait)
        if move is None:
            return
        self.request = move
        self.request_at = now
        self.set_state(MissionState.REQUEST, self.decision)

    def check_guard(self, x, y):
        lx = self.laser[0]
        y0, y1 = -self.guard_half, self.guard_half
        changing = self.state == MissionState.RUNNING and self.phase in (TURN, MID, COUNTER)
        if changing and self.request is not None and self.request[0] in (LEFT, RIGHT):
            if self.request[0] == LEFT:     # the front corner sweeps to the side it moves to
                y1 += self.guard_turn_extra
            else:
                y0 -= self.guard_turn_extra
        self.guard_y = (y0, y1)
        sel = (x > self.front_x - 0.02) & (y > y0) & (y < y1)
        self.guard_near = float(np.min(x[sel]) - lx) if sel.any() else None
        close = sel & (x - lx < self.guard_dist)
        self.guard_count = self.guard_count + 1 \
            if int(np.count_nonzero(close)) >= self.guard_min_points else 0
        if self.guard_count < self.guard_scans:
            return
        why = f'guard: object {self.guard_near:.2f} m from the lidar'
        now = self.now()
        if changing:
            self.guard_count = 0
            self.start_backoff(now, ALIGN, f'{why}, lane change aborted')
        elif self.state in (MissionState.ARMED, MissionState.IDLE) and self.in_section() \
                and self.cmd_v > 0.0:
            self.request = (GUARD, None)
            self.request_at = now
            self.set_state(MissionState.REQUEST, f'{why} -> back off')

    # ========================================================= lane change
    def on_start(self, now):
        self.judge = None
        if self.request is not None and self.request[0] == GUARD:
            self.start_backoff(now, PRE_STOP, 'guard backoff')
        else:
            self.start_pre_stop(now, 'change granted')

    def on_stop(self, now):
        self.request = None
        self.plan = []
        self.judge = None
        self.after_backoff = None
        self.guard_count = 0

    def start_pre_stop(self, now, why):
        self.phase_since = now
        self.centre_samples = []
        self.set_phase(PRE_STOP, f'{why}, hold {self.pre_stop:.1f} s')

    def start_backoff(self, now, after, why, need=None):
        self.after_backoff = after
        self.backoff_need = self.backoff_clear if need is None else need
        self.phase_since = now
        self.set_phase(BACKOFF, why)

    def try_start(self, now):
        """Stopped long enough: decide again, back off when too close for that change
        (one / two lanes), else change."""
        move = self.decide()
        bl = sorted(self.blocked_lanes())
        wait = ''
        if move is None and self.lane in (1, 3) and 2 in self.side_blocked() \
                and self.front_near is not None and self.front_near < self.dodge2:
            wait = f' (lane 2 blocked: straight to {self.dodge:.2f})'
        self.decision = (f'lane {self.lane} home {self.home} blocked {bl or "-"} -> '
                         + (f'{move[0]} to {move[1]}' if move else 'stay') + wait)
        if move is None:
            self.request = None
            self.finish(f'{self.decision}: lane follow')
            return
        side, target = move
        # two lanes with the middle blocked started at dodge (not dodge2): one-lane clearance
        two_far = abs(target - self.lane) == 2 and 2 not in self.side_blocked()
        need = self.backoff_clear2 if two_far else self.backoff_clear
        if self.guard_near is not None and self.guard_near < need:
            self.start_backoff(now, PRE_STOP, f'{side} to {target}: object '
                                              f'{self.guard_near:.2f} m < {need:.2f} m, '
                                              f'back off to {need:.2f} m', need)
            return
        self.request = move
        self.orig_lane = self.lane
        if abs(target - self.lane) == 2:
            t, m, c = self.times2[side]
        else:
            (t, c), m = self.times[side], None
        # sideways offset at the start: the lane centre y (+ = centre left of the car) is
        # extra travel to the left, less travel to the right -> longer / shorter turn
        bias_txt = ''
        if self.centre_samples:
            cy = float(np.mean(self.centre_samples))
            extra = cy if side == LEFT else -cy
            if m is not None:
                # two lanes: only the straight mid step (heading kept, sideways travel only)
                bias = max(-self.mid_bias_max, min(self.mid_bias_max, self.mid_bias_k * extra))
                m = max(0.0, m + bias)
                bias_txt = f' (mid {bias:+.2f} s, centre {cy * 100:+.0f} cm)'
            else:
                # one lane: turn and counter by the same time, the car still ends parallel
                bias = max(-self.turn_bias_max,
                           min(self.turn_bias_max, self.turn_bias_k * extra))
                t, c = max(0.1, t + bias), max(0.1, c + bias)
                bias_txt = f' (turn/counter {bias:+.2f} s, centre {cy * 100:+.0f} cm)'
        self.plan = [(TURN, t)] + ([(MID, m)] if m is not None else []) + [(COUNTER, c)]
        steps = ' / '.join(f'{sec:.2f}' for _, sec in self.plan)
        self.get_logger().info(f'{side} to lane {target}: {steps} s{bias_txt}')
        self.next_step(now, f'{self.decision}, {steps} s{bias_txt}')

    def next_step(self, now, why=''):
        phase, sec = self.plan.pop(0)
        self.phase_until = now + sec
        self.set_phase(phase, why or f'{sec:.1f} s')

    def start_judge(self, now, origin, why):
        """origin: 'change' (after a lane change) or 'guard' (after an aborted one)."""
        self.judge = dict(since=now, cand=None, count=0, single=0, origin=origin)
        self.set_phase(JUDGE, why)

    def judge_frame(self, now, al, lane, mine):
        j = self.judge
        if now - j['since'] < self.judge_discard or not al or mine < self.drivable_min:
            return   # frames from before the stop / not parallel: no answer
        if lane is None:
            j['single'] += 1
            j['cand'], j['count'] = None, 0
            return
        j['single'] = 0
        j['count'] = j['count'] + 1 if lane == j['cand'] else 1
        j['cand'] = lane

    def step_judge(self, now):
        j = self.judge
        if j['count'] >= self.lane_frames:
            self.judge = None
            old, self.lane = self.lane, j['cand']
            self.lane_unknown = False
            self.zone_grace_from = now
            self.get_logger().info(f'judged lane {self.lane} (was {old}, home {self.home})')
            if j['origin'] == 'guard':      # moved while aligning: the full stop first
                self.start_pre_stop(now, f'lane {self.lane} judged')
            else:                           # post_stop already held the car
                self.try_start(now)
            return
        lost = j['single'] >= self.lane_frames
        if not lost and now - j['since'] < self.judge_max:
            return
        self.judge = None
        if j['origin'] == 'guard' and not lost:
            self.lane = self.orig_lane      # not judged: the lane before the aborted change
            self.get_logger().info(f'not judged after the guard: lane {self.lane} kept')
            self.start_pre_stop(now, f'lane {self.lane} kept')
            return
        why = 'my lane only' if lost else 'no lane judged (not parallel?)'
        self.lane, self.lane_unknown = None, True
        self.zone_grace_from = now
        self.request = None
        self.get_logger().info(f'lane unknown: {why}')
        self.finish(f'{why}: lane unknown -> lane follow')

    def step_running(self, now):
        ph = self.phase
        age = now - (self.phase_since or now)
        if ph == PRE_STOP:
            self.publish(0.0, 0.0)
            c = self.centre_y
            if self.aligned(now) and math.isfinite(c) and abs(c) <= self.box_shift_max:
                self.centre_samples.append(c)
            if age >= self.pre_stop:
                self.try_start(now)
        elif ph == BACKOFF:
            clear = self.guard_near is None or self.guard_near >= self.backoff_need
            if (age >= self.backoff_min and clear) or age >= self.backoff_max:
                self.publish(0.0, 0.0)
                self.guard_count = 0
                if self.after_backoff == ALIGN:
                    self.start_align(now, f'backoff {age:.1f} s')
                else:
                    self.start_pre_stop(now, f'backoff {age:.1f} s')
            else:
                self.publish(self.backoff_speed, 0.0)
        elif ph in (TURN, MID, COUNTER):
            if now >= self.phase_until:
                if self.plan:
                    self.next_step(now)
                else:
                    self.phase_since = now
                    self.publish(0.0, 0.0)
                    self.set_phase(POST_STOP, f'lane {self.orig_lane} -> {self.request[1]} '
                                              f'done, hold {self.post_stop:.1f} s')
                    return
            side = self.request[0]
            sign = 1.0 if side == LEFT else -1.0
            w = {TURN: sign * self.w, MID: 0.0}.get(self.phase, -sign * self.w)
            self.publish(self.speed, w)
        elif ph == POST_STOP:
            self.publish(0.0, 0.0)
            if age >= self.post_stop:
                self.start_judge(now, 'change', 'judge the lane standing')
        elif ph == ALIGN:
            self.step_align(now, age)
        elif ph == JUDGE:
            self.publish(0.0, 0.0)
            self.step_judge(now)

    def start_align(self, now, why):
        self.phase_since = now
        if self.aligned(now, self.align_done):
            self.start_judge(now, 'guard', f'{why}, parallel')
        elif self.aligned(now, self.align_trust):
            self.set_phase(ALIGN, f'{why}, yaw {self.yaw_text()} -> {self.align_done:.0f} deg')
        else:
            self.start_judge(now, 'guard', f'{why}, yaw {self.yaw_text()} not trusted, '
                                           f'no align')

    def step_align(self, now, age):
        if self.aligned(now, self.align_done) or not self.aligned(now, self.align_trust) \
                or age >= self.align_max_sec:
            self.publish(0.0, 0.0)
            self.start_judge(now, 'guard', f'align {age:.1f} s, yaw {self.yaw_text()}')
            return
        w = self.align_k * math.radians(self.yaw[0])   # backing off turns the car parallel
        self.publish(self.align_speed, max(-self.align_w_max, min(self.align_w_max, w)))

    # ================================================================ tick
    def tick(self, now):
        if now - self.last_debug >= self.debug_period and self.pub_debug.get_subscription_count():
            self.last_debug = now
            self.publish_debug(now)
        st = self.state
        if st == MissionState.RUNNING and self.active and self.phase:
            self.step_running(now)
        elif st == MissionState.DONE and self.active:
            self.publish(0.0, 0.0)          # until the master switches back to normal
        elif st == MissionState.REQUEST and now - self.request_at > self.request_timeout:
            self.request = None
            self.set_state(MissionState.ARMED if self.lane is not None else MissionState.IDLE,
                           f'not granted within {self.request_timeout:.0f} s')
        elif st == MissionState.ARMED and self.section_lost(now):
            self.leave_section(f'no side lane for {self.zone_exit:.0f} s')
        elif st == MissionState.IDLE and (self.lane is not None or self.lane_unknown):
            if self.section_lost(now):
                self.leave_section(f'no side lane for {self.zone_exit:.0f} s')
            elif self.lane is not None and not self.in_cooldown(now):
                self.set_state(MissionState.ARMED, f'3-lane section, lane {self.lane}')
        self.speed_limit = self.section_speed if self.in_section() else 0.0

    # =============================================================== debug
    def publish_debug(self, now):
        img = np.zeros((300, 640, 3), np.uint8)
        self.draw_classes(img[:240, :320], now)
        self.draw_lidar(img[:240, 320:])
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

    def draw_classes(self, view, now):
        if self.dbg_cls is None:
            put(view, 'no /yolopx/drivable_class', (10, 120), (0, 0, 255), 0.5)
            return
        lut = np.zeros((256, 3), np.uint8)
        lut[1], lut[2], lut[3] = (0, 160, 0), (235, 206, 135), (200, 50, 160)
        view[:] = cv2.resize(lut[self.dbg_cls], (320, 240), interpolation=cv2.INTER_NEAREST)
        y_roi = int(240 * self.roi_top)
        cv2.line(view, (0, y_roi), (319, y_roi), (255, 255, 255), 1)
        for i, (name, v, col) in enumerate((('L', self.dbg_px[0], (235, 206, 135)),
                                            ('M', self.dbg_px[1], (0, 255, 0)),
                                            ('R', self.dbg_px[2], (200, 50, 160)))):
            mark = '' if name == 'M' else (' ok' if v >= self.side_min else ' -')
            put(view, f'{name} {v}{mark}', (8 + 104 * i, 20), col, 0.5)
        lane = f'lane {self.lane}' if self.lane else ('lane ?' if self.lane_unknown else 'lane -')
        put(view, f'{lane}  home {self.home}  cand {self.cand or "-"} x{self.cand_count}',
            (8, 44), (255, 255, 255), 0.45)
        put(view, self.decision[:48], (8, 66), (0, 255, 255), 0.42)
        if self.judge is not None:
            j = self.judge
            put(view, f'judge: cand {j["cand"] or "-"} x{j["count"]}/{self.lane_frames} '
                      f'single {j["single"]}', (8, 88), (0, 255, 255), 0.45)
        al = self.aligned(now)
        put(view, f'{"aligned" if al else "NOT aligned"}  yaw {self.yaw_text()}', (8, 132),
            (0, 255, 0) if al else (0, 0, 255), 0.45)
        sc = self.sign_counts
        put(view, f'A {sc[1]} / B {sc[3]} (need {self.sign_decide})', (8, 154),
            (255, 200, 0), 0.45)

    def draw_lidar(self, view):
        scale, ox, oy = 105.0, 160, 228               # px / m (2.1 m ahead), car origin

        def px(x, y):
            return int(ox - y * scale), int(oy - x * scale)

        for g in np.arange(0.2, 2.01, 0.2):
            cv2.line(view, px(g, -1.4), px(g, 1.4), (50, 50, 50), 1)
        for g in np.arange(-1.4, 1.41, 0.2):
            cv2.line(view, px(0.0, g), px(2.0, g), (50, 50, 50), 1)
        fhit = self.front_scans > 0
        boxes = [('front', self.front_x, self.laser[0] + max(self.dodge, self.dodge2),
                  -self.front_half,
                  self.front_half, fhit)]
        th = self.dbg_angle
        for side, sign in ((LEFT, 1.0), (RIGHT, -1.0)):   # lane boxes, laid along the lane
            c = self.dbg_shift + sign * self.side_offset
            col = (0, 0, 255) if self.box[side] else (0, 200, 0)
            corners = []
            for uu, vv in ((self.side_x[0], c - self.side_half), (self.side_x[1], c - self.side_half),
                           (self.side_x[1], c + self.side_half), (self.side_x[0], c + self.side_half)):
                corners.append(px(uu * math.cos(th) - vv * math.sin(th),
                                  uu * math.sin(th) + vv * math.cos(th)))
            cv2.polylines(view, [np.array(corners, np.int32)], True, col,
                          2 if self.box[side] else 1)
            put(view, side, (corners[2][0] + 3, corners[2][1] + 14), col, 0.4)
        for name, x0, x1, y0, y1, hit in boxes:
            col = (0, 0, 255) if hit else (0, 200, 0)
            cv2.rectangle(view, px(x1, y1), px(x0, y0), col, 2 if hit else 1)
            put(view, name, (px(x1, y1)[0] + 3, px(x1, y1)[1] + 14), col, 0.4)
        # ahead clearance before a change: one lane / two lanes (from the lidar), the one a
        # running backoff aims for drawn thick
        for dist, label, col in ((self.dodge, 'dodge1', (0, 165, 255)),
                                 (self.dodge2, 'dodge2', (0, 100, 255)),
                                 (self.backoff_clear, 'chk1', (255, 255, 0)),
                                 (self.backoff_clear2, 'chk2', (255, 0, 255))):
            cx = self.laser[0] + dist
            hw = self.front_half + self.guard_turn_extra
            near_hit = self.guard_near is not None and self.guard_near < dist
            cv2.line(view, px(cx, hw), px(cx, -hw), (0, 0, 255) if near_hit else col, 1)
            if label.startswith('dodge'):   # dodge labels right of the box, checks left
                put(view, f'{label} {dist:.2f}', (px(cx, -hw)[0] + 4, px(cx, -hw)[1] + 4),
                    col, 0.33)
            else:
                put(view, f'{label} {dist:.2f}', (px(cx, hw)[0] - 70, px(cx, hw)[1] + 4),
                    col, 0.33)
        if self.phase == BACKOFF:   # where the running backoff stops
            cx = self.laser[0] + self.backoff_need
            hw = self.front_half
            cv2.line(view, px(cx, hw), px(cx, -hw), (0, 255, 255), 2)
            put(view, f'back to {self.backoff_need:.2f}', (px(cx, hw)[0] - 90, px(cx, hw)[1] - 4),
                (0, 255, 255), 0.35)
        gx = self.laser[0] + self.guard_dist
        gcol = (0, 0, 255) if self.guard_count else (0, 140, 255)
        cv2.rectangle(view, px(gx, self.guard_y[1]), px(self.front_x, self.guard_y[0]), gcol,
                      2 if self.guard_count else 1)
        near = f'{self.guard_near:.2f} m' if self.guard_near is not None else '-'
        put(view, f'ahead {near}', (200, 236), gcol, 0.45)
        put(view, f'boxes {self.dbg_shift * 100:+.0f}cm {math.degrees(self.dbg_angle):+.0f}deg',
            (180, 14), (180, 180, 180), 0.4)
        x, y = self.dbg_pts
        sel = (x > -0.1) & (x < 2.15) & (np.abs(y) < 1.5)
        for xi, yi in zip(x[sel], y[sel]):
            cv2.circle(view, px(xi, yi), 2, (255, 255, 255), -1)
        cv2.rectangle(view, px(self.front_x, 0.11), px(-0.16, -0.11), (0, 255, 255), 1)
        put(view, f'front x{self.front_scans}/{self.obstacle_scans}', (6, 236),
            (0, 0, 255) if fhit else (180, 180, 180), 0.45)

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
    except (KeyboardInterrupt, ExternalShutdownException):
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
