"""
3road mission node, v3 (2026-11-18): map + odometry, avoid in real time, lanes do not matter.

Trigger
  IDLE -> REQUEST   3-lane section seen in the drivable classes (left and right side lane
                    >= side_min_pixels, my lane >= drivable_min_pixels, lane_frames frames,
                    parallel to the lane) -- not during the cooldown
  REQUEST -> IDLE   not granted within request_timeout
Start (RUNNING)
  The odom pose is pinned to the track drawing (road3_plan.py, x along the road, the car
  drives -x): x = trigger_x, y = lane 2 centre + lane_follow's centre offset, heading = road
  direction - lane_follow's near yaw. Entry line correction: the white line at the entry
  (stop line mask bottom at stop_line_row_ratio = line_dist m ahead of base_link) fixes x,
  whether it was seen up to line_before m before the trigger or line_after m after it.
Drive
  Every plan_period: lidar hits + seen cells into the map, Frenet DP to the exit
  (road3_plan.FrenetDP). Pure pursuit on odom at control_rate_hz (w = w_gain v k).
  Speed: drive_speed, slow_speed when the path ahead is tight or curved.
  No path: squeeze (smaller footprint, creep_speed); still none: back along the driven
  track (<= reverse_max m, checked on the map), then plan again.
  Signs A / B (sign_decide_frames, sign_ratio, also seen sign_pre_sec before the start)
  wall off the other side of the island and pull the path into that road early.
  Guard: lidar points right ahead of the bumper while going forward -> stop, plan again.
Finish
  map x < exit_x (single exit road) -> lane follow; also after max_sec.
"""
from array import array
import math
import signal

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu, LaserScan
from std_msgs.msg import Float32MultiArray, String
from trinity_msgs.msg import MissionState
from vision_msgs.msg import Detection2DArray

from trinity_missions import road3_plan as R3
from trinity_missions.mission_base import MissionNode

try:
    import cv2
except ImportError:      # debug image only
    cv2 = None

DRIVE, SQUEEZE, REVERSE, WAIT, ALIGN = 'drive', 'squeeze', 'reverse', 'wait', 'align'


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class ThreeRoadNode(MissionNode):

    def __init__(self):
        super().__init__('three_road_node', '3road', topic_name='three_road')
        d = self.declare_parameter
        # topics
        d('drivable_class_topic', '/yolopx/drivable_class')
        d('stop_line_mask_topic', '/yolopx/stop_line_mask')
        d('detections_topic', '/yolopx/detections')
        d('scan_topic', '/scan')
        d('odom_topic', '/odometry/filtered')
        d('yaw_topic', '/lane_follow/yaw')
        d('lane_state_topic', '/lane_follow/state')
        d('cmd_topic', '/three_road_cmd_vel')
        d('debug_topic', '/three_road/debug')
        d('debug_rate_hz', 5.0)
        # trigger: 3 lanes
        d('roi_top_ratio', 0.0)
        d('side_min_pixels', 1500)
        d('drivable_min_pixels', 9000)
        d('lane_frames', 3)
        d('align_max_deg', 10.0)
        d('yaw_timeout', 0.5)
        d('request_timeout', 2.0)
        # start pose / entry line
        d('trigger_x', R3.ENTRY_X)          # map x of base_link when the 3 lanes are judged
        d('use_lane_offset', True)           # y from lane_follow's lane centre offset
        d('use_lane_yaw', True)
        d('start_yaw_sec', 0.6)              # s of lane_follow yaw frames averaged for the start
        # heading correction while running: lane_follow's yaw where every line is parallel to x
        d('hc_on', True)
        d('hc_x_min', 4.9)                   # m (map x): no island / taper lines in view
        d('hc_x_max', 8.0)
        d('hc_agree_deg', 3.0)               # near and ahead yaw within this
        d('hc_max_deg', 8.0)                 # larger differences are ignored (a wrong line)
        d('hc_gain', 0.05)                   # share of the difference taken per frame (~20 Hz)
        d('hc_step_deg', 0.3)                # applied (frame + obstacle grid) in steps of this
        d('hc_yaw_max_deg', 25.0)            # lane_follow yaw above this: not trusted
        d('hc_total_max_deg', 6.0)           # most heading correction per run
        d('start_yaw_max_deg', 6.0)          # start: frames with |yaw| above this not used
        # side correction: lane_follow's lane centre offset vs the lane centres on the drawing
        d('lc_on', True)
        d('lc_x_ranges', [4.4, 8.2, 2.85, 3.95])   # m (map x) pairs: straight lines only
        d('lc_states', ['both'])             # lane_follow states trusted (both lines seen)
        d('lc_yaw_max_deg', 6.0)             # only nearly parallel to the lane
        d('lc_max', 0.15)                    # m, bigger differences are ignored (other lane?)
        d('lc_gain', 0.1)                    # share of the difference per frame
        # beside the island one lane per side (no other lane to mix up): looser
        d('lc_island_x', [2.85, 3.95])       # m (map x)
        d('lc_island_states', ['both', 'single_left', 'single_right'])
        d('lc_island_max', 0.30)
        d('lc_island_gain', 0.2)
        d('stop_line_row_ratio', 0.80)
        d('stop_line_min_pixels', 200)
        d('stop_line_frames', 2)
        d('line_dist', 0.35)                 # m, base_link to the line at that row
        d('line_before', 1.5)                # m of odom before the trigger a line still counts
        d('line_after', 1.2)                 # m after the trigger a line is still taken
        d('line_max_fix', 0.6)               # m, larger corrections are ignored
        # start alignment: stop, put the entry line at the stop_line_row_ratio row (forward or
        # back), measure lane_follow standing, then pin the map (driving pins were off)
        d('align_on', True)
        # crosswalk heading reference: the car always stands at the crosswalk before the section
        # (right road, driving map -y). lane_follow yaw standing there gives the car's map heading;
        # map heading - odom yaw is kept and the 3road start heading = odom yaw + that (the
        # position there is not used: its along-road error would be the 3road y error after the
        # 90 deg curve). Used instead of the start alignment when fresh.
        d('cw_on', True)
        d('cw_mode', 'crosswalk')            # master mode of the crosswalk stop
        d('cw_road_heading_deg', -90.0)      # map heading of the crosswalk road
        d('cw_still_sec', 0.4)               # s without odom motion = standing
        d('cw_min_frames', 6)                # standing lane_follow frames needed
        d('cw_yaw_max_deg', 8.0)             # |lane_follow yaw| above this is not used
        d('cw_max_age', 40.0)                # s from the crosswalk stop to the 3road start
        # m/s normal-mode cap from the crosswalk to the 3road start (less slip, steadier start)
        d('approach_speed', 0.25)
        d('cw_max_dist', 6.0)                # m of odom from there (map: ~2.3 m)
        d('align_fwd_speed', 0.10)           # m/s, line ahead of the row: creep to it
        d('align_back_speed', 0.12)          # m/s, line passed / closer: back until it is
        d('align_max_back', 1.0)             # m
        d('align_settle_sec', 0.6)           # s standing, lane_follow yaw / centre medians
        d('align_timeout', 6.0)              # s, then start as without alignment
        d('align_line_fresh', 0.3)           # s, older stop line masks do not count
        # lidar / map
        d('laser_x', 0.103)
        d('laser_y', 0.0)
        d('laser_yaw', 0.0)
        d('scan_range', 4.0)                 # m, rays without a hit end here (seen cells)
        d('self_radius', 0.20)               # m, points this close to base_link are the car
        d('map_res', 0.025)
        d('hit_min', 2)
        d('hit_cap', 6)                      # hits saturate (a wrong cell clears in a few scans)
        d('clear_margin', 0.06)              # m before a ray's end: not cleared by it
        d('static_extra', 0.015)             # m more margin to the yellow lines than to objects
        d('guard_hold', 1.0)                 # s of guard while a path says go -> back off
        d('car_r', 0.13)
        d('circ_off', 0.09)
        d('squeeze_r', 0.115)
        # planner
        d('plan_period', 0.2)
        d('k_hard', 3.0)
        d('pad', 0.04)
        d('w_unseen', 3.0)
        d('w_clear', 4.0)                    # cost weight for clearance below car_r + pad
        d('side_ahead', 1.8)
        d('w_side', 2.0)                     # cost per m on the wrong side of the island line
        d('w_centre', 0.3)                   # cost per m x m off the sign's lane centre
        d('w_hedge', 0.8)                    # sign not decided: cost per m off lane 2 (middle)
        d('nose_len', 0.40)                  # m, no-go wedge in front of the island tip (0 = off)
        d('nose_half', 0.05)                 # m, its half width at the tip
        d('island_bias', 0.06)               # m, beside the island: wanted line this much closer
        d('w_centre_island', 2.0)            # to the island, held with this weight
        d('pad_island', 0.02)                # m, pad there (0.06 held it in the road middle)
        d('hedge_ahead', 2.5)                # m before the island tip where that applies
        d('island_zone_x', 5.0)              # m (map x): from here to the exit at most
        d('island_speed', 0.0)              # this speed (into the A / B road straight)
        # control
        d('control_rate_hz', 20.0)
        d('drive_speed', 0.40)
        d('slow_speed', 0.22)
        d('curve_speed', 0.25)
        d('creep_speed', 0.12)
        d('reverse_speed', 0.15)
        d('reverse_max', 1.0)
        d('reverse_min', 0.3)                # m backed off before planning again
        d('min_partial', 1.2)                # m: a path short of the exit must still go this far
        d('partial_speed', 0.25)             # m/s on a partial path (exit not reachable yet)
        d('tight_clear', 0.20)               # m, path ahead closer than this -> slow_speed
        d('curve_k', 1.5)                    # 1/m, pure pursuit curvature above -> curve_speed
        d('lookahead', 0.30)
        d('lookahead_slow', 0.22)
        d('w_gain', 2.0)                     # cmd w = w_gain v k (the car turns ~0.57 x)
        d('w_max', 1.5)
        d('odom_timeout', 0.3)
        # own dead reckoning: distance from the odom topic, heading changes from heading_source
        # ('imu': /imu orientation, 'odom': the odom topic's yaw) x turn_scale. 2026-11-19: the EKF
        # yaw turned ~98 deg in the 90 deg curve after the crosswalk (start 171-172 deg)
        d('imu_topic', '/imu')
        d('heading_source', 'imu')
        d('imu_timeout', 0.2)                # s, older -> the odom yaw for that step
        d('turn_scale', 1.0)                 # heading change x this
        d('guard_dist', 0.08)                # m from the front bumper
        d('guard_half', 0.11)
        d('front_x', 0.16)                   # m, base_link to the front bumper
        d('guard_min_points', 2)
        # signs
        d('sign_left_class', 'sign_A')       # A road = lane 1 side
        d('sign_right_class', 'sign_B')      # B road = lane 3 side
        d('sign_conf', 0.5)
        d('sign_ratio', 0.8)
        d('sign_decide_frames', 6)
        d('sign_pre_sec', 15.0)
        # finish
        d('exit_x', R3.EXIT_X)
        d('max_sec', 55.0)
        d('side_lost_sec', 0.0)              # s without any side lane -> finish (0 = off)
        d('track_log_period', 0.5)           # s, 'track:' log line while running (localisation)

        def g(n):
            return self.get_parameter(n).value

        self.roi_top = float(g('roi_top_ratio'))
        self.side_min = int(g('side_min_pixels'))
        self.drivable_min = int(g('drivable_min_pixels'))
        self.lane_frames = int(g('lane_frames'))
        self.align_max = float(g('align_max_deg'))
        self.yaw_timeout = float(g('yaw_timeout'))
        self.request_timeout = float(g('request_timeout'))
        self.trigger_x = float(g('trigger_x'))
        self.use_lane_offset = bool(g('use_lane_offset'))
        self.use_lane_yaw = bool(g('use_lane_yaw'))
        self.start_yaw_sec = float(g('start_yaw_sec'))
        self.hc_on = bool(g('hc_on'))
        self.hc_x = (float(g('hc_x_min')), float(g('hc_x_max')))
        self.hc_agree = float(g('hc_agree_deg'))
        self.hc_max = float(g('hc_max_deg'))
        self.hc_gain = float(g('hc_gain'))
        self.hc_step = float(g('hc_step_deg'))
        self.hc_yaw_max = float(g('hc_yaw_max_deg'))
        self.hc_total_max = float(g('hc_total_max_deg'))
        self.start_yaw_max = float(g('start_yaw_max_deg'))
        self.lc_on = bool(g('lc_on'))
        r = [float(v) for v in g('lc_x_ranges')]
        self.lc_ranges = list(zip(r[0::2], r[1::2]))
        self.lc_states = set(g('lc_states'))
        self.lc_yaw_max = float(g('lc_yaw_max_deg'))
        self.lc_max = float(g('lc_max'))
        self.lc_gain = float(g('lc_gain'))
        self.lci_x = tuple(float(v) for v in g('lc_island_x'))
        self.lci_states = set(g('lc_island_states'))
        self.lci_max = float(g('lc_island_max'))
        self.lci_gain = float(g('lc_island_gain'))
        self.stop_row = float(g('stop_line_row_ratio'))
        self.stop_min_px = int(g('stop_line_min_pixels'))
        self.stop_frames = int(g('stop_line_frames'))
        self.line_dist = float(g('line_dist'))
        self.line_before = float(g('line_before'))
        self.line_after = float(g('line_after'))
        self.line_max_fix = float(g('line_max_fix'))
        self.align_on = bool(g('align_on'))
        self.align_fwd = float(g('align_fwd_speed'))
        self.align_back = float(g('align_back_speed'))
        self.align_max_back = float(g('align_max_back'))
        self.align_settle = float(g('align_settle_sec'))
        self.align_timeout = float(g('align_timeout'))
        self.align_line_fresh = float(g('align_line_fresh'))
        self.cw_on = bool(g('cw_on'))
        self.cw_mode = str(g('cw_mode'))
        self.cw_road = math.radians(float(g('cw_road_heading_deg')))
        self.cw_still = float(g('cw_still_sec'))
        self.cw_min_frames = int(g('cw_min_frames'))
        self.cw_yaw_max = float(g('cw_yaw_max_deg'))
        self.cw_max_age = float(g('cw_max_age'))
        self.cw_max_dist = float(g('cw_max_dist'))
        self.approach_speed = float(g('approach_speed'))
        self.slow_from = None         # (t, odom_dist) when the crosswalk mode ended
        self.master_mode = ''
        self.moved_at = None          # s, last odom motion
        self.cw_samples = []          # (map heading - odom yaw) while standing at the crosswalk
        self.cw_ref = None            # (offset rad, t, odom_dist, n, spread deg)
        self.laser = (float(g('laser_x')), float(g('laser_y')), float(g('laser_yaw')))
        self.scan_range = float(g('scan_range'))
        self.self_r = float(g('self_radius'))
        self.squeeze_r = float(g('squeeze_r'))
        self.plan_period = float(g('plan_period'))
        self.drive_speed = float(g('drive_speed'))
        self.slow_speed = float(g('slow_speed'))
        self.curve_speed = float(g('curve_speed'))
        self.creep_speed = float(g('creep_speed'))
        self.rev_speed = float(g('reverse_speed'))
        self.rev_max = float(g('reverse_max'))
        self.rev_min = float(g('reverse_min'))
        self.min_partial = float(g('min_partial'))
        self.partial_speed = float(g('partial_speed'))
        self.tight_clear = float(g('tight_clear'))
        self.curve_k = float(g('curve_k'))
        self.ld = float(g('lookahead'))
        self.ld_slow = float(g('lookahead_slow'))
        self.w_gain = float(g('w_gain'))
        self.w_max = float(g('w_max'))
        self.odom_timeout = float(g('odom_timeout'))
        self.heading_src = str(g('heading_source'))
        self.imu_timeout = float(g('imu_timeout'))
        self.turn_scale = float(g('turn_scale'))
        self.imu_yaw, self.imu_at = None, None
        self.raw = None               # last odom topic pose
        self.imu_prev = None          # IMU yaw at the last odom step
        self.turns = [0.0, 0.0]       # summed heading change: odom topic, IMU (unscaled)
        self.guard_dist = float(g('guard_dist'))
        self.guard_half = float(g('guard_half'))
        self.front_x = float(g('front_x'))
        self.guard_min = int(g('guard_min_points'))
        self.guard_hold = float(g('guard_hold'))
        self.sign_classes = {g('sign_left_class'): 1, g('sign_right_class'): 3}
        self.sign_conf = float(g('sign_conf'))
        self.sign_ratio = float(g('sign_ratio'))
        self.sign_decide = int(g('sign_decide_frames'))
        self.sign_pre = float(g('sign_pre_sec'))
        self.exit_x = float(g('exit_x'))
        self.max_sec = float(g('max_sec'))
        self.side_lost_sec = float(g('side_lost_sec'))
        self.track_log_period = float(g('track_log_period'))
        self.island_zone_x = float(g('island_zone_x'))
        self.island_speed = float(g('island_speed'))
        self.debug_period = 1.0 / float(g('debug_rate_hz'))

        self.nose_len = float(g('nose_len'))
        self.map = R3.Road3Map(res=float(g('map_res')), hit_min=int(g('hit_min')),
                               nose_len=self.nose_len, nose_half=float(g('nose_half')),
                               car_r=float(g('car_r')), circ_off=float(g('circ_off')),
                               hit_cap=int(g('hit_cap')), clear_margin=float(g('clear_margin')),
                               static_extra=float(g('static_extra')))
        self.dp = R3.FrenetDP(k_hard=float(g('k_hard')), pad=float(g('pad')),
                              w_unk=float(g('w_unseen')), w_clear=float(g('w_clear')),
                              side_ahead=float(g('side_ahead')), w_side=float(g('w_side')),
                              w_centre=float(g('w_centre')), x_end=self.exit_x - 0.25,
                              w_hedge=float(g('w_hedge')), hedge_ahead=float(g('hedge_ahead')),
                              island_bias=float(g('island_bias')),
                              w_centre_island=float(g('w_centre_island')),
                              pad_island=float(g('pad_island')))
        if self.nose_len > 0:
            self.dp.hedge_end = R3.ISL_TIP + self.nose_len   # the middle pull ends at the nose

        # inputs
        self.odom = None              # (x, y, yaw) odom frame
        self.odom_at = None
        self.odom_dist = 0.0          # m driven (odom), for the line memory
        self.yaw = (float('nan'), float('nan'))
        self.centre_y = float('nan')
        self.yaw_at = None
        self.yaw_hist = []            # (t, near deg, ahead deg) for the start heading
        self.hc_total = 0.0           # deg, heading corrected so far this run
        self.hc_pending = 0.0         # rad, collected, applied in hc_step_deg steps
        self.lc_pending = 0.0         # m, side correction collected (applied per map cell)
        self.lc_total = 0.0
        self.lc_count = 0
        self.lane_state = ''
        self.start_xy = (R3.ENTRY_X, R3.MID_Y)
        self.hc_count = 0
        self.lane_count = 0
        self.side_seen_at = 0.0       # s, last frame with a left or right side lane
        self.dbg_px = (0, 0, 0)
        self.line_count = 0
        self.line_event = None        # (odom pose, odom_dist) when the entry line reached the row
        self.line_seen = None         # (t, bottom row / (H - 1), pixels) of the last mask
        self.align = None             # dict while the start alignment runs
        self.sign_hist = []           # (t, 1 | 3)
        self.request_at = None
        # run
        self.anchor = None            # (odom pose, map pose) pinned at the start
        self.anchor_dist = 0.0
        self.line_fixed = None        # m correction applied, None = not yet
        self.side_counts = (0, 0)
        self.path = None
        self.mode = WAIT
        self.last_plan = -1e9
        self.plan_ms = 0.0
        self.hist = []                # driven map poses (for the reverse)
        self.rev_dist = 0.0
        self.blocked = 0              # plans in a row without any path
        self.rev_can = True           # the running backoff can go on
        self.reach_x = None           # partial path: its end x (None = to the exit)
        self.last_block_log = 0.0
        self.guard_since = None       # s, guard holding the car while a path says go
        self.guard_scans = 0
        self.last_scan_pts = None
        self.dbg_scan = None          # this scan's hits in the map frame
        self.v_cmd, self.k_cmd = 0.0, 0.0
        self.last_debug = 0.0
        self.last_log = 0.0

        p = g
        self.create_subscription(Image, p('drivable_class_topic'), self.drivable_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(Image, p('stop_line_mask_topic'), self.stop_line_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(Detection2DArray, p('detections_topic'), self.signs_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(LaserScan, p('scan_topic'), self.scan_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, p('odom_topic'), self.odom_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(Float32MultiArray, p('yaw_topic'), self.yaw_callback, 10)
        self.create_subscription(Imu, p('imu_topic'), self.imu_callback, qos_profile_sensor_data)
        self.create_subscription(String, p('lane_state_topic'),
                                 lambda m: setattr(self, 'lane_state', m.data), 10)
        self.pub_cmd = self.create_publisher(Twist, p('cmd_topic'), 10)
        self.pub_debug = self.create_publisher(Image, p('debug_topic'), 1)
        self.create_timer(1.0 / float(p('control_rate_hz')), self.control)
        self.get_logger().info(
            f'ready (v3 map): 3 lanes x{self.lane_frames} -> start at x {self.trigger_x:.2f} '
            f'(entry line {R3.ENTRY_X:.2f}, fixed by the line), Frenet DP every '
            f'{self.plan_period:.1f} s, {self.drive_speed:.2f} m/s, exit at x {self.exit_x:.2f}')

    # ============================================================== inputs
    @staticmethod
    def quat_yaw(q):
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def imu_callback(self, msg):
        self.imu_yaw = self.quat_yaw(msg.orientation)
        self.imu_at = self.now()

    def dead_reckon(self, raw):
        """own pose: step length along the odom topic's heading, heading change from
        heading_source x turn_scale."""
        now = self.now()
        imu_ok = self.imu_at is not None and now - self.imu_at <= self.imu_timeout
        if self.raw is None:
            self.raw = raw
            self.imu_prev = self.imu_yaw if imu_ok else None
            return raw
        dx, dy = raw[0] - self.raw[0], raw[1] - self.raw[1]
        ds = dx * math.cos(self.raw[2]) + dy * math.sin(self.raw[2])   # signed
        if math.hypot(dx, dy) > 0.2:
            ds = 0.0                          # odom reset / jump
        d_odom = wrap(raw[2] - self.raw[2])
        d_imu = wrap(self.imu_yaw - self.imu_prev) if imu_ok and self.imu_prev is not None \
            else None
        self.turns[0] += d_odom
        self.turns[1] += d_odom if d_imu is None else d_imu
        dth = d_imu if (self.heading_src == 'imu' and d_imu is not None) else d_odom
        dth *= self.turn_scale
        x, y, th = self.odom
        mid = th + 0.5 * dth
        self.raw = raw
        self.imu_prev = self.imu_yaw if imu_ok else None
        return (x + ds * math.cos(mid), y + ds * math.sin(mid), wrap(th + dth))

    def odom_callback(self, msg):
        raw = (msg.pose.pose.position.x, msg.pose.pose.position.y,
               self.quat_yaw(msg.pose.pose.orientation))
        pose = self.dead_reckon(raw)
        if self.odom is None:
            self.moved_at = self.now()
        else:
            step = math.hypot(pose[0] - self.odom[0], pose[1] - self.odom[1])
            if step < 0.2:
                self.odom_dist += step
            if step > 0.0015 or abs(wrap(pose[2] - self.odom[2])) > 0.002:
                self.moved_at = self.now()
        self.odom = pose
        self.odom_at = self.now()

    def mode_callback(self, msg):
        prev, self.master_mode = self.master_mode, msg.mode
        if self.cw_on and msg.mode == self.cw_mode and prev != self.cw_mode:
            self.cw_samples = []
        elif self.cw_on and prev == self.cw_mode and msg.mode != self.cw_mode:
            self.cw_close()
            if self.approach_speed > 0.0:
                self.slow_from = (self.now(), self.odom_dist)
                self.speed_limit = self.approach_speed
                self.get_logger().info(f'approach: {self.approach_speed:.2f} m/s until the start')
        super().mode_callback(msg)

    def cw_sample(self, now, near, ahead, state):
        """standing in the crosswalk mode: one heading reference sample."""
        if self.odom is None or self.moved_at is None or now - self.moved_at < self.cw_still:
            return
        yaw = 0.5 * (near + ahead)
        if state not in self.lc_states or abs(near - ahead) > self.hc_agree \
                or abs(yaw) > self.cw_yaw_max:
            return
        car = self.cw_road - math.radians(yaw)    # car heading = road direction - lane yaw
        self.cw_samples.append(wrap(car - self.odom[2]))

    def cw_close(self):
        n = len(self.cw_samples)
        if n < self.cw_min_frames:
            self.get_logger().warn(f'crosswalk heading: {n} standing frames, not used')
            return
        a = np.array(self.cw_samples)
        ref = math.atan2(float(np.median(np.sin(a))), float(np.median(np.cos(a))))
        spread = math.degrees(float(np.percentile(np.abs([wrap(x - ref) for x in a]), 80)))
        self.cw_ref = (ref, self.now(), self.odom_dist, n, spread, tuple(self.turns))
        self.get_logger().info(f'crosswalk heading: map - odom {math.degrees(ref):+.1f} deg '
                               f'({n} standing frames, 80% within {spread:.1f} deg)')

    def cw_usable(self, now):
        r = self.cw_ref
        return self.cw_on and r is not None and now - r[1] <= self.cw_max_age \
            and self.odom_dist - r[2] <= self.cw_max_dist

    def odom_fresh(self, now):
        return self.odom_at is not None and now - self.odom_at <= self.odom_timeout

    def yaw_callback(self, msg):
        if len(msg.data) >= 2:
            self.yaw = (float(msg.data[0]), float(msg.data[1]))
            self.centre_y = float(msg.data[2]) if len(msg.data) >= 3 else float('nan')
            self.yaw_at = now = self.now()
            if all(math.isfinite(a) for a in self.yaw):
                self.yaw_hist.append((now, self.yaw[0], self.yaw[1], self.lane_state,
                                      self.centre_y))
                self.yaw_hist = [e for e in self.yaw_hist if e[0] >= now - 2.0]
                if self.cw_on and self.master_mode == self.cw_mode:
                    self.cw_sample(now, self.yaw[0], self.yaw[1], self.lane_state)
                self.heading_correction(now)
                self.side_correction(now)

    def driving(self):
        """corrections only while driving forward: 2026-11-19 a heading correction of -5 deg
        while the car stood (sign too late) turned the map about the start 3.6 m back and put
        the car 0.3 m over the outer line on the map (it was in the lane): the reverse refused."""
        return self.mode in (DRIVE, SQUEEZE) and self.v_cmd > 0.05

    def heading_correction(self, now):
        """Between hc_x_min and hc_x_max every line on the road (white dashed, outer yellow)
        is parallel to the map x axis, so whatever lane lines lane_follow tracks give the
        road direction: heading = pi - yaw. A start heading a few degrees off rotates the
        whole map about the start point (2026-11-18: the car ran ~0.2 m left of the map at
        the island and was right again back near the entry). Take a small share of the
        difference per frame, turning the map frame about the car (the car does not jump)."""
        if not self.hc_on or self.anchor is None or self.state != MissionState.RUNNING \
                or not self.odom_fresh(now) or not self.driving():
            return
        near, ahead = self.yaw
        # both lines seen and nearly parallel only: 'lost' / 'single' repeated stale 25-37 deg
        # yaws and the correction ran up to -8 deg (2026-11-19 08:43)
        if abs(near - ahead) > self.hc_agree or self.lane_state not in self.lc_states \
                or max(abs(near), abs(ahead)) > self.hc_yaw_max:
            return
        pose = self.to_map(self.odom)
        if not self.hc_x[0] < pose[0] < self.hc_x[1]:
            return
        meas = wrap(math.pi - math.radians(0.5 * (near + ahead)))
        err = wrap(meas - pose[2])
        if abs(math.degrees(err)) > self.hc_max:
            return
        room = math.radians(self.hc_total_max) - abs(math.radians(self.hc_total) + self.hc_pending)
        if room <= 0.0 and (err > 0) == (self.hc_total > 0):
            return                                  # total kept within +-hc_total_max
        self.hc_pending += self.hc_gain * err
        self.hc_count += 1
        if abs(math.degrees(self.hc_pending)) >= self.hc_step:
            # about the start point (a wrong start heading turned the whole map about it, so
            # this also takes back the side offset built up so far); in steps, the obstacle
            # grid turns with the frame (tiny steps would only blur it)
            piv = self.start_xy
            self.rotate_map(self.hc_pending, piv)
            self.map.rotate_about(piv[0], piv[1], self.hc_pending)
            self.hist = [self.turn_point(h, piv, self.hc_pending) for h in self.hist]
            self.hc_total += math.degrees(self.hc_pending)
            self.hc_pending = 0.0
            self.last_plan = -1e9

    def side_correction(self, now):
        """lane_follow's lane centre is +c m left of the car (left = -y here), the lane
        centres on the drawing are fixed: measured car y = lane centre y + c, taking the lane
        whose centre is nearest to where the map thinks the car is. Only with both lines seen,
        nearly parallel, where all lines are straight. Small share per frame, the map frame
        (and what was seen) moves along y in whole cells."""
        if not self.lc_on or self.anchor is None or self.state != MissionState.RUNNING \
                or not self.odom_fresh(now) or not self.driving():
            return
        c = self.centre_y
        if not math.isfinite(c) or max(abs(a) for a in self.yaw) > self.lc_yaw_max:
            return
        pose = self.to_map(self.odom)
        isl = self.lci_x[0] < pose[0] < self.lci_x[1]
        states, lmax, gain = ((self.lci_states, self.lci_max, self.lci_gain) if isl else
                              (self.lc_states, self.lc_max, self.lc_gain))
        if self.lane_state not in states:
            return
        if not isl and not any(a < pose[0] < b for a, b in self.lc_ranges):
            return
        if isl:   # the sign's road, else the nearer one
            lanes = (self.map.side,) if self.map.side in (1, 3) else (1, 3)
        else:
            lanes = (1, 3) if pose[0] < R3.ISL_TIP else (1, 2, 3)
        err = min(((R3.LANE_Y[k] + c) - pose[1] for k in lanes), key=abs)
        if abs(err) > lmax:
            return
        self.lc_pending += gain * err
        self.lc_count += 1
        if abs(self.lc_pending) >= self.map.res:
            dy = round(self.lc_pending / self.map.res) * self.map.res
            (o, (mx, my, mth)) = self.anchor
            self.anchor = (o, (mx, my + dy, mth))
            self.start_xy = (self.start_xy[0], self.start_xy[1] + dy)
            self.map.shift_y(dy)
            self.hist = [(h[0], h[1] + dy, h[2]) for h in self.hist]
            self.lc_total += dy
            self.lc_pending -= dy
            self.last_plan = -1e9

    @staticmethod
    def turn_point(h, pose, dth):
        c, s = math.cos(dth), math.sin(dth)
        ax, ay = h[0] - pose[0], h[1] - pose[1]
        return (pose[0] + c * ax - s * ay, pose[1] + s * ax + c * ay, wrap(h[2] + dth))

    def rotate_map(self, dth, pose):
        """turn the odom -> map transform by dth about the map point pose[:2]."""
        (o, (mx, my, mth)) = self.anchor
        c, s = math.cos(dth), math.sin(dth)
        px, py = pose[0], pose[1]
        # new map point = R(dth) (old map point - car) + car, for the anchor too
        ax, ay = mx - px, my - py
        self.anchor = (o, (px + c * ax - s * ay, py + s * ax + c * ay, wrap(mth + dth)))

    def yaw_known(self, now):
        return self.yaw_at is not None and now - self.yaw_at <= self.yaw_timeout \
            and all(math.isfinite(a) for a in self.yaw)

    def drivable_callback(self, msg):
        cls = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width]
        roi = cls[int(msg.height * self.roi_top):]
        left, mine, right = (int(np.count_nonzero(roi == c)) for c in (2, 1, 3))
        self.dbg_px = (left, mine, right)
        now = self.now()
        if left >= self.side_min or right >= self.side_min:
            self.side_seen_at = now
        if self.state != MissionState.IDLE:
            self.lane_count = 0
            return
        three = left >= self.side_min and right >= self.side_min and mine >= self.drivable_min \
            and self.yaw_known(now) and all(abs(a) <= self.align_max for a in self.yaw)
        self.lane_count = self.lane_count + 1 if three else 0
        if self.lane_count >= self.lane_frames and not self.in_cooldown(now) \
                and self.odom_fresh(now):
            self.request_at = now
            self.set_state(MissionState.REQUEST, f'3 lanes x{self.lane_count}')

    def stop_line_callback(self, msg):
        mask = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width]
        rows = np.flatnonzero(mask.max(axis=1))
        n_px = int(np.count_nonzero(mask))
        reached = n_px >= self.stop_min_px and len(rows) and \
            rows[-1] >= self.stop_row * (msg.height - 1)
        self.line_seen = (self.now(), float(rows[-1]) / max(msg.height - 1, 1) if len(rows)
                          else -1.0, n_px)
        self.line_count = self.line_count + 1 if reached else 0
        if self.line_count != self.stop_frames or self.odom is None:
            return
        self.line_event = (self.odom, self.odom_dist)
        if self.anchor is not None and self.line_fixed is None \
                and self.odom_dist - self.anchor_dist <= self.line_after:
            self.fix_by_line()

    def signs_callback(self, msg):
        best = None
        for det in msg.detections:
            for hyp in det.results:
                h = hyp.hypothesis
                side = self.sign_classes.get(h.class_id)
                if side is not None and h.score >= self.sign_conf \
                        and (best is None or h.score > best[1]):
                    best = (side, h.score)
        now = self.now()
        if best is not None:
            self.sign_hist.append((now, best[0]))
        self.sign_hist = [e for e in self.sign_hist if e[0] >= now - self.sign_pre - 60.0]
        self.decide_side(now)

    def decide_side(self, now):
        if self.anchor is None or self.map.side is not None:
            return
        since = self.active_since - self.sign_pre
        a = sum(1 for t, s in self.sign_hist if t >= since and s == 1)
        b = sum(1 for t, s in self.sign_hist if t >= since and s == 3)
        self.side_counts = (a, b)
        if a + b >= self.sign_decide and max(a, b) >= self.sign_ratio * (a + b):
            self.map.side = 1 if a > b else 3
            self.map.update()
            self.last_plan = -1e9                  # plan again at once
            self.get_logger().info(f'signs A {a} / B {b} -> {"A" if a > b else "B"} road')

    def scan_points(self, msg):
        r = np.asarray(msg.ranges, dtype=np.float64)
        a = msg.angle_min + np.arange(len(r)) * msg.angle_increment + self.laser[2]
        hit = np.isfinite(r) & (r > msg.range_min) & (r < min(msg.range_max, self.scan_range))
        rr = np.where(hit, r, self.scan_range)
        x = rr * np.cos(a) + self.laser[0]
        y = rr * np.sin(a) + self.laser[1]
        hit &= np.hypot(x, y) > self.self_r
        return x, y, hit

    def scan_callback(self, msg):
        x, y, hit = self.scan_points(msg)
        hx, hy = x[hit], y[hit]
        self.last_scan_pts = (hx, hy)
        g = (hx > self.front_x) & (hx < self.front_x + self.guard_dist) & \
            (np.abs(hy) < self.guard_half)
        self.guard_scans = self.guard_scans + 1 if np.count_nonzero(g) >= self.guard_min else 0
        if self.anchor is None or self.odom is None:
            return
        mx, my, mth = self.to_map(self.odom)
        c, s = math.cos(mth), math.sin(mth)
        wx, wy = mx + c * x - s * y, my + s * x + c * y
        self.map.add_hits(wx[hit], wy[hit])
        self.dbg_scan = (wx[hit], wy[hit])
        ox = mx + c * self.laser[0] - s * self.laser[1]
        oy = my + s * self.laser[0] + c * self.laser[1]
        self.map.add_seen(ox, oy, wx, wy)

    # ============================================================== frames
    def to_map(self, pose):
        (ox, oy, oth), (mx, my, mth) = self.anchor
        d = mth - oth
        c, s = math.cos(d), math.sin(d)
        dx, dy = pose[0] - ox, pose[1] - oy
        return (mx + c * dx - s * dy, my + s * dx + c * dy, wrap(pose[2] + d))

    def pin(self, now, x=None, since=None):
        """the current odom pose = the start pose on the drawing. x: known (aligned to the
        entry line), since: lane_follow frames from then on (standing) for yaw and centre."""
        y = R3.MID_Y
        th = math.pi
        if since is not None:
            fr = [(0.5 * (n + a), c) for t, n, a, st, c in self.yaw_hist
                  if t >= since and st in self.lc_states
                  and abs(0.5 * (n + a)) <= self.start_yaw_max]
            if fr and self.use_lane_yaw:
                th = wrap(math.pi - math.radians(float(np.median([f[0] for f in fr]))))
            cs = [c for _, c in fr if math.isfinite(c) and abs(c) < 0.2]
            if cs and self.use_lane_offset:
                y = R3.MID_Y + float(np.median(cs))
        elif self.yaw_known(now):
            if self.use_lane_offset and math.isfinite(self.centre_y) and abs(self.centre_y) < 0.2:
                y = R3.MID_Y + self.centre_y   # lane centre +c left of the car; left = -y here
            cs = [c for t, n, a, st, c in self.yaw_hist
                  if t >= now - self.start_yaw_sec and st in self.lc_states
                  and math.isfinite(c) and abs(c) < 0.2]
            if self.use_lane_offset and len(cs) >= 3:
                y = R3.MID_Y + float(np.median(cs))   # one frame jumped a few cm while driving
            if self.use_lane_yaw:
                # median of the last start_yaw_sec of ahead + near yaw (one near frame was
                # 1-3 deg off right after the curve before the section, always the same way)
                # only frames with both lines and nearly parallel; none -> straight (180 deg):
                # 2026-11-19 08:43 the median took frames from placing the car, 12 deg off
                rec = [0.5 * (n + a) for t, n, a, st, _ in self.yaw_hist
                       if t >= now - self.start_yaw_sec and st in self.lc_states
                       and abs(0.5 * (n + a)) <= self.start_yaw_max]
                if rec:
                    th = wrap(math.pi - math.radians(float(np.median(rec))))
        src = ''
        if x is None and since is None and self.cw_usable(now):
            r = self.cw_ref
            th = wrap(self.odom[2] + r[0])
            t_odom, t_imu = (math.degrees(a - b) for a, b in zip(self.turns, r[5]))
            rec = [0.5 * (n + a) for t, n, a, st, _ in self.yaw_hist
                   if t >= now - self.start_yaw_sec and st in self.lc_states
                   and abs(0.5 * (n + a)) <= self.start_yaw_max]
            lane = (f'{math.degrees(wrap(math.pi - math.radians(float(np.median(rec))))):.1f}'
                    if rec else '-')
            src = (f', heading from the crosswalk ({now - r[1]:.0f} s, '
                   f'{self.odom_dist - r[2]:.1f} m ago; turn since: odom {t_odom:+.1f}, '
                   f'imu {t_imu:+.1f} deg, source {self.heading_src} x{self.turn_scale:.3f}; '
                   f'lane_follow says {lane} deg)')
            self.cw_ref = None                    # one run per crosswalk stop
        x0 = self.trigger_x if x is None else x
        self.anchor = (self.odom, (x0, y, th))
        self.start_xy = (x0, y)               # map point of the start (heading pivot)
        self.anchor_dist = self.odom_dist
        self.hc_total, self.hc_count, self.hc_pending = 0.0, 0, 0.0
        self.lc_total, self.lc_count, self.lc_pending = 0.0, 0, 0.0
        self.line_fixed = None
        ev = self.line_event
        txt = ''
        if x is not None:
            self.line_fixed = 0.0                 # aligned to the line: no later line fix
            txt = ', aligned to the entry line, standing'
        elif ev is not None and self.odom_dist - ev[1] <= self.line_before:
            txt = self.fix_by_line()
        self.get_logger().info(f'start: map ({x0:.2f}, {y:.2f}, '
                               f'{math.degrees(th):.1f} deg){src}{txt}')

    def fix_by_line(self):
        pose, _ = self.line_event
        bx = self.to_map(pose)[0]
        dx = (R3.ENTRY_X + self.line_dist) - bx
        if abs(dx) > self.line_max_fix:
            self.line_fixed = 0.0
            self.get_logger().warn(f'entry line: correction {dx:+.2f} m too large, ignored')
            return ''
        (o, (mx, my, mth)) = self.anchor
        self.anchor = (o, (mx + dx, my, mth))
        self.start_xy = (self.start_xy[0] + dx, self.start_xy[1])
        self.map.shift_x(dx)
        self.hist = [(h[0] + dx, h[1], h[2]) for h in self.hist]
        self.line_fixed = dx
        self.last_plan = -1e9
        txt = f', entry line: x {dx:+.2f} m'
        if self.state == MissionState.RUNNING and self.odom_dist > self.anchor_dist + 0.01:
            self.get_logger().info('entry line after the start' + txt)
        return txt

    # ============================================================== mission
    def on_start(self, now):
        self.end_approach('3road started')
        self.side_seen_at = now          # the side-lane exit counts from the start
        self.map.reset()
        self.path = None
        self.mode = WAIT
        self.hist = []
        self.rev_dist = 0.0
        self.blocked = 0
        self.last_plan = -1e9
        self.side_counts = (0, 0)
        self.anchor = None
        self.align = None
        if self.odom is None:
            return
        if self.align_on and not self.cw_usable(now):
            self.align = {'t0': now, 'dist0': self.odom_dist, 'dir': 0, 'hits': 0,
                          'settle': None}
            self.mode = ALIGN
            self.set_phase(ALIGN, 'stop, entry line to the reference row, measure standing')
            return
        self.pin(now)
        self.decide_side(now)

    def align_step(self, now):
        """start alignment (see align_* params). Returns True while it runs."""
        a = self.align
        moved = self.odom_dist - a['dist0']
        if a['settle'] is not None:
            self.publish(0.0, 0.0)
            if now - a['settle'] < self.align_settle:
                return True
            self.pin(now, x=R3.ENTRY_X + self.line_dist, since=a['settle'] + 0.1)
            self.end_align(now, f'line at the row after {moved:.2f} m')
            return False
        if now - a['t0'] > self.align_timeout or (a['dir'] < 0 and moved > self.align_max_back):
            self.publish(0.0, 0.0)
            self.pin(now)
            self.end_align(now, f'no entry line ({now - a["t0"]:.1f} s, {moved:.2f} m): '
                                'started as without alignment')
            return False
        ls = self.line_seen
        fresh = ls is not None and now - ls[0] <= self.align_line_fresh
        visible = fresh and ls[2] >= self.stop_min_px and ls[1] >= 0.0
        if a['dir'] == 0:
            if not fresh:
                self.publish(0.0, 0.0)            # wait for a mask
                return True
            # line ahead of the row: forward to it; passed or closer: back until it is there
            a['dir'] = 1 if visible and ls[1] < self.stop_row else -1
            self.get_logger().info('align: ' + ('line ahead, creeping forward' if a['dir'] > 0
                                                else 'line passed, backing up'))
        done = visible and (ls[1] >= self.stop_row if a['dir'] > 0 else ls[1] <= self.stop_row)
        a['hits'] = a['hits'] + 1 if done else 0
        if a['hits'] >= self.stop_frames:
            self.publish(0.0, 0.0)
            a['settle'] = now
            return True
        v = self.align_fwd if a['dir'] > 0 else -self.align_back
        self.publish(v, 0.0)
        self.v_cmd, self.k_cmd = v, 0.0
        return True

    def end_align(self, now, why):
        self.align = None
        self.mode = WAIT
        self.last_plan = -1e9
        self.side_seen_at = now
        self.get_logger().info(f'align done: {why}')
        self.decide_side(now)

    def on_stop(self, now):
        self.anchor = None
        self.path = None
        self.publish(0.0, 0.0)

    def end_approach(self, why):
        if self.slow_from is not None:
            self.slow_from = None
            self.speed_limit = 0.0
            self.get_logger().info(f'approach speed off: {why}')

    def tick(self, now):
        if self.slow_from is not None and self.state == MissionState.IDLE:
            t0, d0 = self.slow_from
            if now - t0 > self.cw_max_age or self.odom_dist - d0 > self.cw_max_dist:
                self.end_approach(f'{now - t0:.0f} s / {self.odom_dist - d0:.1f} m, no start')
        if self.state == MissionState.REQUEST and now - self.request_at > self.request_timeout:
            self.set_state(MissionState.IDLE, f'not granted within {self.request_timeout:.0f} s')
        if now - self.last_debug >= self.debug_period and self.pub_debug.get_subscription_count():
            self.last_debug = now
            self.publish_debug()

    def publish(self, v, w):
        t = Twist()
        t.linear.x, t.angular.z = float(v), float(w)
        self.pub_cmd.publish(t)

    def plan_ok(self, p, info, pose, min_partial=None):
        """a path to the exit, or a partial one that still goes min_partial m on."""
        if p is None:
            return None
        mp = self.min_partial if min_partial is None else min_partial
        if info.get('partial') and pose[0] - info['reach_x'] < mp:
            return None
        return p

    def replan(self, now, pose):
        self.map.update()
        p, info = self.dp.plan(self.map, pose)
        p = self.plan_ok(p, info, pose)
        mode = DRIVE
        # the back-off is used up (reverse_max or no room behind) and still nothing: take
        # any partial path that goes 0.3 m on (squeeze) instead of standing in REVERSE until
        # the time limit (2026-11-19: 'backed 1.00 m' for 25 s, twice)
        spent = self.mode == REVERSE and not self.rev_can
        if p is None:
            p, info2 = self.dp.plan(self.map, pose, r_min=self.squeeze_r)
            p = self.plan_ok(p, info2, pose, min_partial=0.3 if spent else None)
            info2['ms'] += info['ms']
            info = info2
            mode = SQUEEZE
        if p is None and spent:
            # still nothing: another back-off from here (rev_can is checked again behind)
            self.rev_dist, self.rev_can = 0.0, True
            if now - getattr(self, 'last_again_log', 0.0) > 2.0:
                self.last_again_log = now
                self.get_logger().info('back-off used up and still blocked: backing off again')
        self.blocked = self.blocked + 1 if p is None else 0
        if p is None:
            # back off only when two plans in a row found nothing (one miss: hold still)
            mode = WAIT if self.blocked < 2 else REVERSE
        if mode != self.mode:
            why = f'x {pose[0]:.2f} y {pose[1]:.2f}'
            if p is None:
                why += ', ' + self.blocked_text(pose)
            self.set_phase(mode, why)
        elif p is None and now - self.last_block_log > 2.0:
            self.last_block_log = now
            self.get_logger().info(f'still blocked ({mode}, backed {self.rev_dist:.2f} m): '
                                   + self.blocked_text(pose))
        if mode in (DRIVE, SQUEEZE):
            self.rev_dist = 0.0
        if mode == REVERSE and self.mode != REVERSE:
            self.rev_dist, self.rev_can = 0.0, True
        self.mode = mode
        self.path = R3.densify(p) if p is not None else None
        self.reach_x = info.get('reach_x') if p is not None and info.get('partial') else None
        self.plan_ms = info['ms']
        self.last_plan = now

    def blocked_text(self, pose):
        """what blocks: obstacle cells in the 1.2 m ahead (map x, y ranges); nearest = in the
        car's own corridor (|y - car y| < 0.25 m)."""
        m = self.map
        ahead = m.occupied() & (m.cx < pose[0]) & (m.cx > pose[0] - 1.2)
        n = int(np.count_nonzero(ahead))
        if n == 0:
            return 'no obstacle cells within 1.2 m ahead (yellow lines / island / side wall?)'
        xs, ys = m.cx[ahead], m.cy[ahead]
        lane = np.abs(ys - pose[1]) < 0.25
        near = f'{float(pose[0] - xs[lane].max()):.2f} m' if lane.any() else 'none'
        return (f'{n} obstacle cells within 1.2 m ahead (x {xs.min():.2f}-{xs.max():.2f}, '
                f'y {ys.min():.2f}-{ys.max():.2f}), nearest in my corridor {near}')

    def control(self):
        now = self.now()
        if self.state != MissionState.RUNNING or not self.active:
            return
        if self.align is not None:
            if not self.odom_fresh(now):
                self.publish(0.0, 0.0)
                return
            if self.align_step(now):
                return
        if self.anchor is None:
            if self.odom is None:
                self.publish(0.0, 0.0)
                return
            self.pin(now)
        if not self.odom_fresh(now):
            self.publish(0.0, 0.0)
            self.set_phase(WAIT, 'no fresh odom')
            return
        pose = self.to_map(self.odom)
        if pose[0] < self.exit_x:
            self.publish(0.0, 0.0)
            self.finish(f'exit reached (x {pose[0]:.2f}, y {pose[1]:.2f})')
            return
        if now - self.active_since > self.max_sec:
            self.publish(0.0, 0.0)
            self.finish(f'{self.max_sec:.0f} s: lane follow')
            return
        if self.side_lost_sec > 0.0 and now - self.side_seen_at > self.side_lost_sec:
            # camera-side exit, independent of the map: neither side lane for side_lost_sec
            self.publish(0.0, 0.0)
            self.finish(f'no side lane for {self.side_lost_sec:.0f} s (map x {pose[0]:.2f} '
                        f'y {pose[1]:.2f}): lane follow')
            return
        if self.guard_scans >= 2 and self.v_cmd > 0.0:
            self.last_plan = -1e9                    # something right ahead: stop, plan again
            self.get_logger().info(f'guard: object within {self.guard_dist:.2f} m',
                                   throttle_duration_sec=1.0)
            self.publish(0.0, 0.0)
            self.v_cmd = 0.0
        backing = self.mode == REVERSE and self.rev_dist < self.rev_min and self.rev_can
        if now - self.last_plan >= self.plan_period and not backing:
            # (a started backoff runs rev_min m first: replanning after a few cm drove the car
            #  straight back into the same spot, sim 2026-11-18)
            self.replan(now, pose)
        v, k = 0.0, 0.0
        if self.mode in (DRIVE, SQUEEZE) and self.path is not None and len(self.path) > 1:
            if not self.hist or math.hypot(self.hist[-1][0] - pose[0],
                                           self.hist[-1][1] - pose[1]) > 0.05:
                self.hist.append(pose)
                self.hist = self.hist[-200:]
            ahead = self.path[:8]
            ca = float(self.map.pose_clear(ahead[:, 0], ahead[:, 1],
                                           np.full(len(ahead), pose[2])).min())
            v = self.drive_speed if ca >= self.tight_clear else self.slow_speed
            k, _ = R3.pure_pursuit(pose, self.path,
                                   self.ld if v > self.slow_speed else self.ld_slow)
            if abs(k) > self.curve_k:
                v = min(v, self.curve_speed)
            if self.reach_x is not None:
                v = min(v, self.partial_speed)
            if self.island_speed > 0.0 and pose[0] < self.island_zone_x:
                v = min(v, self.island_speed)
            if self.mode == SQUEEZE:
                v = self.creep_speed
            if self.guard_scans >= 2:
                v = 0.0
                self.guard_since = self.guard_since or now
                if now - self.guard_since > self.guard_hold:
                    # the path passes too close for the guard: back off, plan again
                    self.set_phase(REVERSE, f'guard {self.guard_hold:.1f} s, x {pose[0]:.2f} '
                                            f'y {pose[1]:.2f}')
                    self.mode, self.rev_dist, self.rev_can = REVERSE, 0.0, True
                    self.blocked, self.guard_since = 2, None
            else:
                self.guard_since = None
        elif self.mode == REVERSE:
            while len(self.hist) > 1 and math.hypot(self.hist[-1][0] - pose[0],
                                                    self.hist[-1][1] - pose[1]) < 0.12:
                self.hist.pop()
            back = np.array([h[:2] for h in self.hist[::-1]])
            nx = pose[0] - 0.05 * math.cos(pose[2])
            ny = pose[1] - 0.05 * math.sin(pose[2])
            ok = self.rev_dist < self.rev_max and \
                float(self.map.pose_clear(nx, ny, pose[2], obs_only=True)) >= self.squeeze_r
            # objects only: the lines are where we just drove (a map error near a line held
            # the car for 16 s, backed 0.00 m, 2026-11-19)
            self.rev_can = ok
            if ok:
                kr = 0.0                         # nothing driven yet: straight back
                if len(back) > 1:
                    kr, _ = R3.pure_pursuit((pose[0], pose[1], pose[2] + math.pi), back, 0.2)
                v, k = -self.rev_speed, -kr
                self.rev_dist += self.rev_speed / self.control_rate()
        self.v_cmd, self.k_cmd = v, k
        w = max(-self.w_max, min(self.w_max, self.w_gain * v * k))
        self.publish(v, w)
        if now - self.last_log > self.track_log_period:
            self.last_log = now
            side = {None: '-', 1: 'A', 3: 'B'}[self.map.side]
            self.reason = (f'{self.mode} x {pose[0]:.2f} y {pose[1]:.2f} '
                           f'{math.degrees(pose[2]):.0f} deg, v {v:.2f}, '
                           f'plan {self.plan_ms:.0f} ms, side {side}')
            # localisation trace: where the map puts the car and what corrected it
            ny, ay = self.yaw
            part = '' if self.reach_x is None else f' partial->{self.reach_x:.2f}'
            self.get_logger().info(
                f'track: {self.mode}{part} x {pose[0]:.2f} y {pose[1]:.2f} '
                f'{math.degrees(pose[2]):.1f} deg v {v:+.2f} | fix hdg {self.hc_total:+.1f} deg '
                f'({self.hc_count}) side {self.lc_total:+.3f} m ({self.lc_count}) line '
                f'{"-" if self.line_fixed is None else f"{self.line_fixed:+.2f}"} | lane '
                f'{self.lane_state or "-"} yaw {ny:+.1f}/{ay:+.1f} c {self.centre_y:+.3f} | '
                f'sign {side} A{self.side_counts[0]}/B{self.side_counts[1]}')

    def control_rate(self):
        return float(self.get_parameter('control_rate_hz').value)

    # ============================================================== debug
    def publish_debug(self):
        """Top view of the section (the track drawing, the car drives right to left):
        grey road (darker = not seen by the lidar yet), red = obstacle cells (map), dark red
        = blocked for the planner (inflation not drawn; wrong island side), white dots = this
        scan, cyan = planned path, green = driven track + car, yellow = yellow lines."""
        if cv2 is None:
            return
        m = self.map
        sc = 110
        W, H = int((R3.X1 - R3.X0) * sc), int((R3.Y1 - R3.Y0) * sc)
        img = np.full((H + 58, W, 3), 25, np.uint8)
        top = img[:H]

        def f(p):
            return (int(round((p[0] - R3.X0) * sc)), int(round((R3.Y1 - p[1]) * sc)))

        def lay(mask, color):
            mm = cv2.resize(mask.astype(np.uint8)[::-1], (W, H), interpolation=cv2.INTER_NEAREST)
            top[mm > 0] = color

        lay(m.free0, (60, 60, 60))                          # road, not seen yet
        lay(m.free0 & m.seen, (105, 105, 105))              # road the lidar has seen
        lay(m.free0 & ~m.free, (50, 40, 120))               # blocked (side wall / objects)
        lay(m.occupied(), (0, 0, 255))                      # obstacle cells
        for x in np.arange(2.0, 9.01, 1.0):                 # 1 m ticks
            cv2.line(top, f((x, R3.Y0)), f((x, R3.Y0 + 0.06)), (150, 150, 150), 1)
            cv2.putText(top, f'{x:.0f}', (f((x, 0))[0] + 2, H - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        for y in (0.68, 1.20):                              # white dashed lane lines
            for x in np.arange(4.295, 8.35, 0.35):
                cv2.line(top, f((x, y)), f((x + 0.2, y)), (220, 220, 220), 1)
        cv2.line(top, f((R3.ENTRY_X, 0.69)), f((R3.ENTRY_X, 1.19)), (240, 240, 240), 2)
        cv2.line(top, f((self.exit_x, 0.69)), f((self.exit_x, 1.19)), (0, 255, 255), 1)
        cv2.polylines(top, [np.array([f(p) for p in R3.ROAD], np.int32)], True, (0, 200, 255), 2)
        cv2.polylines(top, [np.array([f(p) for p in R3.ISLAND], np.int32)], True,
                      (0, 200, 255), 2)
        cv2.putText(top, 'A', f((3.3, 0.33)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if m.side == 1 else (170, 170, 170), 2)
        cv2.putText(top, 'B', f((3.3, 1.38)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if m.side == 3 else (170, 170, 170), 2)
        if self.hist:
            cv2.polylines(top, [np.array([f(p) for p in self.hist], np.int32)], False,
                          (0, 150, 0), 1)
        if self.path is not None:
            cv2.polylines(top, [np.array([f(p) for p in self.path], np.int32)], False,
                          (255, 255, 0), 2)
        if self.dbg_scan is not None:
            for px, py in zip(*self.dbg_scan):
                q = f((px, py))
                if 0 <= q[0] < W and 0 <= q[1] < H:
                    top[q[1], q[0]] = (255, 255, 255)
        if self.anchor is not None and self.odom is not None:
            x, y, th = self.to_map(self.odom)
            c, s = math.cos(th), math.sin(th)
            box = [(0.16, 0.11), (0.16, -0.11), (-0.16, -0.11), (-0.16, 0.11)]
            pts = np.array([f((x + a * c - b * s, y + a * s + b * c)) for a, b in box], np.int32)
            cv2.polylines(top, [pts], True, (0, 255, 0), 2)
            cv2.line(top, f((x, y)), f((x + 0.25 * c, y + 0.25 * s)), (0, 255, 0), 2)
        side = {None: '-', 1: 'A', 3: 'B'}[m.side]
        fix = '-' if self.line_fixed is None else f'{self.line_fixed:+.2f} m'
        names = {0: 'IDLE', 1: 'ARMED', 2: 'REQUEST', 3: 'RUNNING', 4: 'DONE'}
        pose = '-'
        if self.anchor is not None and self.odom is not None:
            x, y, th = self.to_map(self.odom)
            pose = f'x {x:.2f} y {y:.2f} {math.degrees(th):.0f} deg'
        part = '' if self.reach_x is None else f' (partial to x {self.reach_x:.2f})'
        l1 = (f'{names.get(self.state, "?")}/{self.mode}{part} | {pose} | v {self.v_cmd:+.2f} '
              f'k {self.k_cmd:+.1f} | plan {self.plan_ms:.0f} ms')
        l2 = (f'sign {side} (A {self.side_counts[0]} / B {self.side_counts[1]}) | entry line '
              f'{fix} | heading fix {self.hc_total:+.1f} deg ({self.hc_count}) | side fix '
              f'{self.lc_total:+.3f} m ({self.lc_count}) | '
              f'lanes L/M/R {self.dbg_px} | guard {self.guard_scans}')
        cv2.putText(img, l1, (6, H + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, l2, (6, H + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height, msg.width = img.shape[:2]
        msg.encoding = 'bgr8'
        msg.step = img.shape[1] * 3
        msg.data = array('B', img.tobytes())   # bytes: rclpy checks every byte (~0.3 s)
        self.pub_debug.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ThreeRoadNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
