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
               distances, so a backed-off object still counts as ahead (no loop).
               Two lanes across a blocked middle lane: my lane and the middle lane object
               both backoff_clear_mid away, else back off to that and start that same move
               without deciding again
     path      (change_mode path) a quintic S curve from the car to the target lane centre
               (lane frame: start offset = lane centre y, start heading = lane yaw, end
               parallel), path_len1 / path_len2 long (shorter, within path_kmax, when the
               swept car body would come closer than obj_padding to the object in my lane;
               none keeps the padding: back off path_backoff_step more, up to just below the
               dodge distance, then go anyway), then path_ext straight; tracked with
               pure pursuit on /odometry/filtered (EKF wheel + IMU; the wheel odom yaw alone
               is ~2x too large). Done at path_len along the path -> post_stop. Without a
               fresh odom it falls back to the timed steps below.
     turn / (mid) / counter   (change_mode timed) open loop, measured times; turn AND counter
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
     (path counts as a change like turn / mid / counter)
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
from nav_msgs.msg import Odometry
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
PATH = 'path'
CHANGING = (TURN, MID, COUNTER, PATH)
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
        d('sign_pre_sec', 15.0)          # s, signs seen this long before the section count too
        # lane 2 ends at the island (the sign stands on its tip; A road left, B road right):
        # no sign judged after sign_wait_dist m in the section -> this lane is the goal
        d('sign_wait_dist', 2.5)         # m driven in the section (odom)
        d('edge_margin', 0.05)           # m, body corners inside the outer yellow lines on a curve
        d('backoff_budget', 1.0)         # m reversed per section at most (curve backoffs)
        d('hold_max_sec', 3.0)           # s stopped waiting for a clear curve, then lane follow
        d('side_pass_dist', 0.6)         # m from the lidar: a lane 2 object nearer is about to be
                                         #    beside: lane 2 is still the dodge from the goal lane
        d('sign_default_lane', 0)        # 1 (A) / 3 (B) / 0: the side lane the car is in, from
                                         #    lane 2 the free side (both free: lane 1)
        d('sign_ratio', 0.8)             # the winning sign needs this share of the A + B frames
        # yellow island at the end of lane 2 (never drive into it): its tip island_dist m along
        # the lane from where the section was entered (bag 05:10: sign 4.4 .. 4.9 m), lane 2
        # centre; widens from the tip to island_half over island_taper m. It is an object for
        # every curve; a curve it blocks may back off farther (island_backoff_*)
        d('path_abort_frac', 0.3)        # a curve leading away from a new sign goal: abort up to
                                         #    this share of it (still near the start lane)
        d('island_on', True)
        d('island_dist', 4.35)
        d('island_half', 0.20)
        d('island_taper', 0.25)
        d('island_len', 1.5)
        d('sign_fallback_ahead', 1.3)    # m before the island tip: no sign yet -> fallback goal lane
                                         #    (before that, lane 2 is kept: both roads one lane away)
        d('rear_free_min', 0.15)         # m of free room behind needed to start a backoff
        d('rev_rearm_dist', 0.6)         # m driven forward after a backoff stopped by an object
                                         #    behind before backing off is allowed again
        d('island_past', 0.6)            # m after the estimated tip: no more lane changes
        d('island_backoff_max', 0.8)     # m per backoff when only the island blocks the curve
        d('island_budget', 1.2)          # m more reversing per section for that
        d('island_slow_dist', 2.2)       # m before the tip: no sign yet / not in the goal lane ->
        d('island_slow_speed', 0.25)     #    lane follow this slow
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
        # path tracking change (change_mode 'path'), else the timed steps above
        d('change_mode', 'path')
        d('odom_topic', '/odometry/filtered')
        d('odom_timeout', 0.3)           # s, older pose = no odom (timed fallback / stop)
        d('lane_width', 0.5)             # m, lateral shift per lane
        d('path_len1', 1.3)              # m along the lane, one-lane change
        d('path_len2', 1.9)              # m, two-lane change
        d('path_ext', 0.5)               # m straight after the S curve (look-ahead room)
        d('path_lookahead', 0.30)        # m, pure pursuit look-ahead
        d('path_w_gain', 1.0)            # cmd w = gain x v x curvature (base steering mapping)
        d('path_end_tol', 0.03)          # m, done this short of path_len
        d('path_min_len', 0.6)           # m, shortest curve tried
        d('path_kmax', 1.6)              # 1/m, curvature limit of a shorter curve
        d('obj_padding', 0.10)           # m, gap kept between the car body and the object
        d('obj_depth', 0.15)             # m, assumed object depth behind its seen face
        d('car_half_width', 0.11)        # m
        d('car_rear', 0.16)              # m, base_link to the rear bumper
        d('path_backoff_step', 0.10)     # m, no curve keeps the padding: back off this more
        d('path_max_sec_scale', 2.0)     # x nominal time -> give up (stop, judge)
        d('path_speed', 0.0)             # m/s on the path (0: change_speed; the timed steps
                                         #    keep change_speed, their times were measured at it)
        d('path_over_max', 0.04)         # m, the curve may pass the target line / swing to the
                                         #    other side by this much at most (lane edge lines)
        d('path_backoff_tries', 3)       # padding backoffs before going anyway
        d('path_k_soft', 1.2)            # 1/m, curvature of the gentlest S curve tried
        d('path_th_max_deg', 55.0)       # deg, steepest heading across the lanes
        d('path_backoff_max', 0.6)       # m, backing off for a clear curve: at most this far
        d('backoff_max_dist', 0.7)       # m, any backoff: at most this far (odom)
        d('cy_ref_x', 0.43)              # m ahead of base_link where lane_follow measures the
                                         #    lane centre (BEV near edge): corrected by the heading
        d('lmap_min', 8)                 # lane map: samples before it is used
        d('lmap_fit_span', 0.8)          # m of travel before the line fit replaces the camera heading
        d('path_lead_max', 0.5)          # m, straight lead-in before the S curve: tried 0 .. this
        d('path_th_steep_deg', 70.0)     # deg, second heading limit tried (two lanes, tight rows)
        d('path_gap_good', 0.20)         # m, a larger body gap is not better: then the gentlest
        d('path_end_clear', 0.35)        # m straight in the target lane after the curve: free too
        d('omem_keep', 20.0)             # s, object points remembered in odom (beside / behind
                                         #    the car the lidar does not see them any more)
        d('omem_fov_deg', 55.0)          # deg, remembered points outside this half angle are used
        d('path_edge_bias', 0.03)        # m, target lane 1 / 3: aim this much toward lane 2
                                         #    (away from the outer lane line)
        d('path_resume_max', 2)          # new curves to the same target after a guard abort
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
        d('backoff_clear_mid', 1.0)      # two-lane change across a blocked middle lane
        # 7. guard / align
        d('guard_dist', 0.13)
        d('guard_dist_path', 0.13)       # m from the lidar while a curve is driven (the curve
                                         #    already keeps obj_padding)
        d('rear_clear', 0.10)            # m behind the rear bumper: a remembered object nearer
                                         #    stops any backing off (the lidar does not see behind)
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
        self.sign_pre = float(p('sign_pre_sec'))
        self.sign_hist = []          # (s, lane 1 / 3) of every A / B frame, also before the section
        self.section_at = None       # s, the section was entered
        self.sign_wait = float(p('sign_wait_dist'))
        self.side_pass_dist = float(p('side_pass_dist'))
        self.backoff_budget = float(p('backoff_budget'))
        self.edge_margin = float(p('edge_margin'))
        self.hold_max = float(p('hold_max_sec'))
        self.hold_since = None
        self.section_rev = 0.0       # m reversed in this section
        self.sign_default = int(p('sign_default_lane'))
        self.sign_ratio = float(p('sign_ratio'))
        self.island_on = bool(p('island_on'))
        self.path_abort_frac = float(p('path_abort_frac'))
        self.island_dist = float(p('island_dist'))
        self.island_half = float(p('island_half'))
        self.island_taper = max(float(p('island_taper')), 0.01)
        self.island_len = float(p('island_len'))
        self.island_backoff_max = float(p('island_backoff_max'))
        self.island_past = float(p('island_past'))
        self.sign_fallback_ahead = float(p('sign_fallback_ahead'))
        self.rear_free_min = float(p('rear_free_min'))
        self.rev_rearm = float(p('rev_rearm_dist'))
        self.rev_block_fwd = None     # forward odom distance when a backoff hit something behind
        self.backoff_room = float('inf')  # m this backoff may go (free room behind - rear_clear)
        self.island_budget = float(p('island_budget'))
        self.island_slow_dist = float(p('island_slow_dist'))
        self.island_slow_speed = float(p('island_slow_speed'))
        self.isl_ref = None          # (tip x, tip y, lane direction) in odom, set at the entry
        self.backoff_isl = False     # this backoff is for a curve only the island blocks
        self.section_dist = 0.0      # m driven in the section (odom)
        self.section_xy = None       # last odom (x, y) counted
        self.fallback = None         # goal lane picked without a sign (logged once)
        self.speed = float(p('change_speed'))
        self.w = self.speed * math.tan(float(p('steer'))) / float(p('wheelbase'))
        self.times = {LEFT: (float(p('left_turn_sec')), float(p('left_counter_sec'))),
                      RIGHT: (float(p('right_turn_sec')), float(p('right_counter_sec')))}
        self.times2 = {sd: tuple(float(p(f'{sd}2_{k}_sec')) for k in ('turn', 'mid', 'counter'))
                       for sd in (LEFT, RIGHT)}
        self.pre_stop = float(p('pre_stop_sec'))
        self.change_mode = str(p('change_mode'))
        self.odom_timeout = float(p('odom_timeout'))
        self.lane_width = float(p('lane_width'))
        self.path_len = {1: float(p('path_len1')), 2: float(p('path_len2'))}
        self.path_ext = float(p('path_ext'))
        self.path_ld = float(p('path_lookahead'))
        self.path_w_gain = float(p('path_w_gain'))
        self.path_end_tol = float(p('path_end_tol'))
        self.path_min_len = float(p('path_min_len'))
        self.path_kmax = float(p('path_kmax'))
        self.obj_padding = float(p('obj_padding'))
        self.obj_depth = float(p('obj_depth'))
        self.car_half = float(p('car_half_width'))
        self.car_rear = float(p('car_rear'))
        self.path_backoff_step = float(p('path_backoff_step'))
        self.path_max_scale = float(p('path_max_sec_scale'))
        self.path_speed = float(p('path_speed')) or self.speed
        self.path_w = self.path_speed * math.tan(float(p('steer'))) / float(p('wheelbase'))
        self.path_over_max = float(p('path_over_max'))
        self.path_backoff_tries = int(p('path_backoff_tries'))
        self.path_edge_bias = float(p('path_edge_bias'))
        self.path_k_soft = float(p('path_k_soft'))
        self.path_lead_max = float(p('path_lead_max'))
        self.path_th_steep = math.radians(float(p('path_th_steep_deg')))
        self.path_gap_good = float(p('path_gap_good'))
        self.path_end_clear = float(p('path_end_clear'))
        self.omem_keep = float(p('omem_keep'))
        self.omem_fov = math.radians(float(p('omem_fov_deg')))
        self.omem = {}               # odom cell (3 cm) -> [hits, last seen s]
        self.plan_check = (0.0, False)   # backoff: (time, any curve clear) of the last check
        self.path_th_max = math.radians(float(p('path_th_max_deg')))
        self.path_backoff_max = float(p('path_backoff_max'))
        self.backoff_max_dist = float(p('backoff_max_dist'))
        self.cy_ref_x = float(p('cy_ref_x'))
        self.lmap_min = int(p('lmap_min'))
        self.lmap_fit_span = float(p('lmap_fit_span'))
        self.lmap = []               # lane 2 centre points in odom: (X, Y, lane direction)
        self.lmap_at = (0.0, 0.0)
        self.lmap_same = 0           # samples taken standing at the last spot
        self.backoff_from = None     # odom (x, y) where the running backoff started
        self.backoff_plan = None     # (side, lanes, cy): back off until that curve is clear
        self.path_resume_max = int(p('path_resume_max'))
        self.pad_backoffs = 0        # padding backoffs for the change being started
        self.resumes = 0             # curves re-planned after a guard abort
        self.resume_after = False    # guard abort during a path: re-plan after the backoff
        self.odom = None             # (x, y, yaw) in the odom frame
        self.odom_at = None
        self.path = None             # dict: pts (N x 2 odom), s, L, idx, D, kmax, until, ahead
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
        self.backoff_clear_mid = float(p('backoff_clear_mid'))
        self.commit = None           # move fixed before a backoff: started without deciding again
        self.backoff_side = None     # lane box that must be clear up to backoff_need as well
        self.box_near = {LEFT: None, RIGHT: None}   # m from the lidar, nearest lane box point
        self.guard_dist = float(p('guard_dist'))
        self.guard_dist_path = float(p('guard_dist_path'))
        self.rear_clear = float(p('rear_clear'))
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
        self.create_subscription(Odometry, p('odom_topic'), self.odom_callback,
                                 qos_profile_sensor_data)
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
            self.lmap_add(self.yaw_at)

    def lmap_add(self, now):
        """Lane map sample: where the lane 2 centre line is in odom, taken while the car is
        parallel in a known lane (lane follow, or standing before a change)."""
        if self.lane is None or not self.odom_fresh(now) or not self.aligned(now) \
                or not math.isfinite(self.centre_y) or abs(self.centre_y) > self.box_shift_max:
            return
        if not (self.state == MissionState.ARMED
                or (self.state == MissionState.RUNNING and self.phase == PRE_STOP)):
            return
        X, Y, H = self.odom
        if self.lmap:
            lx, ly = self.lmap_at
            if math.hypot(X - lx, Y - ly) < 0.02:
                if self.lmap_same >= 5:
                    return                      # standing: a few samples of one spot are enough
                self.lmap_same += 1
            else:
                self.lmap_same = 0
        self.lmap_at = (X, Y)
        psi = math.radians(self.yaw[0])
        a = H + psi
        off = self.centre_y - self.cy_ref_x * math.tan(psi) + (self.lane - 2) * self.lane_width
        self.lmap.append((X - off * math.sin(a), Y + off * math.cos(a), a))
        del self.lmap[:-300]

    def lane_map(self):
        """(a, c): lane direction in odom and the offset of the lane 2 centre line along its
        left normal, or None. Direction: median camera heading; after lmap_fit_span m of
        travel the line fitted through the centre points instead (the camera heading is
        +-3 deg, the fit is better than 1 deg). Points 20 cm off the median are dropped
        (a wrong lane number shifts a point by a lane width)."""
        if len(self.lmap) < self.lmap_min:
            return None
        arr = np.array(self.lmap)
        da = np.arctan2(np.sin(arr[:, 2] - arr[0, 2]), np.cos(arr[:, 2] - arr[0, 2]))
        a = float(arr[0, 2] + np.median(da))
        for _ in range(2):
            along = arr[:, 0] * math.cos(a) + arr[:, 1] * math.sin(a)
            across = -arr[:, 0] * math.sin(a) + arr[:, 1] * math.cos(a)
            keep = np.abs(across - np.median(across)) < 0.2
            if np.count_nonzero(keep) < self.lmap_min or np.ptp(along[keep]) < self.lmap_fit_span:
                break
            slope = float(np.polyfit(along[keep], across[keep], 1)[0])
            if abs(slope) > math.tan(math.radians(8.0)):
                break
            a += math.atan(slope)
        across = -arr[:, 0] * math.sin(a) + arr[:, 1] * math.cos(a)
        return a, float(np.median(across))

    def omem_add(self, x, y, now):
        """Remember the lidar points of this scan in odom (3 cm cells)."""
        if not self.in_section() or not self.odom_fresh(now):
            return
        sel = (x > 0.0) & (x < 2.5) & (np.abs(y) < 1.3)
        if not sel.any():
            return
        X, Y, H = self.odom
        c, sn = math.cos(H), math.sin(H)
        gx = np.round((X + c * x[sel] - sn * y[sel]) / 0.03).astype(int).tolist()
        gy = np.round((Y + sn * x[sel] + c * y[sel]) / 0.03).astype(int).tolist()
        for key in set(zip(gx, gy)):
            cell = self.omem.get(key)
            if cell is None:
                self.omem[key] = [1, now]
            else:
                cell[0] += 1
                cell[1] = now
        if len(self.omem) > 4000:
            self.omem = {k: v for k, v in self.omem.items() if now - v[1] <= self.omem_keep}

    def omem_points(self, now):
        """Remembered object points in the car frame that the lidar does not see now (beside /
        behind its +-60 deg view), seen at least twice, within 1.5 m."""
        if not self.omem or not self.odom_fresh(now):
            return np.zeros(0), np.zeros(0)
        arr = np.array([k for k, v in self.omem.items()
                        if v[0] >= 2 and now - v[1] <= self.omem_keep], dtype=np.float64)
        if not len(arr):
            return np.zeros(0), np.zeros(0)
        X, Y, H = self.odom
        dx, dy = arr[:, 0] * 0.03 - X, arr[:, 1] * 0.03 - Y
        x = math.cos(H) * dx + math.sin(H) * dy
        y = -math.sin(H) * dx + math.cos(H) * dy
        sel = (np.abs(np.arctan2(y, x - self.laser[0])) > self.omem_fov) & (np.hypot(x, y) < 1.5)
        return x[sel], y[sel]

    # ============================================================== island
    def island_set(self, now):
        """At the entry (lane 2): the island tip island_dist ahead on the lane 2 centre line."""
        if not self.island_on or not self.odom_fresh(now):
            return
        X, Y, H = self.odom
        psi = math.radians(self.yaw[0]) if self.yaw_known(now) and \
            abs(self.yaw[0]) <= self.align_trust else 0.0
        a = H + psi
        cy = self.centre_y if math.isfinite(self.centre_y) and \
            abs(self.centre_y) <= self.box_shift_max else 0.0
        off = cy - self.cy_ref_x * math.tan(psi) + ((self.lane or 2) - 2) * self.lane_width
        px, py = X - off * math.sin(a), Y + off * math.cos(a)
        self.isl_ref = (px + self.island_dist * math.cos(a), py + self.island_dist * math.sin(a), a)

    def island_frame(self):
        """(tip x, tip y, direction) in odom: the lane map's line when it has one (better
        direction and centre), the entry estimate along it."""
        tx, ty, a = self.isl_ref
        lm = self.lane_map()
        if lm is None:
            return tx, ty, a
        a2, c = lm
        s = tx * math.cos(a2) + ty * math.sin(a2)
        return s * math.cos(a2) - c * math.sin(a2), s * math.sin(a2) + c * math.cos(a2), a2

    def island_ahead(self, now):
        """m from base_link to the island tip along the lane (negative: passed), None unknown."""
        if self.isl_ref is None or not self.odom_fresh(now):
            return None
        tx, ty, a = self.island_frame()
        return (tx - self.odom[0]) * math.cos(a) + (ty - self.odom[1]) * math.sin(a)

    def island_points(self, now):
        """Outline of the island (tip + both sides) in the car frame, or empty."""
        if self.isl_ref is None or not self.odom_fresh(now):
            return np.zeros(0), np.zeros(0)
        tx, ty, a = self.island_frame()
        s = np.arange(0.0, self.island_len + 1e-9, 0.03)
        hw = np.minimum(self.island_half, 0.02 + s * (self.island_half - 0.02) / self.island_taper)
        n_tip = np.arange(-0.02, 0.0201, 0.02)
        ss = np.concatenate([s, s, np.zeros(len(n_tip))])
        nn = np.concatenate([hw, -hw, n_tip])
        ox = tx + ss * math.cos(a) - nn * math.sin(a)
        oy = ty + ss * math.sin(a) + nn * math.cos(a)
        X, Y, H = self.odom
        dx, dy = ox - X, oy - Y
        return math.cos(H) * dx + math.sin(H) * dy, -math.sin(H) * dx + math.cos(H) * dy

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.odom = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)
        self.odom_at = self.now()
        if self.section_at is not None:
            xy = self.odom[:2]
            if self.section_xy is not None:
                step = math.hypot(xy[0] - self.section_xy[0], xy[1] - self.section_xy[1])
                if step < 0.2:                  # odom jump: not counted
                    self.section_dist += step
                    if self.cmd_v < 0.0:
                        self.section_rev += step
            self.section_xy = xy

    def odom_fresh(self, now):
        return self.odom_at is not None and now - self.odom_at <= self.odom_timeout

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
        """A / B sign frames are kept for sign_pre_sec even outside the section (the sign can
        stand before it); inside, the frames since sign_pre_sec before the entry decide."""
        best = None
        for det in msg.detections:
            for hyp in det.results:
                h = hyp.hypothesis
                lane = self.sign_classes.get(h.class_id)
                if lane is not None and h.score >= self.sign_conf \
                        and (best is None or h.score > best[1]):
                    best = (lane, h.score)
        now = self.now()
        if best is not None:
            self.sign_hist.append((now, best[0]))
        keep = now - max(self.sign_pre, 0.0) - 120.0
        if self.sign_hist and self.sign_hist[0][0] < keep:
            self.sign_hist = [e for e in self.sign_hist if e[0] >= keep]
        self.sign_decide_home(now)

    def sign_decide_home(self, now):
        if not self.in_section() or self.home != 2 or self.section_at is None:
            return
        since = self.section_at - self.sign_pre
        a = sum(1 for t, ln in self.sign_hist if t >= since and ln == 1)
        b = sum(1 for t, ln in self.sign_hist if t >= since and ln == 3)
        self.sign_counts = {1: a, 3: b}
        if a + b >= self.sign_decide and max(a, b) >= self.sign_ratio * (a + b):
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
        if old is None and self.section_at is None:
            self.section_at = self.now()        # entered: signs from sign_pre_sec before count
            self.sign_decide_home(self.section_at)
            self.island_set(self.section_at)
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
        self.section_at = None
        self.section_dist, self.section_xy, self.fallback = 0.0, None, None
        self.section_rev, self.hold_since = 0.0, None
        self.isl_ref = None
        self.rev_block_fwd = None
        self.lmap = []
        self.omem = {}
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
        self.omem_add(x, y, self.now())
        self.dbg_shift = shift = self.box_shift()
        self.dbg_angle = th = self.box_angle()
        u = x * math.cos(th) + y * math.sin(th)       # along the lane
        v = -x * math.sin(th) + y * math.cos(th)      # across the lane (+ left)
        for side, sign in ((LEFT, 1.0), (RIGHT, -1.0)):
            sel = ((u > self.side_x[0]) & (u < self.side_x[1])
                   & (np.abs(v - (shift + sign * self.side_offset)) < self.side_half))
            self.box[side] = int(np.count_nonzero(sel)) >= self.side_min_points
            self.box_near[side] = float(np.min(u[sel]) - self.laser[0]) if self.box[side] else None
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

    def goal_lane(self):
        """Lane to end the section in: the sign's (home 1 / 3). Lane 2 ends at the island, so
        without a sign after sign_wait_dist m: sign_default_lane, or the side lane the car is
        in, or from lane 2 the free side (both free: lane 1)."""
        if self.home != 2 or self.section_dist < self.sign_wait:
            return self.home
        if self.fallback is None:
            ahead = self.island_ahead(self.now())
            if ahead is not None and ahead > self.sign_fallback_ahead and self.lane in (2, None):
                return self.home        # keep lane 2 (sign straight ahead, both roads one lane)
            if self.sign_default in (1, 3):
                g = self.sign_default
            elif self.state == MissionState.RUNNING and self.phase == PATH and \
                    self.request is not None and self.request[1] in (1, 3):
                g = self.request[1]         # on the way to an outer lane: that one
            elif self.lane in (1, 3):
                g = self.lane
            else:
                side = self.side_blocked()
                g = 3 if 1 in side and 3 not in side else 1
            self.fallback = g
            self.get_logger().warn(f'no A / B sign after {self.section_dist:.1f} m: goal lane {g} '
                                   f'(lane 2 ends at the island)')
        return self.fallback

    def dodge_target(self, side):
        """Lane to dodge to if my lane is blocked: the free lane nearest to home (lane 2,
        both free, home 2: away from the object, y ~ 0 -> right); None if none."""
        lane, home = self.lane, self.goal_lane()
        free = [ln for ln in (1, 2, 3) if ln != lane and ln not in side]
        if lane == home and home in (1, 3) and 2 in side:
            # in the goal lane: lane 2 blocked only by an object about to be beside the car ->
            # still the dodge (the curve check decides: go / back off / wait). Across to the
            # far side the car would have to come back two lanes before the island
            bn = self.box_near[LEFT if lane == 3 else RIGHT]
            if bn is not None and bn < self.side_pass_dist:
                free = [2] + free
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
        lane, home = self.lane, self.goal_lane()
        if lane is None:
            return None
        ahead = self.island_ahead(self.now())
        if ahead is not None and ahead < -self.island_past:
            return None         # beside the island (A / B road): no lane change any more. Up
            #                     to island_past after the (conservative) tip the island
            #                     outline still blocks a crossing curve -> backoff
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
        changing = self.state == MissionState.RUNNING and self.phase in CHANGING
        if changing and self.request is not None and self.request[0] in (LEFT, RIGHT):
            if self.request[0] == LEFT:     # the front corner sweeps to the side it moves to
                y1 += self.guard_turn_extra
            else:
                y0 -= self.guard_turn_extra
        self.guard_y = (y0, y1)
        sel = (x > self.front_x - 0.02) & (y > y0) & (y < y1)
        self.guard_near = float(np.min(x[sel]) - lx) if sel.any() else None
        close = sel & (x - lx < (self.guard_dist_path if changing else self.guard_dist))
        self.guard_count = self.guard_count + 1 \
            if int(np.count_nonzero(close)) >= self.guard_min_points else 0
        if self.guard_count < self.guard_scans:
            return
        why = f'guard: object {self.guard_near:.2f} m from the lidar'
        now = self.now()
        if changing:
            self.guard_count = 0
            self.resume_after = self.phase == PATH
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
        self.path = None
        self.judge = None
        self.after_backoff = None
        self.commit = self.backoff_side = None
        self.pad_backoffs = self.resumes = 0
        self.resume_after = False
        self.backoff_plan = None
        self.guard_count = 0

    def start_pre_stop(self, now, why):
        self.phase_since = now
        self.centre_samples = []
        self.set_phase(PRE_STOP, f'{why}, hold {self.pre_stop:.1f} s')

    def start_backoff(self, now, after, why, need=None, commit=None, side=None, plan=None):
        self.after_backoff = after
        if plan is None:
            self.backoff_isl = False
        self.commit, self.backoff_side = commit, side
        self.backoff_plan = plan
        self.backoff_from = self.odom[:2] if self.odom_fresh(now) else None
        self.backoff_room = self.rear_free(now) - self.rear_clear
        self.backoff_need = self.backoff_clear if need is None else need
        self.phase_since = now
        self.set_phase(BACKOFF, why)

    def try_start(self, now):
        """Stopped long enough: decide again, back off when too close for that change
        (one / two lanes), else change."""
        committed, self.commit = self.commit, None
        if committed is not None and self.lane is not None:
            goal = self.goal_lane()
            mine_blocked = self.front_near is not None and self.front_near < self.dodge
            # ...unless it dodges an object right ahead in my lane (2026-11-18 12:14: lane 3
            # blocked, goal 3 -> 'left to 2' dropped 14 x, stuck in front of the object)
            if goal in (1, 3) and abs(committed[1] - goal) > abs(self.lane - goal) \
                    and not mine_blocked:
                # the move was fixed before the backoff; the goal (sign) changed since and it
                # would now lead away from it (2026-11-17 10:30: lane 1 reached, sign A came,
                # the old 'right to 3' still ran -> B road): decide again
                self.get_logger().info(f'{committed[0]} to {committed[1]} dropped: goal lane '
                                       f'{goal} now (lane {self.lane})')
                committed = None
        move = committed or self.decide()
        bl = sorted(self.blocked_lanes())
        wait = ''
        if move is None and self.lane in (1, 3) and 2 in self.side_blocked() \
                and self.front_near is not None and self.front_near < self.dodge2:
            wait = f' (lane 2 blocked: straight to {self.dodge:.2f})'
        self.decision = (f'lane {self.lane} home {self.home} blocked {bl or "-"} -> '
                         + (f'{move[0]} to {move[1]}' if move else 'stay') + wait
                         + (' (fixed before the backoff)' if committed else ''))
        if move is None:
            self.request = None
            self.finish(f'{self.decision}: lane follow')
            return
        side, target = move
        two = abs(target - self.lane) == 2
        if committed or (self.change_mode == 'path' and self.odom_fresh(now)):
            pass        # backed off for this move / path mode: the curve itself is checked below
        elif two and 2 in self.side_blocked():
            # across a blocked middle lane: my lane AND the middle lane object at least
            # backoff_clear_mid away. That is beyond the dodge distance, so the move is fixed
            # (deciding again after the backoff would say 'stay' and drive back: loop)
            need = self.backoff_clear_mid
            near = [d for d in (self.guard_near, self.box_near[side]) if d is not None]
            if near and min(near) < need:
                self.start_backoff(now, PRE_STOP, f'{side} to {target} across lane 2: object '
                                                  f'{min(near):.2f} m < {need:.2f} m, back off '
                                                  f'to {need:.2f} m', need, commit=move, side=side)
                return
        else:
            need = self.backoff_clear2 if two else self.backoff_clear
            if self.guard_near is not None and self.guard_near < need:
                self.start_backoff(now, PRE_STOP, f'{side} to {target}: object '
                                                  f'{self.guard_near:.2f} m < {need:.2f} m, '
                                                  f'back off to {need:.2f} m', need)
                return
        self.request = move
        self.orig_lane = self.lane
        cy = float(np.mean(self.centre_samples)) if self.centre_samples else 0.0
        self.path = None
        if self.change_mode == 'path' and self.odom_fresh(now):
            self.change_path(now, move, cy)
            return
        if self.change_mode == 'path':
            self.get_logger().warn('no fresh odom: timed lane change')
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

    def change_path(self, now, move, cy):
        """Pick the curve: (1) the decided move or a lane no farther from the goal lane, clear;
        (2) none: back off straight (bounded: path_backoff_tries per change, backoff_budget per
        section, never with an object remembered behind); (3) then any lane with a clear
        curve; (4) none: my lane free ahead -> lane follow (pass the object beside first),
        else hold (stopped) and look again, at most hold_max_sec, then lane follow (the guard
        stops the car before an object)."""
        side, target = move
        opts = self.path_options(move, cy, now)
        pick = next((o for o in opts if o[1]['clear']), None)
        # the island blocks the first choice: backing off is better than driving into it
        isl = bool(opts) and opts[0][1].get('isl', False)
        tries = self.path_backoff_tries + (1 if isl else 0)
        budget = self.backoff_budget + (self.island_budget if isl else 0.0)
        if pick is None and opts and self.pad_backoffs < tries \
                and self.section_rev < budget and self.may_reverse(now):
            self.pad_backoffs += 1
            self.request = None
            self.path = opts[0][1]
            self.backoff_isl = isl
            self.plan_check = (now, False)
            self.start_backoff(now, PRE_STOP, f'{side} to {target}: {self.path_text(opts[0][1])}, '
                                              f'back off until a curve clears (<= '
                                              f'{self.path_backoff_max:.1f} m, reversed '
                                              f'{self.section_rev:.1f} / {self.backoff_budget:.1f} m)',
                               commit=move, plan=(move, cy))
            return
        if pick is None:
            pick = next((o for o in self.path_options(move, cy, now, any_lane=True)
                         if o[1]['clear']), None)
        if pick is None:
            free = self.front_near is None or self.front_near >= self.dodge
            if free or now - (self.hold_since or now) >= self.hold_max:
                why = 'my lane free ahead' if free else f'held {self.hold_max:.0f} s'
                self.get_logger().info(f'{side} to {target}: no clear curve, {why}: lane follow')
                self.hold_since = None
                self.request, self.path = None, None
                self.finish(f'{self.decision}: no clear curve, lane follow')
                return
            self.hold_since = self.hold_since or now
            self.commit = move              # look again with the same move
            self.request = None
            self.start_pre_stop(now, f'{side} to {target}: no clear curve, hold')
            return
        self.hold_since = None
        (side, target), self.path = pick
        if (side, target) != move:
            self.decision += f' -> {side} to {target} (curve of the first choice not clear)'
        self.request = (side, target)
        txt = self.path_text(self.path)
        self.pad_backoffs = self.resumes = 0
        self.get_logger().info(f'{side} to lane {target}: {txt}')
        self.path['goal'] = self.goal_lane()    # the goal this curve was chosen for
        self.set_phase(PATH, f'{self.decision}, {txt}')

    def path_options(self, move, cy, now, any_lane=False):
        """[((side, target), path)]: the decided move, then the other lanes (nearest to home
        first). Empty without fresh odom."""
        lane, goal = self.lane, self.goal_lane()
        # other lanes only if they end no farther from the goal lane than the decided move
        # (a curve to the far side would have to come back two lanes before the island)
        # any_lane (nothing else clear): still never a lane farther from the goal lane (A / B
        # road) than the car is now, unless my lane is blocked right ahead (2026-11-17 09:1x:
        # goal 3, the lane-3 curve blocked -> it went to lane 1 = the A road)
        mine_blocked = self.front_near is not None and self.front_near < self.dodge
        away_ok = goal not in (1, 3) or mine_blocked
        targets = [move[1]] + sorted((ln for ln in (1, 2, 3) if ln not in (lane, move[1])
                                      and (abs(ln - goal) <= abs(move[1] - goal) or
                                           (any_lane and (away_ok or
                                                          abs(ln - goal) <= abs(lane - goal))))),
                                     key=lambda ln: abs(ln - goal))
        out = []
        for tg in targets:
            sd = LEFT if tg < lane else RIGHT
            pa = self.make_path(sd, abs(tg - lane), cy, now)
            if pa is not None:
                out.append(((sd, tg), pa))
        return out

    def rear_blocked(self, now):
        """A remembered object right behind the car (within rear_clear of the rear bumper,
        car width + 4 cm): backing off would hit it."""
        x, y = self.omem_points(now)
        if not len(x):
            return False
        sel = (x < -self.car_rear + 0.02) & (x > -self.car_rear - self.rear_clear) \
            & (np.abs(y) < self.car_half + 0.04)
        return int(np.count_nonzero(sel)) >= 2

    def rear_free(self, now):
        """m free behind the rear bumper (remembered objects in the car's width + 5 cm), inf
        when nothing is remembered there (the lidar does not see behind)."""
        x, y = self.omem_points(now)
        if not len(x):
            return float('inf')
        sel = (x < -self.car_rear + 0.02) & (np.abs(y) < self.car_half + 0.05)
        if np.count_nonzero(sel) < 2:
            return float('inf')
        return max(0.0, float(np.sort(-x[sel])[1]) - self.car_rear)

    def forward_dist(self):
        return self.section_dist - 2.0 * self.section_rev     # section_dist counts reversing too

    def may_reverse(self, now):
        """Backing off allowed: room behind, and not just stopped by an object behind."""
        if self.rev_block_fwd is not None:
            if self.forward_dist() - self.rev_block_fwd < self.rev_rearm:
                return False
            self.rev_block_fwd = None
        return self.rear_free(now) - self.rear_clear >= self.rear_free_min

    def blocked_beside(self, pa):
        """True when an object (seen or remembered) is beside the car on the side it moves
        to (between the rear and the front bumper, up to one lane out) AND my lane ahead is
        free for a while: driving on passes it. With my lane blocked ahead, driving on would
        run into that object: False (back off instead)."""
        if self.front_near is not None and self.front_near < 0.6:
            return False
        if pa.get('obj') is None or self.odom is None:
            return False
        X, Y, H = self.odom
        dx, dy = pa['obj'][:, 0] - X, pa['obj'][:, 1] - Y
        u = math.cos(H) * dx + math.sin(H) * dy
        v = -math.sin(H) * dx + math.cos(H) * dy
        sg = 1.0 if pa['D'] >= 0.0 else -1.0
        sel = (u > -self.car_rear - 0.05) & (u < self.front_x + 0.05) \
            & (sg * v > self.car_half) & (sg * v < self.car_half + self.lane_width)
        return int(np.count_nonzero(sel)) >= 2

    def path_text(self, pa):
        txt = (f'path L {pa["L"]:.2f} m (lead {pa["lead"]:.1f}), shift {pa["D"]:+.2f} m, heading '
               f'{math.degrees(pa["psi"]):+.0f} deg ({pa["src"]}), curvature {pa["kmax"]:.1f} 1/m')
        if pa['obj'] is not None:
            txt += f', gap to object {pa["gap"] * 100:.0f} cm (padding {self.obj_padding * 100:.0f} cm)'
            if pa.get('isl'):
                txt += ' [island]'
        return txt

    def resume_path(self, now, force=False):
        """After a guard backoff during a path: a new curve from here to the SAME target line.
        The lane frame of the first curve is kept in odom, so this needs no camera (the car
        stands at an angle there and the lane heading is not trusted)."""
        old = self.path
        if old is None or self.request is None or self.request[0] not in (LEFT, RIGHT) \
                or self.resumes >= self.path_resume_max * (2 if force else 1):
            return False
        pa = self.make_path(self.request[0], 1, 0.0, now, resume=old['frame'])
        if pa is None:
            return False
        self.resumes += 1
        self.path = pa
        txt = 'new ' + self.path_text(pa)
        self.get_logger().info(f'resume {self.request[0]} to lane {self.request[1]}: {txt}')
        self.set_phase(PATH, f'after the guard ({self.resumes}/{self.path_resume_max}): {txt}')
        return True

    def make_path(self, side, lanes, cy, now, resume=None):
        """S curve (arc, straight, arc) in the lane frame (x along the lane from the car, y
        left), stored in odom. Start: the car's pose; end: on the target lane centre line,
        parallel. Lane frame: the lane map when it has enough samples and agrees with the lane
        number, else the camera (lane heading + lane centre y averaged over pre_stop).
        Tried from path_k_soft to path_kmax curvature (sharper = shorter); the gentlest whose
        swept car body keeps obj_padding from every object in my lane, the lanes crossed and
        the target lane, and stays within path_over_max of the target line. None does ->
        the best one, 'clear' False."""
        if not self.odom_fresh(now):
            return None
        X0, Y0, H0 = self.odom
        src = 'cam'
        if resume is None:
            psi = math.radians(self.yaw[0]) if self.yaw_known(now) and \
                abs(self.yaw[0]) <= self.align_trust else 0.0   # lane direction, car frame
            sign = 1.0 if side == LEFT else -1.0
            cy -= self.cy_ref_x * math.tan(psi)     # measured ahead of the car -> at the car
            lm = self.lane_map()
            if lm is not None and self.lane is not None:
                a, c = lm
                cy_m = -(self.lane - 2) * self.lane_width \
                    - (-X0 * math.sin(a) + Y0 * math.cos(a) - c)
                if abs(cy_m) <= self.lane_width / 2.0 + 0.05:    # agrees with the lane number
                    psi = max(-1.2, min(1.2, math.atan2(math.sin(a - H0), math.cos(a - H0))))
                    cy, src = cy_m, 'map'
            D = cy + sign * lanes * self.lane_width
            if self.lane is not None and self.lane - int(sign) * lanes in (1, 3):
                D -= sign * self.path_edge_bias         # outer lane: stay off its edge line
            frame = dict(a=H0 + psi, org=(X0, Y0), D=D)     # lane frame + target line in odom
        else:                                   # the same lane frame and target line, from here
            frame = resume
            psi = math.atan2(math.sin(frame['a'] - H0), math.cos(frame['a'] - H0))
            psi = max(-1.2, min(1.2, psi))
            gx, gy = X0 - frame['org'][0], Y0 - frame['org'][1]
            D = frame['D'] - (-gx * math.sin(frame['a']) + gy * math.cos(frame['a']))
            cy = 0.0
            src = 'odom'
        tsign = 1.0 if D >= 0.0 else -1.0
        # object points (lane frame) in my lane, the lanes crossed and the target lane: this
        # scan + the remembered ones beside / behind the car; each also obj_depth further
        # along (the lidar sees the near face only)
        px, py = self.dbg_pts
        mx, my = self.omem_points(now)
        px, py = np.concatenate([px, mx]), np.concatenate([py, my])
        ox = px * math.cos(psi) + py * math.sin(psi)
        oy = -px * math.sin(psi) + py * math.cos(psi)
        half = self.lane_width / 2.0 - 0.03     # not the things beyond the outer lane lines
        sel = (ox > -0.5) & (ox < 2.5) & (oy > min(cy, D) - half) & (oy < max(cy, D) + half)
        obj = None
        if np.count_nonzero(sel) >= 2:
            obj = np.stack([ox[sel], oy[sel]], axis=1)
            # the hidden back: per 4 cm across the lane, the near face + obj_depth, only where
            # the points do not show the depth already (a side face seen along the lane would
            # otherwise be made obj_depth longer than the object)
            bins = np.round(obj[:, 1] / 0.04).astype(int)
            back = []
            for b_ in np.unique(bins):
                xs = obj[bins == b_, 0]
                if np.ptp(xs) < self.obj_depth:
                    back.append((float(xs.min()) + self.obj_depth, b_ * 0.04))
            if back:
                obj = np.concatenate([obj, np.array(back)])
        # the island outline (painted: the lidar does not see it), in every lane
        isl = None
        ix, iy = self.island_points(now)
        if len(ix):
            iu = ix * math.cos(psi) + iy * math.sin(psi)
            iv = -ix * math.sin(psi) + iy * math.cos(psi)
            s_ = (iu > -0.5) & (iu < 3.0)
            if s_.any():
                isl = np.stack([iu[s_], iv[s_]], axis=1)
                obj = isl if obj is None else np.concatenate([obj, isl])
        # candidates: straight lead-in (along the car's heading) x curvature x steepest heading.
        # Best: clear of the objects and the lane edge, then the widest gap up to path_gap_good,
        # then the shortest lead-in, the gentlest curvature, the flatter heading
        h0 = -psi
        ks = sorted({round(float(k), 2) for k in
                     np.linspace(self.path_k_soft, max(self.path_kmax, self.path_k_soft), 3)})
        ths = sorted({self.path_th_max, max(self.path_th_max, self.path_th_steep)})
        leads = np.arange(0.0, self.path_lead_max + 1e-9, 0.1) if obj is not None else [0.0]
        ne = int(round(self.path_end_clear / 0.01))
        # outer yellow lines in the lane frame (lane 1 left edge, lane 3 right edge): every
        # body corner stays edge_margin inside them (the rear swings out in a curve)
        edges = None
        if resume is None and self.lane is not None:
            edges = (cy + (self.lane - 1) * self.lane_width + self.lane_width / 2.0,
                     cy - (3 - self.lane) * self.lane_width - self.lane_width / 2.0)
        cax = np.array([self.front_x, self.front_x, -self.car_rear, -self.car_rear])
        cay = np.array([self.car_half, -self.car_half, self.car_half, -self.car_half])
        best = None
        for d in leads:
            nl = int(round(d / 0.01))
            for k in ks:
                for thm in ths:
                    x, y, th = self.s_curve(h0, D - d * math.sin(h0), k, thm)
                    if nl:
                        t = np.arange(nl) * 0.01
                        x = np.concatenate([t * math.cos(h0), d * math.cos(h0) + x])
                        y = np.concatenate([t * math.sin(h0), d * math.sin(h0) + y])
                        th = np.concatenate([np.full(nl, h0), th])
                    # past the target line, or swinging out on the other side: the lane edges
                    over = float(max(np.max(tsign * (y - D)), np.max(-tsign * y)))
                    if obj is None:
                        gap, pt = float('inf'), None
                    else:
                        e = 0.01 * np.arange(1, ne + 1)
                        gap, pt = self.sweep_gap(np.concatenate([x, x[-1] + e]),
                                                 np.concatenate([y, np.full(ne, y[-1])]),
                                                 np.concatenate([th, np.zeros(ne)]), obj)
                    ok_over = over <= self.path_over_max
                    if edges is not None:
                        cs, sn = np.cos(th[::3]), np.sin(th[::3])
                        ys = y[::3, None] + sn[:, None] * cax + cs[:, None] * cay
                        em = min(edges[0] - float(ys.max()), float(ys.min()) - edges[1])
                        ok_over = ok_over and em >= self.edge_margin
                    key = (ok_over and gap >= self.obj_padding, ok_over,
                           int(min(gap, self.path_gap_good) / 0.02), -d, -k, -thm)
                    if best is None or key > best[0]:
                        best = (key, x, y, th, k, gap, pt, over, float(d))
                if obj is None:
                    break
            if best[0][0] and best[0][2] >= int(self.path_gap_good / 0.02):
                break                                   # good enough: no longer lead-in
        _, x, y, th, kmax, gap, pt, over, lead = best
        L = float(x[-1])
        xe = np.arange(L + 0.01, L + self.path_ext + 1e-9, 0.01)
        n_curve = len(x)
        x, y = np.concatenate([x, xe]), np.concatenate([y, np.full(len(xe), y[-1])])
        a = H0 + psi                                          # lane direction in odom

        def to_odom(lx, ly):
            return (X0 + lx * math.cos(a) - ly * math.sin(a),
                    Y0 + lx * math.sin(a) + ly * math.cos(a))

        pts = np.stack(to_odom(x, y), axis=1)
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
        s_end = float(s[n_curve - 1])
        by_isl = pt is not None and isl is not None and gap < self.obj_padding and \
            float(np.min(np.hypot(isl[:, 0] - pt[0], isl[:, 1] - pt[1]))) < 1e-6
        return dict(pts=pts, s=s, L=L, s_end=s_end, idx=0, D=D, psi=psi, kmax=kmax, src=src,
                    lead=lead, isl=by_isl,
                    gap=gap, clear=gap >= self.obj_padding and over <= self.path_over_max,
                    over=over, frame=frame, ok=False,
                    obj=None if obj is None else np.stack(to_odom(obj[:, 0], obj[:, 1]), axis=1),
                    near=None if pt is None else to_odom(*pt),
                    until=now + self.path_max_scale * max(s_end, 0.3) / max(self.path_speed, 0.05),
                    ahead=None)

    @staticmethod
    def s_curve(h0, D, k, th_max, ds=0.01):
        """Points (x, y, heading) every ds from (0, 0, h0) to the line y = D, heading 0:
        an arc of curvature k to the heading +-th, a straight, an arc back. th: the one that
        ends on the line without a straight, at most th_max (then with a straight)."""
        sg = 1.0 if D >= 0.0 else -1.0

        def build(th, m):
            n1 = int(round(abs(sg * th - h0) / (k * ds)))
            n2 = int(round(m / ds))
            n3 = int(round(th / (k * ds)))
            hs = np.concatenate([np.linspace(h0, sg * th, n1 + 1)[1:], np.full(n2, sg * th),
                                 np.linspace(sg * th, 0.0, n3 + 1)[1:]])
            if len(hs) == 0:
                hs = np.zeros(1)
            mid = (np.concatenate([[h0], hs[:-1]]) + hs) / 2.0
            return (np.concatenate([[0.0], np.cumsum(np.cos(mid) * ds)]),
                    np.concatenate([[0.0], np.cumsum(np.sin(mid) * ds)]),
                    np.concatenate([[h0], hs]))

        lo = max(0.0, sg * h0)          # already turned toward the target by this much
        hi = max(th_max, lo)
        best = build(lo, 0.0)
        if sg * (best[1][-1] - D) >= 0.0:
            return best                 # even turning straight back reaches / passes the line
        best = build(hi, 0.0)
        if sg * (best[1][-1] - D) < 0.0:            # the steepest S is short: add the straight
            return build(hi, sg * (D - best[1][-1]) / max(math.sin(hi), 0.1))
        # no straight: lateral = (cos h0 + 1 - 2 cos th) / k -> th, then refined on the points
        th = math.acos(max(-1.0, min(1.0, (math.cos(h0) + 1.0 - k * abs(D)) / 2.0)))
        lo, hi = max(lo, th - 0.06), min(hi, th + 0.06)
        for _ in range(8):
            th = (lo + hi) / 2.0
            best = build(th, 0.0)
            if sg * (best[1][-1] - D) < 0.0:
                lo = th
            else:
                hi = th
        return best

    def sweep_gap(self, x, y, th, obj):
        """Smallest distance from the object points to the car body (front_x / car_rear /
        car_half_width) moving along the curve, -1 inside: (gap, nearest object point)."""
        x, y, th = x[::4], y[::4], th[::4]
        cs, sn = np.cos(th), np.sin(th)
        dx = obj[None, :, 0] - x[:, None]               # object in each car pose frame
        dy = obj[None, :, 1] - y[:, None]
        u = cs[:, None] * dx + sn[:, None] * dy
        v = -sn[:, None] * dx + cs[:, None] * dy
        ex = np.maximum(np.maximum(u - self.front_x, -self.car_rear - u), 0.0)
        ey = np.maximum(np.abs(v) - self.car_half, 0.0)
        d = np.where((ex == 0.0) & (ey == 0.0), -1.0, np.hypot(ex, ey))
        k = np.unravel_index(int(np.argmin(d)), d.shape)
        return float(d[k]), (float(obj[k[1], 0]), float(obj[k[1], 1]))

    def step_path(self, now):
        """Pure pursuit on the stored path; True when done (or given up)."""
        pa = self.path
        if not self.odom_fresh(now):
            self.get_logger().warn('odom lost during the path: stop')
            pa['quit'] = True
            return True
        if now > pa['until']:
            self.get_logger().warn('path took too long: stop')
            pa['quit'] = True
            return True
        X, Y, H = self.odom
        pts = pa['pts']
        i0 = pa['idx']
        win = pts[i0:min(len(pts), i0 + 60)]
        pa['idx'] = i = i0 + int(np.argmin(np.hypot(win[:, 0] - X, win[:, 1] - Y)))
        if pa['s'][i] >= pa['s_end'] - self.path_end_tol:
            pa['ok'] = True                     # the whole curve driven: the car is on target
            return True
        j = int(np.searchsorted(pa['s'], pa['s'][i] + self.path_ld))
        j = min(j, len(pts) - 1)
        dx, dy = pts[j, 0] - X, pts[j, 1] - Y
        lx = math.cos(H) * dx + math.sin(H) * dy          # look-ahead point in the car frame
        ly = -math.sin(H) * dx + math.cos(H) * dy
        pa['ahead'] = (lx, ly)
        k = 2.0 * ly / max(lx * lx + ly * ly, 1e-4)
        w = self.path_w_gain * self.path_speed * k
        self.publish(self.path_speed, max(-self.path_w, min(self.path_w, w)))
        return False

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
        if j['origin'] == 'guard' and self.resume_path(now, force=True):
            return      # the car stands at an angle: finish the change on odom, not a fresh
            #             plan that would take the car's heading for the lane's
        if j['origin'] == 'guard' and not lost:
            self.lane = self.orig_lane      # not judged: the lane before the aborted change
            self.get_logger().info(f'not judged after the guard: lane {self.lane} kept')
            self.start_pre_stop(now, f'lane {self.lane} kept')
            return
        why = 'my lane only' if lost else 'no lane judged (not parallel?)'
        if j['origin'] == 'change' and self.path is not None and self.path['ok'] \
                and self.request is not None and self.request[1] is not None:
            # the curve was driven to its end on odom: the car is in the target lane
            self.lane, self.lane_unknown = self.request[1], False
            self.zone_grace_from = now
            self.get_logger().info(f'{why}: path driven to the end -> lane {self.lane}')
            self.try_start(now)
            return
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
            if self.backoff_side is not None:       # the lane to cross as well
                bn = self.box_near[self.backoff_side]
                clear = clear and (bn is None or bn >= self.backoff_need)
            far = False
            if self.backoff_from is not None and self.odom_fresh(now):
                gone = math.hypot(self.odom[0] - self.backoff_from[0],
                                  self.odom[1] - self.backoff_from[1])
                far = gone >= min((self.island_backoff_max if self.backoff_isl
                                   else self.path_backoff_max) if self.backoff_plan else
                                  self.backoff_max_dist, self.backoff_room)
            if self.backoff_plan is not None:       # until a curve is clear (checked at ~7 Hz)
                if now - self.plan_check[0] >= 0.15:
                    opts = self.path_options(*self.backoff_plan, now)
                    self.plan_check = (now, any(o[1]['clear'] for o in opts))
                clear = self.plan_check[1]
            behind = self.rear_blocked(now)
            if behind:
                self.get_logger().warn('object behind the car (remembered): backoff stopped')
                self.rev_block_fwd = self.forward_dist()
            if (age >= self.backoff_min and clear) or age >= self.backoff_max or far or behind:
                self.publish(0.0, 0.0)
                self.guard_count = 0
                resume, self.resume_after = self.resume_after, False
                if self.after_backoff == ALIGN and resume and self.resume_path(now):
                    pass
                elif self.after_backoff == ALIGN:
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
        elif ph == PATH:
            pa = self.path
            goal = self.goal_lane()
            if self.request is not None and goal in (1, 3) and self.orig_lane is not None \
                    and goal != pa.get('goal') and self.home == goal \
                    and abs(self.request[1] - goal) > abs(self.orig_lane - goal) \
                    and pa['s'][pa['idx']] < self.path_abort_frac * pa['s_end']:
                # the sign came just after this curve started and it leads away from the sign's
                # road (2026-11-17 10:54: 'right to 3' started 0.1 s before sign A -> B road):
                # stop while still near the start lane and decide again from there
                self.get_logger().info(f'{self.request[0]} to {self.request[1]} aborted at '
                                       f'{pa["s"][pa["idx"]]:.2f} / {pa["s_end"]:.2f} m: goal '
                                       f'lane {goal} now')
                self.publish(0.0, 0.0)
                self.request, self.path = None, None
                self.lane = self.orig_lane
                self.start_pre_stop(now, f'curve aborted, goal lane {goal}')
                return
            if self.step_path(now) and self.path['ok']:
                # the whole curve driven on odom: the car is in the target lane. Decide at
                # once: nothing to do -> lane follow without stopping
                old, self.lane = self.lane, self.request[1]
                self.lane_unknown = False
                self.zone_grace_from = now
                self.pad_backoffs = 0
                self.get_logger().info(f'path done: lane {old} -> {self.lane}')
                if self.decide() is None:
                    self.request = None
                    self.finish(f'lane {old} -> {self.lane} path done: lane follow')
                else:
                    self.publish(0.0, 0.0)
                    self.start_pre_stop(now, f'lane {old} -> {self.lane} path done, next change')
            elif not self.path['ok'] and self.phase == PATH and self.path.get('quit'):
                self.phase_since = now
                self.publish(0.0, 0.0)
                self.set_phase(POST_STOP, f'lane {self.orig_lane} -> {self.request[1]} path '
                                          f'done, hold {self.post_stop:.1f} s')
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
        if self.rear_blocked(now):
            self.publish(0.0, 0.0)
            self.start_judge(now, 'guard', f'align {age:.1f} s stopped: object behind')
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
        if self.speed_limit > 0.0 and (self.home == 2 or self.lane != self.home):
            ahead = self.island_ahead(now)     # no sign yet / not in its lane: slow near the island
            if ahead is not None and -0.3 < ahead < self.island_slow_dist:
                self.speed_limit = min(self.speed_limit, self.island_slow_speed)

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
        if self.path is not None and self.odom is not None:   # planned path, car frame now
            X, Y, H = self.odom
            d = self.path['pts'] - (X, Y)
            u = math.cos(H) * d[:, 0] + math.sin(H) * d[:, 1]
            v = -math.sin(H) * d[:, 0] + math.cos(H) * d[:, 1]
            poly = np.array([px(a, b) for a, b in zip(u[::3], v[::3])], np.int32)
            cv2.polylines(view, [poly], False, (255, 255, 0), 2)
            if self.path['ahead'] is not None and self.phase == PATH:
                cv2.circle(view, px(*self.path['ahead']), 4, (0, 255, 255), -1)
            pa = self.path

            def car(q):
                ex, ey = q[0] - X, q[1] - Y
                return px(math.cos(H) * ex + math.sin(H) * ey, -math.sin(H) * ex + math.cos(H) * ey)

            if pa['obj'] is not None:                   # object points grown by the padding
                r = max(1, int(self.obj_padding * scale))
                for q in pa['obj'][::2]:
                    cv2.circle(view, car(q), r, (0, 90, 200), 1)
                cv2.circle(view, car(pa['near']), 4, (0, 0, 255), -1)
            gap = '' if pa['obj'] is None else f' gap {pa["gap"] * 100:.0f}cm'
            put(view, f'path {self.change_mode} k {pa["kmax"]:.2f}{gap}'
                      + ('' if pa['clear'] else ' < padding'), (8, 34),
                (255, 255, 0) if pa['clear'] else (0, 0, 255), 0.4)
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
