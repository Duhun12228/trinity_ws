"""
Cone mission node: drive through a corridor of cones with the 2D lidar.

The camera (yolopx detections) is only used to enter and leave; the path, the steering and
the safety come from /scan. Lane marks and the drivable area are not used for driving, so
the corridor may leave the lane.

Frames: everything is in base_link (x forward, y left). The lidar and camera poses are
parameters (measured from /tf_static: laser x 0.103, camera x 0.230 y 0.020).

Trigger (always on)
  IDLE -> REQUEST  a cone box (score >= trigger_conf) in trigger_frames consecutive frames
                   AND the lidar sees something within enter_dist in the box direction
                   (bearing seen from the camera position, so camera and lidar look at the
                   same object)
Running (one step per scan)
  corridor  lidar points in 0 < x < roi_x_max, |y| < roi_y_half -> clusters (neighbouring
            beams closer than cluster_gap) = cones, then
            1. labels from opposite pairs: two cones lane_width +- pair_width_tol apart whose
               connecting line is within pair_max_angle_deg of perpendicular to the travel
               direction (previous path slope there), each cone in one pair only (most
               perpendicular first); the cone left of the travel direction is 'left'
            2. the labels spread along each row: an unlabelled cone within row_link_max of
               a labelled one takes its label (nearest first)
            3. one curve per row (2nd order with >= 3 cones if fit_order 2, a line with 2,
               the previous path shape with 1) -> centre = middle of the two rows, or one
               row shifted lane_width / 2 inwards -> EMA with path_alpha
              no pair  : the row of the cone nearest the car (cones linked by gaps <=
                         row_link_max), on the side of that cone's y sign
              no cone  : keep the last command for lost_hold_sec, then stop
            The lookahead target (lookahead past the nearest observed cone) and the
            feedforward curvature are read only inside the observed x range (no
            extrapolation into the lidar blind area near the car).
  drive     omega = steer_gain x pure pursuit (path point lookahead ahead, arc length)
                  + ff_gain x speed x path curvature at x = ff_dist (turns into a bend
                    before the lookahead point reaches it), limited by max_steer;
                  the path followed is the centre line shifted by path_offset (+ left /
                  - right; the row labelling keeps the unshifted centre), then pushed away
                  from cones closer than cone_radius + car_half_width (<= cone_push_max)
  backoff   any lidar point in the danger box in front of the car (x < front_x +
            danger_ahead, |y| < safe_half_width): backoff_speed for backoff_sec with the
            steering angle opposite to the last forward command (x backoff_steer_ratio)
  escape    stuck_count backoffs within stuck_window: an S at max steering turning the nose to
            escape_first: reverse for escape_sec (wheels to the other side), then forward for
            escape_fwd_sec (wheels to that side; stopped early by an object in the danger
            box). The next escape goes to the other side, alternating; the
            escape counts as the first backoff of the next window; no backoff for
            stuck_reset_sec starts again from escape_first
  exit      no cluster in the ROI + no cone box on the camera, and either
              the lane state in lane_ok_states for exit_hold_sec, or
              any lane state for exit_no_cone_sec
            -> DONE (normal); the base cooldown blocks re-entry
Debug: /cone/debug_markers (MarkerArray in base_link) for RViz: cones (orange), opposite
pairs (yellow) and their midpoints (white), left row (green), right row (blue), path (cyan),
lookahead target (magenta), danger box (red), ROI (grey).
"""
from collections import deque
import math
import signal

from geometry_msgs.msg import Point, Twist
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from trinity_msgs.msg import MissionState
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import Marker, MarkerArray

from trinity_missions.mission_base import MissionNode

TRACK, LOST, BACKOFF = 'track', 'lost', 'backoff'
ESCAPE, ESCAPE_FWD = 'escape', 'escape_fwd'   # reverse / forward half of the escape S


class ConeNode(MissionNode):

    def __init__(self):
        super().__init__('cone_node', 'cone')
        d = self.declare_parameter
        d('scan_topic', '/scan')
        d('detections_topic', '/yolopx/detections')
        d('lane_state_topic', '/lane_follow/state')
        d('cmd_topic', '/cone_cmd_vel')
        d('debug_topic', '/cone/debug_markers')
        d('frame_id', 'base_link')
        # sensor poses in base_link
        d('laser_x', 0.103)
        d('laser_y', 0.0)
        d('laser_yaw', 0.0)
        d('camera_x', 0.230)
        d('camera_y', 0.020)
        d('camera_fx', 489.424)
        d('camera_cx', 319.539)
        # entry
        d('trigger_class', 'cone')
        d('trigger_conf', 0.5)
        d('trigger_frames', 3)
        d('enter_dist', 0.5)
        d('bearing_margin_deg', 2.0)
        d('request_timeout', 1.0)
        # corridor
        d('roi_x_max', 1.2)
        d('roi_y_half', 0.8)
        d('cluster_gap', 0.08)
        d('cluster_min_points', 2)
        d('cluster_max_size', 0.25)
        d('lane_width', 0.50)
        d('pair_width_tol', 0.15)
        d('pair_max_angle_deg', 45.0)
        d('row_link_max', 0.42)
        d('path_alpha', 0.8)
        d('fit_order', 2)
        d('max_curvature_coef', 1.5)
        d('lost_hold_sec', 0.5)
        # drive
        d('speed', 0.2)
        d('lookahead', 0.35)
        d('steer_gain', 1.5)
        d('ff_gain', 1.0)
        d('ff_dist', 0.5)
        d('wheelbase', 0.2)
        d('max_steer', 0.6)
        d('path_offset', 0.0)
        # keep the followed path cone_radius + car_half_width away from every cone
        d('cone_radius', 0.08)           # m, cones seen as discs of this radius
        d('car_half_width', 0.11)        # m
        d('cone_push_max', 0.15)         # m, |push| limit
        # backoff
        d('front_x', 0.16)
        d('danger_ahead', 0.20)
        d('safe_half_width', 0.21)
        d('backoff_speed', -0.2)
        d('backoff_sec', 1.0)
        d('backoff_steer_ratio', 1.0)
        d('stuck_window', 6.0)
        d('stuck_count', 3)
        d('stuck_reset_sec', 3.0)
        d('escape_first', 'right')
        d('escape_sec', 1.0)
        d('escape_fwd_sec', 1.0)
        # exit
        d('exit_hold_sec', 1.0)
        d('exit_no_cone_sec', 2.0)
        d('lane_ok_states', ['both'])
        d('input_timeout', 0.5)

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.frame_id = p('frame_id')
        self.laser = (float(p('laser_x')), float(p('laser_y')), float(p('laser_yaw')))
        self.cam_x, self.cam_y = float(p('camera_x')), float(p('camera_y'))
        self.fx, self.cx = float(p('camera_fx')), float(p('camera_cx'))
        self.trigger_class = p('trigger_class')
        self.trigger_conf = float(p('trigger_conf'))
        self.trigger_frames = int(p('trigger_frames'))
        self.enter_dist = float(p('enter_dist'))
        self.bearing_margin = math.radians(float(p('bearing_margin_deg')))
        self.request_timeout = float(p('request_timeout'))
        self.roi_x_max = float(p('roi_x_max'))
        self.roi_y_half = float(p('roi_y_half'))
        self.cluster_gap = float(p('cluster_gap'))
        self.cluster_min_points = int(p('cluster_min_points'))
        self.cluster_max_size = float(p('cluster_max_size'))
        self.lane_width = float(p('lane_width'))
        self.pair_tol = float(p('pair_width_tol'))
        self.pair_sin_max = math.sin(math.radians(float(p('pair_max_angle_deg'))))
        self.row_link = float(p('row_link_max'))
        self.path_alpha = float(p('path_alpha'))
        self.fit_order = int(p('fit_order'))
        self.max_c2 = float(p('max_curvature_coef'))
        self.lost_hold = float(p('lost_hold_sec'))
        self.speed = float(p('speed'))
        self.lookahead = float(p('lookahead'))
        self.steer_gain = float(p('steer_gain'))
        self.ff_gain = float(p('ff_gain'))
        self.ff_dist = float(p('ff_dist'))
        self.wheelbase = float(p('wheelbase'))
        self.max_steer = float(p('max_steer'))
        self.path_offset = float(p('path_offset'))
        self.cone_clear = float(p('cone_radius')) + float(p('car_half_width'))
        self.cone_push_max = float(p('cone_push_max'))
        self.cone_push = 0.0         # m, sideways push of the followed path (last, debug)
        self.danger_x = float(p('front_x')) + float(p('danger_ahead'))
        self.safe_half = float(p('safe_half_width'))
        self.backoff_speed = float(p('backoff_speed'))
        self.backoff_sec = float(p('backoff_sec'))
        self.backoff_steer_ratio = float(p('backoff_steer_ratio'))
        self.stuck_window = float(p('stuck_window'))
        self.stuck_count = int(p('stuck_count'))
        self.stuck_reset = float(p('stuck_reset_sec'))
        self.escape_first = str(p('escape_first')).strip().lower()
        if self.escape_first not in ('right', 'left'):
            raise ValueError(f"escape_first must be 'right' or 'left', not {self.escape_first!r}")
        self.escape_sec = float(p('escape_sec'))
        self.escape_fwd_sec = float(p('escape_fwd_sec'))
        self.exit_hold = float(p('exit_hold_sec'))
        self.exit_no_cone = float(p('exit_no_cone_sec'))
        self.lane_ok = set(p('lane_ok_states'))
        self.input_timeout = float(p('input_timeout'))

        self.points = np.zeros((0, 2))   # newest scan in base_link
        self.scan_at = None
        self.clusters = []               # [(x, y)] cone centres in the ROI, newest scan
        self.pairs = []                  # [(cone, cone, midpoint)] of the newest scan (debug)
        self.left_row, self.right_row = [], []   # labelled cones of the newest scan (debug)
        self.cam_cone_at = None          # s, last camera cone box
        self.lane_state = ''
        self.lane_state_at = None
        self.trig_count = 0
        self.request_at = None
        self.reset_run()

        self.create_subscription(LaserScan, p('scan_topic'), self.scan_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(Detection2DArray, p('detections_topic'),
                                 self.detections_callback, qos_profile_sensor_data)
        self.create_subscription(String, p('lane_state_topic'), self.lane_state_callback, 10)
        self.pub_cmd = self.create_publisher(Twist, p('cmd_topic'), 10)
        self.pub_debug = self.create_publisher(MarkerArray, p('debug_topic'), 1)
        self.get_logger().info(
            f'ready: {self.trigger_class} x{self.trigger_frames} + lidar < {self.enter_dist:.1f} m'
            f' -> [cone] lidar corridor (ROI x < {self.roi_x_max:.1f} m, |y| < '
            f'{self.roi_y_half:.1f} m, width {self.lane_width:.2f} m), {self.speed:.2f} m/s, '
            f'lookahead {self.lookahead:.2f} m, backoff {self.backoff_speed:.2f} m/s x '
            f'{self.backoff_sec:.1f} s, exit after {self.exit_hold:.1f} s clear')

    def reset_run(self):
        self.path = np.zeros(3)      # centre line y = c2 x^2 + c1 x + c0 (np.polyval order)
        self.obs_x = (0.05, self.roi_x_max)  # x range the path was observed in
        self.last_cmd = Twist()
        self.last_fwd_cmd = Twist()  # last command with v > 0 (steering before a backoff)
        self.last_path_at = None     # s, last time a path was found
        self.backoff_until = None
        self.backoff_w = 0.0         # rad/s while backing off
        self.backoff_starts = deque()  # s, recent backoff starts (stuck detection)
        self.escape_level = 0        # escapes done since the last reset (side alternates)
        self.escape_head = ''        # nose direction of the current escape
        self.escape_fwd_until = None  # s, end of the forward half of the escape S
        self.escape_fwd_w = 0.0      # rad/s in the forward half
        self.clear_since = None      # s, no cone + lane ok since
        self.no_cone_since = None    # s, no cone (lidar + camera) since, any lane state

    # ------------------------------------------------------------------ inputs
    def scan_callback(self, msg):
        now = self.now()
        r = np.asarray(msg.ranges, dtype=np.float64)
        ang = msg.angle_min + np.arange(len(r)) * msg.angle_increment
        ok = np.isfinite(r) & (r > max(msg.range_min, 0.01)) & (r < msg.range_max)
        lx, ly, lyaw = self.laser
        x = lx + r * np.cos(ang + lyaw)
        y = ly + r * np.sin(ang + lyaw)
        self.points = np.stack([x[ok], y[ok]], axis=1)
        self.scan_at = now
        self.clusters = self.find_clusters(x, y, ok)
        if self.active and self.state == MissionState.RUNNING:
            self.step(now)

    def detections_callback(self, msg):
        now = self.now()
        boxes = []
        for det in msg.detections:
            for hyp in det.results:
                h = hyp.hypothesis
                if h.class_id == self.trigger_class and h.score >= self.trigger_conf:
                    u0 = det.bbox.center.position.x - det.bbox.size_x / 2.0
                    u1 = det.bbox.center.position.x + det.bbox.size_x / 2.0
                    boxes.append((u0, u1))
        if boxes:
            self.cam_cone_at = now
        if self.state != MissionState.IDLE:
            self.trig_count = 0
            return
        matched = [self.lidar_range_in_box(u0, u1) for u0, u1 in boxes]
        near = [m for m in matched if m is not None and m <= self.enter_dist]
        if not near or self.in_cooldown(now):
            self.trig_count = 0
            return
        self.trig_count += 1
        if self.trig_count >= self.trigger_frames:
            self.trig_count = 0
            self.request_at = now
            self.set_state(MissionState.REQUEST,
                           f'{self.trigger_class} x{self.trigger_frames}, lidar {min(near):.2f} m')

    def lane_state_callback(self, msg):
        self.lane_state, self.lane_state_at = msg.data, self.now()

    # -------------------------------------------------------------- perception
    def lidar_range_in_box(self, u0, u1):
        """Nearest lidar range (from the lidar) inside the box bearing, seen from the camera."""
        if len(self.points) == 0 or self.scan_at is None \
                or self.now() - self.scan_at > self.input_timeout:
            return None
        th_left = math.atan2(self.cx - u0, self.fx) + self.bearing_margin
        th_right = math.atan2(self.cx - u1, self.fx) - self.bearing_margin
        dx, dy = self.points[:, 0] - self.cam_x, self.points[:, 1] - self.cam_y
        bearing = np.arctan2(dy, dx)
        sel = (bearing <= th_left) & (bearing >= th_right) & (dx > 0)
        if not sel.any():
            return None
        lx, ly, _ = self.laser
        return float(np.min(np.hypot(self.points[sel, 0] - lx, self.points[sel, 1] - ly)))

    def find_clusters(self, x, y, ok):
        """Group neighbouring beams in the ROI; return cone centres [(x, y)]."""
        roi = ok & (x > 0.0) & (x < self.roi_x_max) & (np.abs(y) < self.roi_y_half)
        clusters, cur = [], []
        for i in range(len(x)):
            if not roi[i]:
                if cur:
                    clusters.append(cur)
                    cur = []
                continue
            if cur and math.hypot(x[i] - x[cur[-1]], y[i] - y[cur[-1]]) > self.cluster_gap:
                clusters.append(cur)
                cur = []
            cur.append(i)
        if cur:
            clusters.append(cur)
        cones = []
        for idx in clusters:
            if len(idx) < self.cluster_min_points:
                continue
            cx, cy = x[idx], y[idx]
            if math.hypot(cx.max() - cx.min(), cy.max() - cy.min()) > self.cluster_max_size:
                continue  # wall or something bigger than a cone
            cones.append((float(cx.mean()), float(cy.mean())))
        return cones

    def fit_row(self, row, prev):
        """Curve through one row: 2nd order with >= 3 cones (fit_order 2), a line with 2, the
        previous path shape moved onto the cone with 1."""
        xs, ys = np.array([c[0] for c in row]), np.array([c[1] for c in row])
        spread = xs.max() - xs.min()
        if self.fit_order >= 2 and len(row) >= 3 and spread > 0.15:
            c = np.polyfit(xs, ys, 2)
            if abs(c[0]) <= self.max_c2:
                return c  # otherwise too bent to trust: a line
        if len(row) >= 2 and spread > 0.05:
            return np.concatenate([[0.0], np.polyfit(xs, ys, 1)])
        c = prev.copy()
        c[2] += ys.mean() - np.polyval(prev, xs.mean())
        return c

    def through_point(self, prev, x, y):
        """Line through (x, y) with the previous path direction there."""
        slope = 2.0 * prev[0] * x + prev[1]
        return np.array([0.0, slope, y - slope * x])

    def pair_cones(self, prev):
        """Opposite cone pairs: lane_width +- pair_width_tol apart, the connecting line within
        pair_max_angle_deg of perpendicular to the travel direction at their midpoint. Each cone
        is used once, best (most perpendicular) pairs first, so diagonal pairs that would bend
        the centre line are left out when the straight-across partner exists."""
        cones, cand = self.clusters, []
        for i in range(len(cones)):
            for j in range(i + 1, len(cones)):
                (x1, y1), (x2, y2) = cones[i], cones[j]
                dx, dy = x2 - x1, y2 - y1
                dist = math.hypot(dx, dy)
                if abs(dist - self.lane_width) > self.pair_tol:
                    continue
                mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                slope = 2.0 * prev[0] * mx + prev[1]          # travel direction there
                along = abs(dx + dy * slope) / (dist * math.sqrt(1.0 + slope * slope))
                if along > self.pair_sin_max:
                    continue  # too much along the travel direction: same row, not opposite
                cost = along + 0.5 * abs(dist - self.lane_width) / max(self.pair_tol, 1e-3)
                cand.append((cost, i, j, (mx, my)))
        pairs, used = [], set()
        for _, i, j, mid in sorted(cand):
            if i in used or j in used:
                continue
            used.update((i, j))
            pairs.append((cones[i], cones[j], mid))
        return pairs

    def label_rows(self, prev, pairs):
        """Left / right rows: pair cones labelled by their side of the travel direction, then
        the labels spread to cones within row_link_max (nearest labelled neighbour first)."""
        label = {}
        for a, b, (mx, my) in pairs:
            slope = 2.0 * prev[0] * mx + prev[1]          # travel direction (1, slope)
            a_left = (a[1] - my) - slope * (a[0] - mx) > 0  # cross product sign
            label[a], label[b] = ('left', 'right') if a_left else ('right', 'left')
        todo = [c for c in self.clusters if c not in label]
        while todo and label:
            dist = [(min(math.hypot(c[0] - k[0], c[1] - k[1]) for k in label), c) for c in todo]
            d, c = min(dist)
            if d > self.row_link:
                break
            nearest = min(label, key=lambda k: math.hypot(c[0] - k[0], c[1] - k[1]))
            label[c] = label[nearest]
            todo.remove(c)
        left = sorted(c for c, s in label.items() if s == 'left')
        right = sorted(c for c, s in label.items() if s == 'right')
        return left, right

    def corridor(self):
        """Centre path from the newest clusters, or None when no cone is seen."""
        prev = self.path
        self.pairs = self.pair_cones(prev)
        if self.pairs:
            left, right = self.label_rows(prev, self.pairs)
        elif self.clusters:
            row = self.row_of_nearest()          # one row: side from its nearest cone
            left, right = (row, []) if row[0][1] > 0 else ([], row)
        else:
            self.left_row, self.right_row = [], []
            return None, ''
        self.left_row, self.right_row = left, right
        xs = [c[0] for c in left + right]
        self.obs_x = (min(xs), max(xs))
        if left and right:
            c = (self.fit_row(left, prev) + self.fit_row(right, prev)) / 2.0
            return c, f'both ({len(left)}L {len(right)}R)'
        row = left or right
        c = self.fit_row(row, prev)
        slope = 2.0 * c[0] * self.obs_x[0] + c[1]
        c[2] += (-1.0 if left else 1.0) * self.lane_width / 2.0 * math.sqrt(1.0 + slope ** 2)
        return c, 'left row only' if left else 'right row only'

    def row_of_nearest(self):
        """Cones linked to the cone nearest the car by gaps <= row_link_max (nearest first)."""
        todo = sorted(self.clusters, key=lambda c: math.hypot(*c))
        row = [todo.pop(0)]
        grown = True
        while grown:
            grown = False
            for c in list(todo):
                if min(math.hypot(c[0] - r[0], c[1] - r[1]) for r in row) <= self.row_link:
                    row.append(c)
                    todo.remove(c)
                    grown = True
        return row

    def danger(self):
        """Lidar point in the box in front of the car (closer than the safety margin)."""
        if len(self.points) == 0:
            return None
        x, y = self.points[:, 0], self.points[:, 1]
        sel = (x > 0.0) & (x < self.danger_x) & (np.abs(y) < self.safe_half)
        if not sel.any():
            return None
        i = np.flatnonzero(sel)[np.argmin(x[sel])]
        return float(x[i]), float(y[i])

    # ------------------------------------------------------------------- drive
    def step(self, now):
        """One control step per scan while running."""
        if self.backoff_until is not None and now < self.backoff_until:
            self.publish(self.backoff_speed, self.backoff_w)
            return
        hit = self.danger()
        if self.escape_fwd_until is not None and now < self.escape_fwd_until:
            if hit is None:  # second half of the S: forward, wheels to the other side
                self.set_phase(ESCAPE_FWD, f'escape #{self.escape_level}: forward, nose '
                                           f'{self.escape_head}')
                self.publish(self.speed, self.escape_fwd_w)
                return
            self.escape_fwd_until = None  # blocked again: handled as a new backoff below
        if hit is not None:
            self.clear_since = None
            where = f'object at x {hit[0]:.2f} y {hit[1]:+.2f}'
            if self.stuck(now):
                head = self.escape_side(self.escape_level)
                self.escape_level += 1
                self.escape_head = head
                self.backoff_until = now + self.escape_sec
                # S: reverse with the wheels to the other side, then forward with the wheels
                # to `head`; both turn the nose to `head` (same wheels forward = same circle back)
                delta = (self.max_steer if head == 'right' else -self.max_steer)
                self.backoff_w = self.backoff_speed * math.tan(delta) / self.wheelbase
                self.escape_fwd_w = self.speed * math.tan(-delta) / self.wheelbase
                self.escape_fwd_until = self.backoff_until + self.escape_fwd_sec
                self.set_phase(ESCAPE, f'stuck ({self.stuck_count} backoffs in '
                                       f'{self.stuck_window:.0f} s), escape #{self.escape_level}'
                                       f': reverse, nose {head}, {where}')
            else:
                self.backoff_until = now + self.backoff_sec
                delta = self.backoff_steer()
                self.set_phase(BACKOFF, f'{where}, steer {math.degrees(delta):+.0f} deg')
            self.publish(self.backoff_speed, self.backoff_w)
            return
        path, sides = self.corridor()
        if path is not None:
            self.path = self.path_alpha * path + (1.0 - self.path_alpha) * self.path
            self.last_path_at = now
            self.set_phase(TRACK, sides)
            self.publish(self.speed, self.pursuit(self.path))
        elif self.last_path_at is not None and now - self.last_path_at <= self.lost_hold:
            self.set_phase(LOST, f'no cone, keeping the last command {self.lost_hold:.1f} s')
            self.pub_cmd.publish(self.last_cmd)
        else:
            self.set_phase(LOST, 'no cone, stop')
            self.publish(0.0, 0.0)
        self.publish_debug(path is not None)

    def target(self, c):
        """Path point lookahead (arc length) past the nearest observed point, kept inside the
        observed x range: the lidar sees the corridor only from ~0.3 m on, so a target
        measured from the car would always be that first point and ignore the path direction."""
        x0, x1 = max(self.obs_x[0], 0.05), max(self.obs_x[1], self.obs_x[0] + 0.01)
        xs = np.linspace(x0, x1, 60)
        ys = np.polyval(c, xs)
        arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
        i = min(int(np.searchsorted(arc, self.lookahead)), len(xs) - 1)
        return float(xs[i]), float(ys[i])

    def shifted(self, c):
        """
        The followed path: the centre line moved sideways by path_offset (+ left), then
        pushed away from cones closer than cone_radius + car_half_width to it (measured
        sideways at each cone's x). Cones on the right push it left and the other way round;
        both sides at once: the difference (narrow gap: as centred as it gets).
        """
        c = np.array(c, dtype=float)
        c[2] += self.path_offset
        push_left = push_right = 0.0
        for cx, cy in self.clusters:
            if cx <= 0.0:
                continue
            d = cy - float(np.polyval(c, cx))        # + : cone left of the path
            if abs(d) < self.cone_clear:
                need = self.cone_clear - abs(d)
                if d > 0.0:
                    push_right = max(push_right, need)
                else:
                    push_left = max(push_left, need)
        push = max(-self.cone_push_max, min(self.cone_push_max, push_left - push_right))
        self.cone_push = push
        c[2] += push
        return c

    def pursuit(self, c):
        """Pure pursuit towards the lookahead point + curvature feedforward of the path."""
        xt, yt = self.target(self.shifted(c))
        w = self.steer_gain * self.speed * 2.0 * yt / (xt * xt + yt * yt)
        xf = min(max(self.ff_dist, self.obs_x[0]), self.obs_x[1])   # observed range only
        slope = 2.0 * c[0] * xf + c[1]
        kappa = 2.0 * c[0] / (1.0 + slope * slope) ** 1.5
        w += self.ff_gain * self.speed * kappa
        w_max = abs(self.speed) * math.tan(self.max_steer) / self.wheelbase
        return max(-w_max, min(w_max, w))

    def stuck(self, now):
        """
        Record a backoff start; True when it makes stuck_count starts within stuck_window.

        The escape itself stays in the window as the first start of the next count, so
        stuck_count - 1 more backoffs within stuck_window lead to the next escape (other side).
        No backoff for stuck_reset_sec restarts the escape order from escape_first.
        """
        if self.backoff_starts and now - self.backoff_starts[-1] > self.stuck_reset:
            self.escape_level = 0          # got free: count from scratch, first side again
            self.backoff_starts.clear()
        self.backoff_starts.append(now)
        while self.backoff_starts and now - self.backoff_starts[0] > self.stuck_window:
            self.backoff_starts.popleft()
        if len(self.backoff_starts) < self.stuck_count:
            return False
        self.backoff_starts.clear()
        self.backoff_starts.append(now)
        return True

    def escape_side(self, level):
        """Nose direction of escape number level (0, 1, ...): escape_first, then alternating."""
        other = 'left' if self.escape_first == 'right' else 'right'
        return self.escape_first if level % 2 == 0 else other

    def backoff_steer(self):
        """
        Steering for the backoff: the steering angle opposite to the last forward command.

        delta = atan(w L / v) of the last command with v > 0; backing off uses
        -backoff_steer_ratio * delta (limited by max_steer). With v < 0 that gives
        w = v tan(-delta) / L, i.e. the same sign of angular.z as before.
        Sets self.backoff_w and returns the backoff steering angle.
        """
        v, w = self.last_fwd_cmd.linear.x, self.last_fwd_cmd.angular.z
        prev = math.atan(w * self.wheelbase / v) if v > 1e-3 else 0.0
        delta = max(-self.max_steer, min(self.max_steer, -self.backoff_steer_ratio * prev))
        self.backoff_w = self.backoff_speed * math.tan(delta) / self.wheelbase
        return delta

    def publish(self, v, w):
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = float(v), float(w)
        self.last_cmd = cmd
        if v > 0.0:
            self.last_fwd_cmd = cmd
        self.pub_cmd.publish(cmd)

    # ------------------------------------------------------------------- state
    def tick(self, now):
        if self.state == MissionState.REQUEST and now - self.request_at > self.request_timeout:
            self.set_state(MissionState.IDLE, f'not granted within {self.request_timeout:.0f} s')
        if self.state != MissionState.RUNNING or not self.active:
            return
        if self.backoff_until is not None and now < self.backoff_until:
            return
        if self.escape_fwd_until is not None and now < self.escape_fwd_until:
            return  # escaping (forward half of the S): no exit check
        if self.scan_at is None or now - self.scan_at > self.input_timeout:
            self.publish(0.0, 0.0)  # no lidar: hold still
            self.clear_since = self.no_cone_since = None  # not seeing is not "no cone"
            return
        no_lidar_cone = not self.clusters
        no_cam_cone = self.cam_cone_at is None or now - self.cam_cone_at > self.input_timeout
        lane_ok = (self.lane_state in self.lane_ok and self.lane_state_at is not None
                   and now - self.lane_state_at <= self.input_timeout)
        no_cone = no_lidar_cone and no_cam_cone
        if not no_cone:
            self.no_cone_since = self.clear_since = None
            return
        if self.no_cone_since is None:
            self.no_cone_since = now
        if lane_ok and self.clear_since is None:
            self.clear_since = now
        elif not lane_ok:
            self.clear_since = None
        if self.clear_since is not None and now - self.clear_since >= self.exit_hold:
            self.finish(f'no cone (lidar, camera) + lane {self.lane_state} '
                        f'for {self.exit_hold:.1f} s')
        elif now - self.no_cone_since >= self.exit_no_cone:
            self.finish(f'no cone (lidar, camera) for {self.exit_no_cone:.1f} s '
                        f'(lane {self.lane_state or "-"})')

    def on_start(self, now):
        self.reset_run()
        self.set_phase(TRACK, 'start')

    # ------------------------------------------------------------------- debug
    def publish_debug(self, have_path):
        if self.pub_debug.get_subscription_count() == 0:
            return
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        def marker(mid, mtype, rgb, scale):
            m = Marker()
            m.header.frame_id, m.header.stamp = self.frame_id, stamp
            m.ns, m.id, m.type, m.action = 'cone', mid, mtype, Marker.ADD
            m.scale.x = m.scale.y = m.scale.z = scale
            m.color.r, m.color.g, m.color.b, m.color.a = (*rgb, 1.0)
            m.pose.orientation.w = 1.0
            return m

        cones = marker(0, Marker.SPHERE_LIST, (1.0, 0.5, 0.0), 0.06)
        cones.points = [Point(x=c[0], y=c[1]) for c in self.clusters]
        pairs = marker(5, Marker.LINE_LIST, (1.0, 1.0, 0.0), 0.01)
        pairs.points = [Point(x=q[0], y=q[1]) for a, b, _ in self.pairs for q in (a, b)]
        mids = marker(6, Marker.SPHERE_LIST, (1.0, 1.0, 1.0), 0.04)
        mids.points = [Point(x=m[0], y=m[1]) for _, _, m in self.pairs]
        lrow = marker(7, Marker.SPHERE_LIST, (0.0, 0.8, 0.0), 0.07)
        lrow.points = [Point(x=c[0], y=c[1]) for c in self.left_row]
        rrow = marker(8, Marker.SPHERE_LIST, (0.0, 0.3, 1.0), 0.07)
        rrow.points = [Point(x=c[0], y=c[1]) for c in self.right_row]
        line = marker(1, Marker.LINE_STRIP, (0.0, 1.0, 1.0) if have_path else (0.5, 0.5, 0.5),
                      0.02)
        line.points = [Point(x=float(x), y=float(np.polyval(self.path, x)))
                       for x in np.linspace(0.0, self.roi_x_max, 25)]
        target = marker(2, Marker.SPHERE, (1.0, 0.0, 1.0), 0.05)
        target.pose.position.x, target.pose.position.y = self.target(self.shifted(self.path))
        box = marker(3, Marker.LINE_STRIP, (1.0, 0.0, 0.0), 0.01)
        box.points = [Point(x=0.0, y=self.safe_half), Point(x=self.danger_x, y=self.safe_half),
                      Point(x=self.danger_x, y=-self.safe_half), Point(x=0.0, y=-self.safe_half)]
        roi = marker(4, Marker.LINE_STRIP, (0.5, 0.5, 0.5), 0.01)
        roi.points = [Point(x=0.0, y=self.roi_y_half), Point(x=self.roi_x_max, y=self.roi_y_half),
                      Point(x=self.roi_x_max, y=-self.roi_y_half),
                      Point(x=0.0, y=-self.roi_y_half)]
        arr.markers = [cones, pairs, mids, lrow, rrow, line, target, box, roi]
        self.pub_debug.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = ConeNode()
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
