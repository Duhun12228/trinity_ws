"""
Stanley lateral controller on the BEV centre path.

  delta = yaw_k * yaw + atan2(steer_k * e, max(v, min_stanley_speed))
  e   = lateral distance [m] from the reference point (vehicle centre at the near
        edge of the ROI) to the centre path, + = path is left of the car,
        plus center_offset: the car follows the centre path shifted by it (+ left / - right)
  yaw = (1 - w) * heading of the path from the nearest point to near_span ahead
        + w * heading lookahead_dist further along the path   (w = lookahead_weight)
        heading + = path turns left. The look-ahead term starts the turn before a
        sharp corner reaches the reference point.
The centre path is first smoothed over center_smooth metres (also at the near end):
a longer window averages out zigzags / detection jitter but rounds sharp corners.
Then: clip to max_steer, smooth with the previous steer, slow down on fast steer
changes, omega = v * tan(delta) / wheelbase  (ROS: +angular.z = left).
"""
import math

import numpy as np

from mission import path as pth

DEFAULT_STANLEY_PARAMS = {
    'base_speed': 0.4,              # m/s
    'steer_k': 1.0,                 # 1/m, cross-track gain (0.002/px of the original ~ 1/m)
    'yaw_k': 1.0,
    'max_steer': 0.6,               # rad
    'steer_smoothing_alpha': 1.0,   # weight of the new steer (1.0 = no smoothing)
    'steer_slowdown_ratio': 0.35,   # speed drop at a full max_steer change
    'min_smooth_speed': 0.25,       # m/s, floor of the slowdown (never above the target)
    'min_stanley_speed': 0.1,       # m/s, floor of v inside atan2
    'wheelbase': 0.23,              # m
    'center_smooth': 0.30,          # m, moving average over the centre path (0 = off)
    'near_span': 0.06,              # m, near heading = direction over this path length
    'lookahead_dist': 0.20,         # m, along the path from the nearest point
    'lookahead_weight': 0.4,        # share of the look-ahead heading in yaw
    'center_offset': 0.0,           # m, drive this far beside the centre path (+ left / - right)
}


class StanleyController:

    def __init__(self, **params):
        p = dict(DEFAULT_STANLEY_PARAMS)
        p.update({k: v for k, v in params.items() if k in DEFAULT_STANLEY_PARAMS})
        self.p = p
        self.prev_steer = None
        self.yaw = 0.0
        self.yaw_near = 0.0
        self.yaw_ahead = 0.0
        self.error = 0.0
        self.lateral = 0.0          # m, centre path y from the car (+ = left), no center_offset
        self.steer = 0.0
        self.speed = 0.0
        self.omega = 0.0
        self.near_pt = None         # nearest path point (debug)
        self.ahead_pt = None        # look-ahead path point (debug)
        self.cpath_used = None      # smoothed centre path the control used (debug)

    def yaw_error(self, cpath, ref, resolution):
        """Heading [rad] (near / look-ahead blend) and lateral error [m]."""
        s0, q, _ = pth.project(cpath, ref)
        span_pt, _ = pth.point_at(cpath, s0 + self.p['near_span'] / resolution)
        ahead, t_ahead = pth.point_at(cpath, s0 + self.p['lookahead_dist'] / resolution)
        self.near_pt, self.ahead_pt = q, ahead
        # direction over a short span instead of one segment: robust to window jitter
        self.yaw_near = pth.heading(span_pt - q)
        self.yaw_ahead = pth.heading(t_ahead)
        w = self.p['lookahead_weight']
        yaw = (1.0 - w) * self.yaw_near + w * self.yaw_ahead
        self.lateral = pth.signed_lateral(cpath, ref) * resolution
        return yaw, self.lateral + self.p['center_offset']

    def compute(self, cpath, speed_scale, ref, resolution):
        """Return (linear speed, angular velocity). Stops when cpath is None or scale is 0."""
        if cpath is None or speed_scale <= 0.0:
            self.prev_steer = None
            self.speed = self.omega = self.steer = 0.0
            self.near_pt = self.ahead_pt = self.cpath_used = None
            return 0.0, 0.0

        p = self.p
        cpath = pth.smooth_extended(cpath, p['center_smooth'] / resolution)
        self.cpath_used = cpath
        self.yaw, self.error = self.yaw_error(cpath, ref, resolution)
        v_target = p['base_speed'] * speed_scale

        raw = p['yaw_k'] * self.yaw + math.atan2(p['steer_k'] * self.error,
                                                 max(abs(v_target), p['min_stanley_speed']))
        raw = float(np.clip(raw, -p['max_steer'], p['max_steer']))

        if self.prev_steer is None:
            delta_change, steer = 0.0, raw
        else:
            delta_change = raw - self.prev_steer
            steer = self.prev_steer + p['steer_smoothing_alpha'] * delta_change
            steer = float(np.clip(steer, -p['max_steer'], p['max_steer']))

        change_ratio = min(abs(delta_change) / max(abs(p['max_steer']), 0.01), 1.0)
        v = v_target * (1.0 - p['steer_slowdown_ratio'] * change_ratio)
        v = max(v, min(p['min_smooth_speed'], v_target))

        self.prev_steer = steer
        self.steer = steer
        self.speed = float(v)
        self.omega = float(v * math.tan(steer) / p['wheelbase'])
        return self.speed, self.omega
