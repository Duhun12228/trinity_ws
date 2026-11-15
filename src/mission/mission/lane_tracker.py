"""
Turn the lane paths into a centre path for the controller.

Centre candidates: each lane path shifted half a lane width along its normal
toward the road (left line -> to its right, right line -> to its left). Shifting
along the normal keeps the width right in curves and corners.

Cases
  both lines  -> average of the two centre candidates, reset the side decision
  one line    -> decide whether it is the left or right line (decide_side),
                 blend it with the previous path of that side (track_alpha),
                 centre = its candidate; speed x single_speed_scale
  no line     -> reuse the last centre, speed x lost_speed_scale; after
                 lost_stop_frames consecutive frames -> stop
The centre path is then smoothed over frames (EMA, center_ema_prev = weight of
the previous centre); the history restarts after a stop.

Paths are compared point by point after resampling them from the reference point
(the vehicle centre at the near edge of the ROI), see path.anchored().

decide_side score (positive = left line):
  drivable   : drivable ratio in bands right vs left of the line (BEV drivable mask)
  position   : lateral position Y of the line point nearest to the reference point
  continuity : line left / right of the previous centre path (recent only)
All lengths are metres, converted with the BEV resolution.
"""
from dataclasses import dataclass, field

import numpy as np

from mission import path as pth

DEFAULT_TRACK_PARAMS = {
    'lane_width': 0.50,             # m, distance between the two lane lines
    'side_w_drivable': 1.0,
    'side_w_position': 0.3,
    'side_w_continuity': 0.6,
    'side_threshold': 0.2,          # |score| above this decides, else keep the decision
    'side_switch_frames': 3,        # same new decision this many frames in a row to switch
    'drivable_band_min': 0.04,      # m, band start from the line (skips the line itself)
    'drivable_band_max': 0.30,      # m, band end
    'drivable_band_step': 0.02,     # m, sample spacing inside the band
    'drivable_n_points': 8,         # sample points along the line
    'position_saturation': 0.20,    # m, |Y| where the position score is maximal
    'continuity_max_age': 5,        # frames, previous centre older than this is not used
    'track_alpha': 0.8,             # new path weight when blending with the previous one
    'center_ema_prev': 0.3,         # centre path EMA: weight of the previous centre
    'single_speed_scale': 0.8,
    'lost_speed_scale': 0.5,
    'lost_stop_frames': 15,
}


@dataclass
class TrackResult:
    cpath: np.ndarray = None        # centre path anchored at the reference point, or None
    lpath: np.ndarray = None        # left line used (observed or virtual), for debug
    rpath: np.ndarray = None
    status: str = 'none'            # both / single_left / single_right / lost / stop
    speed_scale: float = 0.0
    virtual: str = None             # 'left' / 'right': which line was synthesised
    scores: dict = field(default_factory=dict)


class LaneTracker:

    def __init__(self, resolution, step_px, **params):
        p = dict(DEFAULT_TRACK_PARAMS)
        p.update({k: v for k, v in params.items() if k in DEFAULT_TRACK_PARAMS})
        self.p = p
        self.res = float(resolution)
        self.step = float(step_px)
        self.half_px = p['lane_width'] / 2.0 / self.res
        offs = np.arange(p['drivable_band_min'], p['drivable_band_max'] + 1e-9,
                         p['drivable_band_step'])
        self.band_offsets_px = offs / self.res

        self.prev_l = None          # previous left / right line paths (anchored)
        self.prev_r = None
        self.prev_c = None          # previous centre path (anchored, after EMA)
        self.center_age = 10 ** 6   # frames since the centre came from an observed line
        self.lost_count = 0
        self.reset_side()

    def reset_side(self):
        self.side = None            # current single-line decision
        self.pending = None
        self.pending_count = 0
        self.prev_start = None      # start half of the single line in the previous frame

    # ---------------------------------------------------------------- update
    def update(self, lanes, ref, drivable=None, visible=None):
        """
        Lanes: {'left'/'right': SideResult}; ref: (col, row) reference point.

        drivable / visible: ROI-sized bool arrays (BEV drivable mask, pixels inside the
        camera view) or None when no drivable mask of the same frame is available.
        """
        valid = {k: v for k, v in lanes.items() if v.path is not None}
        out = TrackResult()
        center = None

        if len(valid) == 2:
            self.reset_side()
            out.lpath = pth.anchored(valid['left'].path, ref, self.step)
            out.rpath = pth.anchored(valid['right'].path, ref, self.step)
            center = self.average(
                pth.anchored(self.center_candidate(out.lpath, 'left'), ref, self.step),
                pth.anchored(self.center_candidate(out.rpath, 'right'), ref, self.step))
            out.status, out.speed_scale = 'both', 1.0
            self.lost_count = 0
        elif len(valid) == 1:
            res = next(iter(valid.values()))
            side, out.scores = self.decide_side(res, ref, drivable, visible)
            line = self.track_one(pth.anchored(res.path, ref, self.step), side)
            virtual = pth.offset(line, (-2.0 if side == 'left' else 2.0) * self.half_px)
            if side == 'left':
                out.lpath, out.rpath, out.virtual = line, virtual, 'right'
            else:
                out.lpath, out.rpath, out.virtual = virtual, line, 'left'
            center = pth.anchored(self.center_candidate(line, side), ref, self.step)
            out.status = f'single_{side}'
            out.speed_scale = self.p['single_speed_scale']
            self.lost_count = 0
        else:
            self.lost_count += 1
            out.lpath, out.rpath = self.prev_l, self.prev_r
            if self.prev_c is None or self.lost_count >= self.p['lost_stop_frames']:
                out.status, out.speed_scale = 'stop', 0.0
            else:
                out.status, out.speed_scale = 'lost', self.p['lost_speed_scale']
                center = pth.anchored(self.prev_c, ref, self.step)

        if out.status == 'stop':
            self.prev_c = None      # no EMA history across a stop
        elif center is not None:
            k = self.p['center_ema_prev']
            prev = None if self.prev_c is None else pth.anchored(self.prev_c, ref, self.step)
            out.cpath = pth.blend(center, prev, 1.0 - k)
            self.prev_c = out.cpath
            if out.lpath is not None:
                self.prev_l, self.prev_r = out.lpath, out.rpath
        observed = out.status in ('both', 'single_left', 'single_right')
        self.center_age = 0 if observed else self.center_age + 1
        return out

    def center_candidate(self, line, side):
        """Lane line shifted half a lane width along its normal toward the road."""
        return pth.offset(line, (-1.0 if side == 'left' else 1.0) * self.half_px)

    @staticmethod
    def average(a, b):
        """Point-wise mean of two anchored paths over the common length."""
        n = min(len(a), len(b))
        if n < 2:
            return a if len(a) >= len(b) else b
        return (a[:n] + b[:n]) / 2.0

    # ------------------------------------------------------------ decide side
    def decide_side(self, res, ref, drivable, visible):
        """Left / right decision for a single line, with hysteresis."""
        scores = {
            'drivable': self.drivable_score(res.path, drivable, visible),
            'position': self.position_score(res.path, ref),
            'continuity': self.continuity_score(res.path, ref),
        }
        total = (self.p['side_w_drivable'] * scores['drivable']
                 + self.p['side_w_position'] * scores['position']
                 + self.p['side_w_continuity'] * scores['continuity'])
        scores['total'] = total

        if total > self.p['side_threshold']:
            cand = 'left'
        elif total < -self.p['side_threshold']:
            cand = 'right'
        else:
            cand = self.side if self.side is not None else res.name  # keep / fall back

        start_changed = self.prev_start is not None and res.name != self.prev_start
        self.prev_start = res.name

        if self.side is None or cand == self.side or start_changed:
            self.side = cand
            self.pending, self.pending_count = None, 0
        else:
            if cand == self.pending:
                self.pending_count += 1
            else:
                self.pending, self.pending_count = cand, 1
            if self.pending_count >= self.p['side_switch_frames']:
                self.side = cand
                self.pending, self.pending_count = None, 0
        return self.side, scores

    def drivable_score(self, line, drivable, visible):
        """(drivable ratio right of the line - left of it) x sample confidence."""
        if drivable is None or visible is None:
            return 0.0
        h, w = drivable.shape
        s = pth.arc_length(line)
        ss = np.linspace(0.0, s[-1], int(self.p['drivable_n_points']))
        pts = pth.points_at(line, ss)
        t = pth.tangents(line)
        t = np.stack([np.interp(ss, s, t[:, 0]), np.interp(ss, s, t[:, 1])], axis=1)
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
        n_right = -pth.left_normals(t)
        ratio, n_vis = {}, 0
        for sign, name in ((1.0, 'right'), (-1.0, 'left')):
            off = sign * self.band_offsets_px[None, :, None] * n_right[:, None, :]
            q = np.rint(pts[:, None, :] + off).astype(int).reshape(-1, 2)
            c, r = q[:, 0], q[:, 1]
            inside = (c >= 0) & (c < w) & (r >= 0) & (r < h)
            c, r = c[inside], r[inside]
            vis = visible[r, c]
            cnt = int(vis.sum())
            if cnt == 0:
                return 0.0  # one side completely unknown
            ratio[name] = float(drivable[r[vis], c[vis]].mean())
            n_vis += cnt
        total = 2 * len(ss) * len(self.band_offsets_px)
        return (ratio['right'] - ratio['left']) * min(1.0, n_vis / total)

    def position_score(self, line, ref):
        """Line point nearest to the reference left (+Y) of the car -> positive."""
        _, q, _ = pth.project(line, ref)
        y_m = (ref[0] - q[0]) * self.res
        return float(np.clip(y_m / self.p['position_saturation'], -1.0, 1.0))

    def continuity_score(self, line, ref):
        """Line left of the previous centre path -> +1, right -> -1 (recent centre only)."""
        if self.prev_c is None or self.center_age > self.p['continuity_max_age']:
            return 0.0
        _, q, _ = pth.project(line, ref)
        # + when the previous centre is left of the line point -> the line is a right line
        return -1.0 if pth.signed_lateral(self.prev_c, q) > 0 else 1.0

    # -------------------------------------------------------------- track one
    def track_one(self, line, side):
        """Blend the anchored line with the previous same-side line."""
        prev = self.prev_l if side == 'left' else self.prev_r
        return pth.blend(line, prev, self.p['track_alpha'])
