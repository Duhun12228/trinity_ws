"""
Sliding-window lane detection on the BEV lane-mask ROI -> one path per side.

All tuning values are given in metres and converted to pixels with the BEV
resolution, so they stay valid when the BEV range / resolution changes.

Image coordinates: col = x (right), row = y (down, toward the car).
Result per side: path = (N, 2) polyline (col, row) through the window centres,
near -> far, resampled every path_step and smoothed; None when there is no line.
A single line fit cannot follow a sharp corner, the window path can.

Pipeline (detect):
  1. start point per side: scan from the image centre outward (left / right) in a
     band at the bottom of the ROI; the first line met is the start, so the line
     closest to the vehicle centre wins. No hit -> move the band up.
  2. the side whose start is closer to the car tracks first and occupies a
     band along its path; the other side restarts on unoccupied pixels
  3. windows follow the line (up to 2 * n_windows)
  4. make_path: polyline through the found window centres (no angle limit)
"""
from dataclasses import dataclass, field
import cv2
import numpy as np

from mission import path as pth

DEFAULT_LANE_PARAMS = {
    'occupy_width': 0.20,       # m, band width marked as taken by the first side
    'window_width': 0.20,       # m, window size across the line
    'n_windows': 6,             # window height = ROI height / n_windows, max 2 * n_windows
    'start_band_height': 0.10,  # m, height of the band scanned for the start point
    'start_bin_width': 0.05,    # m, column bin moved outward from the centre
    'min_start_pixels': 20,     # pixels in a bin to count it as a line
    'min_window_pixels': 10,    # pixels in a window to count it as found
    'max_empty_windows': 3,     # consecutive empty windows after the first hit -> stop
    'dir_blend': 0.5,           # weight of the new movement direction
    'min_fit_pixels': 30,       # fewer pixels in all found windows -> no line
    'min_fit_windows': 2,       # fewer found windows -> no line
    'path_step': 0.02,          # m, path resampling step
    'path_smooth': 5,           # points in the moving average over the resampled path
}

# stop when the heading points this much toward the car (down) -- tolerates noise on
# horizontal lines, where the vertical component hovers around 0
DOWN_STOP_DY = 0.1


@dataclass
class Window:
    cx: float
    cy: float
    half_w: float
    half_h: float
    found: bool


@dataclass
class SideResult:
    name: str
    start: tuple = None                          # (x, y) start point, None if no line
    windows: list = field(default_factory=list)  # Window list (found and empty)
    pixel_idx: np.ndarray = None                 # indices into the point arrays
    centers: np.ndarray = None                   # raw path points (start + window centres)
    path: np.ndarray = None                      # (N, 2) smoothed path, None if no line
    drivable_dist: float = None                  # m, median distance to the drivable area
    rejected_path: np.ndarray = None             # path dropped by a later filter (debug)
    reject: str = ''                             # why the side has no path


class SlidingWindowLaneDetector:

    def __init__(self, resolution, **params):
        p = dict(DEFAULT_LANE_PARAMS)
        p.update({k: v for k, v in params.items() if k in DEFAULT_LANE_PARAMS})
        self.p = p
        self.res = float(resolution)
        px = self.m_to_px
        self.occupy_px = max(1, int(round(px(p['occupy_width']))))
        self.window_w_px = px(p['window_width'])
        self.band_h_px = max(1, int(round(px(p['start_band_height']))))
        self.bin_w_px = max(1, int(round(px(p['start_bin_width']))))
        self.path_step_px = max(1.0, px(p['path_step']))

    def m_to_px(self, metres):
        return float(metres) / self.res

    # ------------------------------------------------------------- 1. start
    def find_start(self, xs, ys, avail, side, center_x, roi_h):
        """Start point and heading of one side, or None (no line on that side)."""
        in_half = (xs < center_x) if side == 'left' else (xs >= center_x)
        sel = avail & in_half
        if not sel.any():
            return None

        hit = self.scan_from_center(xs, ys, sel, side, center_x, roi_h)
        if hit is None:
            return None
        x0, bottom = hit

        hw = self.window_w_px / 2.0
        nb = sel & (np.abs(xs - x0) <= hw) & (ys >= bottom - self.window_w_px) & (ys <= bottom)
        d = self.principal_direction(xs[nb], ys[nb])
        # Face forward (up) whenever up/down is distinguishable, also for lying lines:
        # facing the centre would point an outward-going line (outer line of a curve)
        # toward the car and stop tracking at once. Only a near-horizontal line, where
        # up/down is noise, faces the image centre.
        if abs(d[1]) > DOWN_STOP_DY:
            if d[1] > 0:
                d = -d
        elif (side == 'left') != (d[0] > 0):
            d = -d
        return np.array([x0, float(bottom)]), d

    def scan_from_center(self, xs, ys, sel, side, center_x, roi_h):
        """
        Scan outward from the centre column, bottom band first.

        Returns (x0, bottom) of the first column bin with >= min_start_pixels,
        or None. x0 = median column of the bin pixels, bottom = their lowest row.
        """
        c = int(round(center_x))
        band_top = roi_h - self.band_h_px
        while band_top > -self.band_h_px:
            in_band = sel & (ys >= max(band_top, 0)) & (ys < band_top + self.band_h_px)
            if in_band.sum() >= self.p['min_start_pixels']:
                bx = xs[in_band]
                hist = np.bincount(bx, minlength=c * 2 + 1)
                # sliding bin sums, scanned one column at a time away from the centre
                sums = np.convolve(hist, np.ones(self.bin_w_px, int), mode='valid')
                if side == 'left':  # bin [k, k + w) must end left of the centre
                    starts = range(min(c - self.bin_w_px, len(sums) - 1), -1, -1)
                else:
                    starts = range(c, len(sums))
                for k in starts:
                    if sums[k] >= self.p['min_start_pixels']:
                        in_bin = in_band & (xs >= k) & (xs < k + self.bin_w_px)
                        return float(np.median(xs[in_bin])), float(ys[in_bin].max())
            band_top -= self.band_h_px
        return None

    @staticmethod
    def principal_direction(x, y):
        pts = np.stack([x, y], axis=1).astype(np.float64)
        pts -= pts.mean(axis=0)
        cov = pts.T @ pts / max(1, len(pts) - 1)
        w, v = np.linalg.eigh(cov)
        d = v[:, np.argmax(w)]
        return d / (np.linalg.norm(d) + 1e-12)

    # ------------------------------------------------------------- 3. track
    def window_half_size(self, d, win_h):
        """Long side across the line: lying line -> stand the window up."""
        if abs(d[0]) > abs(d[1]):
            return win_h / 2.0, self.window_w_px / 2.0
        return self.window_w_px / 2.0, win_h / 2.0

    def track(self, xs, ys, avail, start, d, roi_w, roi_h):
        """Move windows along the line. Returns (windows, pixel indices, path points)."""
        win_h = roi_h / float(self.p['n_windows'])
        start = np.asarray(start, float)
        # first window: half a step from the start point along the heading, so it covers
        # the start pixels for vertical and lying lines alike
        pos = start + d * (win_h / 2.0)
        prev_c = start.copy()  # movement is measured from the start point
        windows, idx_list = [], []
        centers = [start.copy()]
        found_any, empty_run = False, 0

        for _ in range(2 * int(self.p['n_windows'])):
            if not (0 <= pos[0] < roi_w and 0 <= pos[1] < roi_h):
                break
            hw, hh = self.window_half_size(d, win_h)
            inside = avail & (np.abs(xs - pos[0]) <= hw) & (np.abs(ys - pos[1]) <= hh)
            idx = np.flatnonzero(inside)
            found = len(idx) >= self.p['min_window_pixels']
            windows.append(Window(pos[0], pos[1], hw, hh, found))

            if found:
                found_any, empty_run = True, 0
                idx_list.append(idx)
                c = np.array([xs[idx].mean(), ys[idx].mean()])
                move = c - prev_c
                n = np.linalg.norm(move)
                if n > 1.0:
                    blend = (1 - self.p['dir_blend']) * d + self.p['dir_blend'] * move / n
                    d = blend / (np.linalg.norm(blend) + 1e-12)
                centers.append(c)
                prev_c = c
                pos = c + d * win_h
            else:
                if found_any:
                    empty_run += 1
                    if empty_run >= self.p['max_empty_windows']:
                        break
                pos = pos + d * win_h

            if d[1] > DOWN_STOP_DY:  # heading turned toward the car
                break

        pix = np.unique(np.concatenate(idx_list)) if idx_list else np.empty(0, int)
        return windows, pix, np.array(centers)

    def occupy(self, windows, roi_shape):
        """Band of occupy_width along the found-window path."""
        occ = np.zeros(roi_shape, np.uint8)
        pts = np.array([[w.cx, w.cy] for w in windows if w.found]).round().astype(np.int32)
        if len(pts) == 1:
            cv2.circle(occ, tuple(int(v) for v in pts[0]), self.occupy_px // 2, 255, -1)
        elif len(pts) > 1:
            cv2.polylines(occ, [pts], False, 255, self.occupy_px)
        return occ

    # -------------------------------------------------------------- 4. path
    def make_path(self, res):
        """Polyline through the found window centres."""
        n_win = sum(w.found for w in res.windows)
        n_pix = 0 if res.pixel_idx is None else len(res.pixel_idx)
        if n_pix < self.p['min_fit_pixels'] or n_win < self.p['min_fit_windows']:
            res.reject = f'few pixels/windows ({n_pix}px, {n_win}win)'
            return
        # window centres only: the start point sits on the inner edge of the line (the
        # centre-outward scan stops there), which would bend the near end of the path
        pts = res.centers[1:]
        if len(pts) < 2:
            res.reject = 'path too short'
            return
        path = pth.smooth(pth.resample(pts, self.path_step_px), self.p['path_smooth'])
        if len(path) < 2:
            res.reject = 'path too short'
            return
        res.path = path

    # --------------------------------------------------------------- detect
    def detect(self, roi):
        """Return {'left': SideResult, 'right': SideResult} for a 0/255 ROI mask."""
        h, w = roi.shape[:2]
        ys, xs = np.nonzero(roi)
        center_x = w / 2.0
        results = {s: SideResult(s) for s in ('left', 'right')}
        self.points = (xs, ys)  # ROI pixel coordinates, indexed by pixel_idx

        avail = np.ones(len(xs), bool)
        starts = {s: self.find_start(xs, ys, avail, s, center_x, h) for s in results}
        found = [s for s in starts if starts[s] is not None]
        # closer to the car (larger row) goes first
        order = sorted(found, key=lambda s: -starts[s][0][1])
        if len(order) == 1:
            order.append('right' if order[0] == 'left' else 'left')

        for i, side in enumerate(order):
            res = results[side]
            st = starts.get(side) if i == 0 else self.find_start(xs, ys, avail, side, center_x, h)
            if st is None:
                res.reject = 'no start'
                continue
            res.start = tuple(st[0])
            res.windows, res.pixel_idx, res.centers = self.track(
                xs, ys, avail, st[0], st[1], w, h)
            if i == 0:
                occ = self.occupy(res.windows, (h, w))
                avail = occ[ys, xs] == 0
        for side in ('left', 'right'):
            res = results[side]
            if res.start is None:
                res.reject = 'no start'
            else:
                self.make_path(res)
        return results

    # ---------------------------------------------------------------- debug
    def draw_debug(self, roi, results):
        """
        Debug image of the detection.

        White = lane pixels, red / blue = left / right window pixels,
        green = windows (dark = empty), yellow = lane paths.
        """
        h, w = roi.shape[:2]
        vis = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        xs, ys = self.points
        colors = {'left': (0, 0, 255), 'right': (255, 0, 0)}
        for side, res in results.items():
            if res.pixel_idx is not None and len(res.pixel_idx):
                vis[ys[res.pixel_idx], xs[res.pixel_idx]] = colors[side]
        for res in results.values():
            for win in res.windows:
                p1 = (int(round(win.cx - win.half_w)), int(round(win.cy - win.half_h)))
                p2 = (int(round(win.cx + win.half_w)), int(round(win.cy + win.half_h)))
                cv2.rectangle(vis, p1, p2, (0, 255, 0) if win.found else (0, 100, 0), 1)
        for res in results.values():
            if res.path is not None:
                cv2.polylines(vis, [np.round(res.path).astype(np.int32)], False,
                              (0, 255, 255), 1, cv2.LINE_AA)
        return vis
