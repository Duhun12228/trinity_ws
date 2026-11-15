"""
Polyline (path) helpers in ROI pixel coordinates.

A path is an (N, 2) float array of (col, row) points ordered from the car (near)
to the far end. col grows to the right, row grows toward the car.
"Left" means the vehicle's left (-col when heading forward).
Arc lengths and distances are in pixels.
"""
import math

import numpy as np


def arc_length(pts):
    """Cumulative arc length at every point (starts at 0)."""
    seg = np.diff(pts, axis=0)
    return np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])


def resample(pts, step):
    """Points every `step` px along the path (end point kept)."""
    pts = np.asarray(pts, float)
    s = arc_length(pts)
    if len(pts) < 2 or s[-1] < 1e-6:
        return pts[:1].copy()
    ss = np.arange(0.0, s[-1], step)
    if s[-1] - ss[-1] > step * 0.5:
        ss = np.append(ss, s[-1])
    else:
        ss[-1] = s[-1]
    return np.stack([np.interp(ss, s, pts[:, 0]), np.interp(ss, s, pts[:, 1])], axis=1)


def smooth(pts, window):
    """Smooth with a moving average; the window shrinks near the ends (ends stay put)."""
    n = len(pts)
    half = int(window) // 2
    if n < 3 or half < 1:
        return pts.copy()
    out = np.empty_like(pts)
    for i in range(n):
        h = min(half, i, n - 1 - i)
        out[i] = pts[i - h:i + h + 1].mean(axis=0)
    return out


def tangents(pts):
    """Return the unit tangent at every point (direction of travel: near -> far)."""
    t = np.gradient(pts, axis=0) if len(pts) > 1 else np.array([[0.0, -1.0]])
    return t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-12)


def left_normals(t):
    """Return unit normals pointing to the left of the travel direction."""
    return np.stack([t[:, 1], -t[:, 0]], axis=1)


def offset(pts, dist):
    """
    Shift every point `dist` px along its normal (+ = left).

    Back-folding points (inside of a sharp corner) are removed so the result keeps
    moving forward.
    """
    t = tangents(pts)
    moved = pts + dist * left_normals(t)
    keep = [0]
    for i in range(1, len(moved)):
        if np.dot(moved[i] - moved[keep[-1]], t[i]) > 1e-6:
            keep.append(i)
    return moved[keep]


def project(pts, p):
    """
    Closest point on the path to p.

    The first segment is extended backward and the last one forward, so a point
    before the start / past the end still gets a perpendicular foot.
    Returns (s, q, t): arc length of the foot (may be < 0 or > length), foot
    point, unit tangent of that segment.
    """
    p = np.asarray(p, float)
    if len(pts) < 2:
        return 0.0, pts[0].copy(), np.array([0.0, -1.0])
    a = pts[:-1]
    ab = pts[1:] - a
    seg_len = np.hypot(ab[:, 0], ab[:, 1])
    ok = seg_len > 1e-9
    if not ok.any():
        return 0.0, pts[0].copy(), np.array([0.0, -1.0])
    seg_len = np.where(ok, seg_len, 1.0)
    u = ((p - a) * ab).sum(axis=1) / (seg_len * seg_len)
    lo = np.zeros(len(u))
    hi = np.ones(len(u))
    lo[0] = -np.inf     # first segment extends backward
    hi[-1] = np.inf     # last segment extends forward
    u = np.clip(u, lo, hi)
    q = a + u[:, None] * ab
    d = np.hypot(q[:, 0] - p[0], q[:, 1] - p[1])
    d[~ok] = np.inf
    i = int(np.argmin(d))
    s_cum = arc_length(pts)
    return float(s_cum[i] + u[i] * seg_len[i]), q[i], ab[i] / seg_len[i]


def points_at(pts, ss):
    """Points at the arc lengths ss (array), straight extension outside the path."""
    ss = np.asarray(ss, float)
    if len(pts) < 2:
        return np.repeat(pts[:1], len(ss), axis=0)
    s_cum = arc_length(pts)
    out = np.stack([np.interp(ss, s_cum, pts[:, 0]), np.interp(ss, s_cum, pts[:, 1])], axis=1)
    t0 = pts[1] - pts[0]
    t0 = t0 / (np.linalg.norm(t0) + 1e-12)
    t1 = pts[-1] - pts[-2]
    t1 = t1 / (np.linalg.norm(t1) + 1e-12)
    before = ss < 0.0
    after = ss > s_cum[-1]
    out[before] = pts[0] + ss[before, None] * t0
    out[after] = pts[-1] + (ss[after, None] - s_cum[-1]) * t1
    return out


def point_at(pts, s):
    """Point and unit tangent at arc length s (straight extension outside the path)."""
    if len(pts) < 2:
        return pts[0].copy(), np.array([0.0, -1.0])
    s_cum = arc_length(pts)
    i = int(np.clip(np.searchsorted(s_cum, s) - 1, 0, len(pts) - 2))
    seg = pts[i + 1] - pts[i]
    t = seg / (np.linalg.norm(seg) + 1e-12)
    return points_at(pts, [s])[0], t


def anchored(pts, p, step):
    """
    Resample the path from the foot of p (s = 0 at p), every `step` px.

    At least two points (extended straight if the path ends right there).
    """
    s0, _, _ = project(pts, p)
    end = max(arc_length(pts)[-1], s0 + step)
    ss = s0 + np.arange(0.0, end - s0 + 1e-9, step)
    if len(ss) < 2:
        ss = np.array([s0, s0 + step])
    return points_at(pts, ss)


def blend(new, prev, w_new):
    """
    Mix two anchored paths point by point over their common length.

    The rest of `new` is kept as is.
    """
    if prev is None:
        return new.copy()
    n = min(len(new), len(prev))
    out = new.copy()
    out[:n] = w_new * new[:n] + (1.0 - w_new) * prev[:n]
    return out


def signed_lateral(pts, p):
    """Lateral distance [px] from p to the path, + when the path is left of p."""
    _, q, t = project(pts, p)
    n_left = np.array([t[1], -t[0]])
    return float(np.dot(q - np.asarray(p, float), n_left))


def heading(t):
    """Angle [rad] of a direction vs straight ahead (up), + = turning left."""
    return math.atan2(-t[0], -t[1])


def smooth_extended(pts, length):
    """
    Smooth with a moving average over `length` px of path, also at the ends.

    The path is extended straight past both ends first, so the near end (where the
    controller looks) is smoothed too instead of being pinned.
    """
    if length <= 0 or len(pts) < 2:
        return pts
    step = float(np.mean(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    if step < 1e-6:
        return pts
    half = int(length / 2.0 / step)
    if half < 1:
        return pts
    t0 = pts[1] - pts[0]
    t0 = t0 / (np.linalg.norm(t0) + 1e-12)
    t1 = pts[-1] - pts[-2]
    t1 = t1 / (np.linalg.norm(t1) + 1e-12)
    k = np.arange(1, half + 1)[:, None] * step
    ext = np.vstack([pts[0] - t0 * k[::-1], pts, pts[-1] + t1 * k])
    ker = np.ones(2 * half + 1) / (2 * half + 1)
    sm = np.stack([np.convolve(ext[:, i], ker, mode='same') for i in range(2)], axis=1)
    return sm[half:half + len(pts)]
