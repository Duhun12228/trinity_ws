"""
3road map + planner (pure numpy, no ROS).

Frame "map": the 2026 track drawing (DXF), metres, origin = track corner, the car drives
-x through the section. Lane 1 = A road (y 0.16-0.68), lane 2 (0.68-1.20, ends at the
island), lane 3 = B road (1.20-1.72). The yellow lines (outer border, entry / exit tapers,
island) are the only hard limits; the white dashed lines do not matter.

Map
  free0  drivable cells (inside the yellow lines), a RES grid
  hits   lidar hits per cell (map frame); >= hit_min = occupied
  seen   cells some lidar ray went through (unseen = behind an object: costs a little)
  side   1 (A) / 3 (B) once the sign decided: the other side of the island is a wall
  clear  distance (m) from each cell to the nearest blocked cell (EDT)

Planner: Frenet DP over the whole rest of the section (s = -x, d = y), stations every DS,
lateral samples every DD through the car's y. State (d_prev, d_cur) so the second
difference (curvature) is limited and costed. Edge cost: clearance below car_r + pad,
lateral slope, unseen cells, the wrong side of the island / far from the sign's lane.
A footprint = three circles (radius car_r) along the body.
"""
import math
import time

import cv2
import numpy as np
from scipy import ndimage

# ------------------------------------------------------------------ track (DXF, m)
ENTRY_X = 8.632                 # white line at the section entry (lane 2 width)
EXIT_X = 1.96                   # single exit road starts (x below this)
ISL_X0 = 2.53                   # island left tip x
ISL_TIP = 4.26                  # island right tip x (y 0.94)
LANE_Y = {1: 0.42, 2: 0.94, 3: 1.46}
MID_Y = 0.94
ROAD = np.array([
    (9.10, 0.68), (8.56, 0.68), (8.04, 0.16), (2.49, 0.16), (1.96, 0.68), (1.40, 0.68),
    (1.40, 1.20), (1.96, 1.20), (2.49, 1.72), (8.04, 1.72), (8.56, 1.20), (9.10, 1.20)])
ISLAND = np.array([(2.53, 0.94), (2.78, 0.68), (4.01, 0.68), (4.26, 0.94),
                   (4.01, 1.20), (2.78, 1.20)])
X0, Y0, X1, Y1 = 1.40, 0.05, 9.10, 1.85     # grid window


class Road3Map:

    def __init__(self, res=0.025, hit_min=2, car_r=0.13, circ_off=0.09, hit_cap=6,
                 clear_margin=0.06, static_extra=0.0, nose_len=0.0, nose_half=0.05,
                 isl_shift=0.0, isl_shift_x=4.75):
        self.res = res
        self.static_extra = static_extra  # m more margin to the yellow lines than to objects
        self.hit_cap = hit_cap            # hits saturate here, so a wrong cell clears quickly
        self.clear_margin = clear_margin  # m before a ray's end that it does not clear
        self.nx = int(round((X1 - X0) / res))
        self.ny = int(round((Y1 - Y0) / res))
        self.hit_min = hit_min
        self.car_r = car_r
        self.circ_off = circ_off
        # assumed drawing error from the island on: everything with x < isl_shift_x (island,
        # nose, A / B roads, exit) lies isl_shift m further +y than drawn (2026-11-20 test)
        self.isl_shift, self.isl_shift_x = isl_shift, isl_shift_x
        xs = X0 + (np.arange(self.nx) + 0.5) * res
        ys = Y0 + (np.arange(self.ny) + 0.5) * res
        img = self._static(0.0, nose_len, nose_half)
        if isl_shift != 0.0:
            cols = xs < isl_shift_x
            img[:, cols] = self._static(isl_shift, nose_len, nose_half)[:, cols]
        self.free0 = img.astype(bool)
        self.cx, self.cy = np.meshgrid(xs, ys)
        self.reset()

    def _static(self, dy, nose_len, nose_half):
        """the drawing (road inside the yellow lines, minus island and nose), moved dy in y."""
        off = np.array([0.0, dy])
        img = np.zeros((self.ny, self.nx), np.uint8)
        cv2.fillPoly(img, [self._px(ROAD + off)], 1)
        cv2.fillPoly(img, [self._px(ISLAND + off)], 0)
        cv2.polylines(img, [self._px(ISLAND + off)], True, 0, 1)
        if nose_len > 0:
            # no-go wedge in front of the island tip: the roads part there, nobody drives it;
            # keeps the car off the tip head-on (sign late -> the tip lines were followed)
            nose = np.array([(ISL_TIP - 0.05, MID_Y - nose_half), (ISL_TIP + nose_len, MID_Y),
                             (ISL_TIP - 0.05, MID_Y + nose_half)]) + off
            cv2.fillPoly(img, [self._px(nose)], 0)
        return img

    def yoff(self, x):
        """the drawing's y offset at map x (isl_shift beyond isl_shift_x, else 0)."""
        return np.where(np.asarray(x) < self.isl_shift_x, self.isl_shift, 0.0)

    def _px(self, poly):
        return np.round((poly - (X0, Y0)) / self.res).astype(np.int32)

    def reset(self):
        self.hits = np.zeros((self.ny, self.nx), np.int16)
        self.seen = np.zeros((self.ny, self.nx), bool)
        self.side = None
        self.update()

    def cell(self, x, y):
        return (np.floor((np.asarray(x) - X0) / self.res).astype(int),
                np.floor((np.asarray(y) - Y0) / self.res).astype(int))

    def _inside(self, cx, cy):
        return (cx >= 0) & (cx < self.nx) & (cy >= 0) & (cy < self.ny)

    def add_hits(self, x, y):
        """lidar hits in the map frame; points outside the road are dropped."""
        cx, cy = self.cell(x, y)
        ok = self._inside(cx, cy)
        cx, cy = cx[ok], cy[ok]
        ok = self.free0[cy, cx]
        cells = np.unique(cy[ok] * self.nx + cx[ok])      # one hit per cell and scan
        h = self.hits.reshape(-1)
        h[cells] = np.minimum(h[cells] + 1, self.hit_cap)

    def add_seen(self, ox, oy, ex, ey, step=0.03):
        """rays from the laser (ox, oy) to their ends (hit or max range): cells seen, and
        cleared: a ray that passes through a cell and ends further away says the cell is
        empty -> its hits go down by one. The objects do not move, so this only removes
        lidar noise around them (they grew a cell per side over time, 2026-11-18)."""
        ex, ey = np.asarray(ex), np.asarray(ey)
        if not len(ex):
            return
        L = np.hypot(ex - ox, ey - oy)
        n = max(int(np.max(L) / step), 1)
        f = np.linspace(0.0, 1.0, n, endpoint=False)
        px = ox + (ex - ox)[:, None] * f
        py = oy + (ey - oy)[:, None] * f
        cx, cy = self.cell(px, py)
        ok = self._inside(cx, cy)
        self.seen[cy[ok], cx[ok]] = True
        through = ok & (f[None, :] * L[:, None] < L[:, None] - self.clear_margin)
        cells = np.unique(cy[through] * self.nx + cx[through])
        h = self.hits.reshape(-1)
        h[cells] = np.maximum(h[cells] - 1, 0)

    def shift_x(self, dx):
        """the map frame moved by dx along x (entry line correction): move what was seen."""
        k = int(round(dx / self.res))
        if k == 0:
            return
        for a in (self.hits, self.seen):
            b = np.zeros_like(a)
            if k > 0:
                b[:, k:] = a[:, :-k]
            else:
                b[:, :k] = a[:, -k:]
            a[...] = b

    def shift_y(self, dy):
        """the map frame moved by dy along y (lane centre correction): move what was seen."""
        k = int(round(dy / self.res))
        if k == 0:
            return
        for a in (self.hits, self.seen):
            b = np.zeros_like(a)
            if k > 0:
                b[k:, :] = a[:-k, :]
            else:
                b[:k, :] = a[-k:, :]
            a[...] = b

    def rotate_about(self, px, py, dth):
        """the map frame turned by dth about (px, py): turn what was seen with it."""
        ci = (px - X0) / self.res - 0.5          # grid coords of the pivot (cell centres)
        cj = (py - Y0) / self.res - 0.5
        # rows grow with y, so a world turn of +dth is -dth in cv2's (y down) convention
        M = cv2.getRotationMatrix2D((float(ci), float(cj)), -math.degrees(dth), 1.0)
        size = (self.nx, self.ny)
        self.hits = cv2.warpAffine(self.hits.astype(np.float32), M, size,
                                   flags=cv2.INTER_NEAREST).astype(np.int16)
        self.seen = cv2.warpAffine(self.seen.astype(np.uint8), M, size,
                                   flags=cv2.INTER_NEAREST).astype(bool)

    def occupied(self):
        return self.hits >= self.hit_min

    def update(self):
        """clearance = min(distance to the yellow lines / island / side wall - static_extra,
        distance to the obstacle cells): the lines keep more margin than the objects (a
        smaller footprint let a body corner touch the exit taper, sim 2026-11-18)."""
        static = self.free0.copy()
        if self.side is not None:           # the other side of the island = wall
            mid = MID_Y + self.yoff(self.cx)
            wrong = (self.cx > 2.5) & (self.cx < ISL_TIP + 0.05) & \
                ((self.cy < mid) if self.side == 3 else (self.cy > mid))
            static &= ~wrong
        occ = self.occupied()
        self.free = static & ~occ
        half = self.res / 2
        c_static = ndimage.distance_transform_edt(static) * self.res - half - self.static_extra
        c_obs = ndimage.distance_transform_edt(~occ) * self.res - half if occ.any() else \
            np.full(occ.shape, 9.0)
        self.clear = np.minimum(c_static, c_obs)
        self.clear_obs = c_obs

    def clearance(self, x, y, obs_only=False):
        x, y = np.asarray(x), np.asarray(y)
        cx, cy = self.cell(x, y)
        g = self.clear_obs if obs_only else self.clear
        c = g[np.clip(cy, 0, self.ny - 1), np.clip(cx, 0, self.nx - 1)]
        return np.where((x < X0) | (x >= X1), 0.30, c)   # beyond the window: fine

    def pose_clear(self, x, y, th, obs_only=False):
        """min clearance of the three footprint circles' centres (obs_only: objects only)."""
        c, s = np.cos(th), np.sin(th)
        o, k = self.circ_off, obs_only
        return np.minimum(np.minimum(self.clearance(x + o * c, y + o * s, k),
                                     self.clearance(x - o * c, y - o * s, k)),
                          self.clearance(x, y, k))


class FrenetDP:

    def __init__(self, ds=0.15, dd=0.05, d_min=0.18, d_max=1.70, k_hard=3.0, pad=0.04,
                 w_clear=4.0, w_k=0.02, w_slope=0.3, w_side=2.0, w_centre=0.3, w_unk=3.0,
                 side_ahead=1.8, x_end=EXIT_X - 0.25, w_hedge=0.0, hedge_ahead=2.5,
                 island_bias=0.0, w_centre_island=None, pad_island=None):
        self.ds, self.dd = ds, dd
        self.d_min, self.d_max = d_min, d_max
        self.k_hard, self.pad = k_hard, pad
        self.w_clear, self.w_k, self.w_slope = w_clear, w_k, w_slope
        self.w_side, self.w_centre, self.w_unk = w_side, w_centre, w_unk
        self.side_ahead = side_ahead
        self.w_hedge, self.hedge_ahead = w_hedge, hedge_ahead
        self.hedge_end = ISL_TIP - 0.1     # m (map x): the middle pull ends here
        self.island_bias = island_bias
        self.escape_n = 3                  # stations (x 0.15 m) without the line check, escaping
        self.pad_island = pad if pad_island is None else pad_island
        self.w_centre_island = w_centre if w_centre_island is None else w_centre_island
        self.x_end = x_end

    def plan(self, m, start, r_min=None):
        """start (x, y, th) in the map frame -> (path [N, 2] or None, info dict)."""
        t0 = time.time()
        r_min = m.car_r if r_min is None else r_min
        DS, DD = self.ds, self.dd
        x0, y0, th0 = start
        # escape: the map puts the car on / over a yellow line (odometry error, 2026-11-19 the
        # pose ended 0.2 m outside and nothing was ever planned): start from the nearest lateral
        # grid point inside and check only objects (not the lines) for the first escape_n
        # stations, so the path leads back into the road
        # lateral range, wider by the drawing shift, inside the map window
        dmin = max(self.d_min + min(0.0, m.isl_shift), Y0 + 0.1)
        dmax = min(self.d_max + max(0.0, m.isl_shift), Y1 - 0.1)
        y0c = float(np.clip(y0, dmin, dmax))
        escape = y0c != y0
        if not escape:
            c_all = float(m.pose_clear(x0, y0, th0))
            c_obj = float(m.pose_clear(x0, y0, th0, obs_only=True))
            escape = c_all < r_min <= c_obj
        y0 = y0c
        slope0 = math.tan(math.pi - th0) if math.cos(th0) < -0.3 else 0.0   # dd/ds now
        slope0 = float(np.clip(slope0, -2.0, 2.0))
        if escape:
            slope0 = 0.0       # pointing out of the road: the path may turn in right away
        lo = y0 - math.floor((y0 - dmin) / DD) * DD   # lateral grid through y0
        d = np.arange(lo, dmax + 1e-6, DD)
        J = len(d)
        j0 = int(round((y0 - lo) / DD))
        n = int((x0 - self.x_end) / DS) + 1
        if n < 2 or not 0 <= j0 < J:
            return None, {'ms': 0.0, 'why': 'outside'}
        xs = x0 - DS * np.arange(n + 1)
        a = np.linspace(0.25, 1.0, 3)
        dd = d[None, :] - d[:, None]                         # [jc, jn]
        phi = math.pi - np.arctan(dd / DS)
        big = 1e9
        pts_y = d[None, :, None] + dd[None] * a[:, None, None]  # [3, jc, jn]
        phis = np.broadcast_to(phi, pts_y.shape)
        edges = []
        for i in range(n):
            px = np.broadcast_to(xs[i] - DS * a[:, None, None], pts_y.shape)
            c = m.pose_clear(px, pts_y, phis)
            pad = self.pad_island if (m.side is not None and ISL_X0 <= xs[i + 1] <= ISL_TIP) \
                else self.pad
            e = self.w_clear * np.clip(m.car_r + pad - c, 0, None).sum(0) \
                + self.w_slope * (dd / DS) ** 2 * DS
            if self.w_unk:
                ux, uy = m.cell(px, pts_y)
                unseen = ~m.seen[np.clip(uy, 0, m.ny - 1), np.clip(ux, 0, m.nx - 1)]
                e += self.w_unk * unseen.mean(0) * DS
            if escape and i < self.escape_n:
                e[m.pose_clear(px, pts_y, phis, obs_only=True).min(0) < r_min] = big
            else:
                e[c.min(0) < r_min] = big
            if m.side is not None and xs[i + 1] < ISL_TIP + self.side_ahead:
                yo = float(m.yoff(xs[i + 1]))
                want = LANE_Y[m.side] + yo
                wc = self.w_centre
                if ISL_X0 <= xs[i + 1] <= ISL_TIP:
                    # beside the island: the wanted line closer to the island (it went out on
                    # the outer side), held harder than the clearance pad pushing to the middle
                    want += self.island_bias if m.side == 1 else -self.island_bias
                    wc = self.w_centre_island
                wrong = (d < MID_Y + yo) if m.side == 3 else (d > MID_Y + yo)
                e += (self.w_side * wrong + wc * np.abs(d - want))[None, :] * DS
            elif m.side is None and self.w_hedge > 0 and \
                    self.hedge_end < xs[i + 1] < ISL_TIP + self.hedge_ahead:
                # no sign yet near the island: keep to the middle (lane 2), both roads one
                # lane away
                e += (self.w_hedge * np.abs(d - MID_Y - float(m.yoff(xs[i + 1]))))[None, :] * DS
            edges.append(e)
        smax = max(int(self.k_hard * DS ** 2 / DD + 1e-6), 1)
        steps = np.arange(-smax, smax + 1)
        kk = (steps * DD / DS ** 2) ** 2 * self.w_k * DS
        dprev = y0 - DS * slope0
        sec = (d - 2 * y0 + dprev) / DS ** 2
        C = np.full((J, J), big)
        C[j0, :] = edges[0][j0] + np.where(np.abs(sec) <= self.k_hard + 1e-6,
                                           self.w_k * sec ** 2 * DS, big)
        jj = np.arange(J)
        back = []
        # the furthest station any state reaches: when nothing gets to the exit (objects far
        # ahead seen from one side only, 2026-11-18), drive there and plan again on the way
        last = (0, C) if (C < big / 2).any() else None
        for i in range(1, n):
            new = np.full((J, J), big)
            bp = np.zeros((J, J), int)
            for s_, kc in zip(steps, kk):
                jp = s_ + 2 * jj[:, None] - jj[None, :]      # jn - 2 jc + jp = s_
                valid = (jp >= 0) & (jp < J)
                cand = np.where(valid, C[np.clip(jp, 0, J - 1), jj[:, None]], big) + kc
                better = cand < new
                new = np.where(better, cand, new)
                bp = np.where(better, jp, bp)
            back.append(bp)
            C = np.minimum(new + edges[i], big)
            if not (C < big / 2).any():
                break
            last = (i, C)
        info = {'ms': (time.time() - t0) * 1e3}
        if last is None:
            info['why'] = 'blocked'
            return None, info
        k, Ck = last
        jc, jn = np.unravel_index(np.argmin(Ck), Ck.shape)
        info['cost'] = float(Ck[jc, jn])
        info['partial'] = k < n - 1
        seq = [jn, jc]
        for bp in reversed(back[:k]):
            seq.append(bp[seq[-1], seq[-2]])
        seq = seq[::-1]                                      # stations 0 .. k + 1
        path = np.stack([xs[:len(seq)], d[seq]], 1)
        info['reach_x'] = float(path[-1, 0])
        return path, info


def densify(path, step=0.05):
    out = [path[0, :2]]
    for a, b in zip(path[:-1, :2], path[1:, :2]):
        n = max(int(np.hypot(*(b - a)) / step), 1)
        for t in np.linspace(1.0 / n, 1.0, n):
            out.append(a + t * (b - a))
    return np.array(out)


def pure_pursuit(pose, path, ld):
    """curvature (1/m, + left) to the path point ld ahead, and the distance to the path."""
    x, y, th = pose
    d = np.hypot(path[:, 0] - x, path[:, 1] - y)
    i0 = int(np.argmin(d))
    j = i0
    while j < len(path) - 1 and np.hypot(path[j, 0] - x, path[j, 1] - y) < ld:
        j += 1
    tx, ty = path[j]
    a = math.atan2(ty - y, tx - x) - th
    a = math.atan2(math.sin(a), math.cos(a))
    L = max(math.hypot(tx - x, ty - y), 0.1)
    return 2.0 * math.sin(a) / L, float(d[i0])
