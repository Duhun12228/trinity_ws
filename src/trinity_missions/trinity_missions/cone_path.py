"""
ji2 (2026-11-17): cone corridor by a planned path (drive_mode 'path' of cone_node).

Instead of reacting to the cones seen in one scan, the cones are kept in a map (odom), the
corridor centre line is traced through the map, and a short path is planned and tracked:

  map       every scan: lidar clusters (centre = seen surface + cone_surface_r away from the
            lidar) into a cone map in odom (merged within map_merge). The cones beside and
            behind the car (outside the lidar's +-60 deg) stay known.
  centre    marched from the car in corr_step steps: at each station the nearest cone on the
            left and on the right (window corr_back .. corr_ahead along, |lateral| <
            corr_lat_max) -> their middle; one side only -> corr_width / 2 inside it.
            Stops where no cone is near any more (corridor end) or at corr_len.
  plan      candidate paths from the car: curvature k1 for plan_l1, then k2 for plan_l2
            (plan_n_k values each in +-plan_k_max). The car body (front_x / car_rear /
            car_half_width) swept along each must keep plan_gap_min from every cone base
            (cone_body_r). Among those: least mean distance to the centre line + end heading
            error + change of k1 from the last plan. None clear: the one with the biggest gap.
  track     pure pursuit on the chosen path (odom) at the timer rate, omega = track_gain x v x
            curvature (the car turns ~0.5 x the command, bag 2026-11-17).
  speed     v_max when the gap >= gap_fast and |k| <= k_fast, down to v_min.
  guard     a lidar point within guard_x ahead of the bumper (car width + 3 cm): stop; still
            there after guard_wait_sec: back off straight up to guard_back m if the map shows
            room behind, then plan again.
  exit      no cone ahead (centre line shorter than exit_corr_min) and no camera cone: pure
            pursuit onto the lane centre from /lane_follow/yaw (carried on odom for
            exit_lane_mem s), DONE when centred (exit_ok_cy) and parallel (exit_ok_yaw) for
            exit_ok_sec; exit_align_max s without that, or no lane at all for
            exit_no_lane_sec: DONE anyway (lane follow).
"""
import math

import numpy as np

PATH, GUARD, BACK, ALIGN, SEEK = 'path', 'guard', 'back', 'align', 'seek'


def declare_path_params(d):
    d('drive_mode', 'reactive')      # 'path': this module, else the reactive corridor
    d('odom_topic', '/odometry/filtered')
    d('odom_timeout', 0.3)
    d('yaw_topic', '/lane_follow/yaw')
    d('map_range', 1.6)
    d('map_merge', 0.08)
    d('map_min_hits', 2)
    d('cone_surface_r', 0.04)
    d('cone_body_r', 0.08)
    d('car_rear', 0.16)
    d('car_half', 0.11)
    d('corr_width', 0.60)
    d('corr_step', 0.10)
    d('corr_len', 1.5)
    d('corr_back', 0.25)
    d('corr_ahead', 0.35)
    d('corr_lat_max', 0.60)
    d('centre_offset', 0.0)          # m, followed line = corridor middle + this (+ left, - right)
    d('corr_gap_min', 0.35)          # m, lateral gap between rows taken as the corridor
    d('corr_gap_max', 0.95)
    d('plan_k_max', 1.8)
    d('plan_n_k', 13)
    d('plan_l1', 0.30)
    d('plan_l2', 0.50)
    d('plan_gap_min', 0.03)
    d('plan_w_centre', 1.0)
    d('plan_w_head', 0.3)
    d('plan_w_smooth', 0.03)
    d('plan_w_gap', 4.0)             # cost per m of gap below plan_gap_want
    d('plan_gap_want', 0.10)         # m, the cone base corners stick out, tracking is +-3 cm
    d('back_max_sec', 3.0)           # s, a backoff ends after this anyway
    d('tight_backs_max', 2)          # in the corridor, no clear path: short backoffs (then creep)
    d('track_lookahead', 0.20)
    d('track_gain', 2.3)
    d('v_max', 0.30)
    d('v_min', 0.25)
    d('gap_fast', 0.08)
    d('k_fast', 1.0)
    d('guard_x', 0.05)
    d('guard_wait_sec', 0.5)
    d('guard_back', 0.15)
    d('rear_clear', 0.10)
    d('way_in_off', 0.20)            # m, chosen path this far (mean) from the centre line: no way in
    d('way_back', 0.25)              # m backed off then
    d('backs_max', 3)
    d('exit_corr_min', 0.20)
    d('exit_speed', 0.20)
    d('exit_lookahead', 0.20)
    d('exit_gain', 2.0)
    d('cy_ref_x', 0.43)
    d('exit_ok_cy', 0.12)
    d('exit_ok_yaw', 20.0)
    d('exit_ok_sec', 0.2)
    d('exit_align_max', 8.0)
    d('exit_lane_mem', 4.0)
    d('exit_no_lane_sec', 2.0)


class ConePathMixin:
    """Methods for cone_node (self is the ConeNode)."""

    def init_path(self, p):
        self.drive_mode = str(p('drive_mode'))
        g = lambda n: float(p(n))  # noqa: E731
        self.odom_timeout = g('odom_timeout')
        self.map_range, self.map_merge = g('map_range'), g('map_merge')
        self.map_min_hits = int(p('map_min_hits'))
        self.cone_surface_r, self.cone_body_r = g('cone_surface_r'), g('cone_body_r')
        self.car_rear, self.car_half = g('car_rear'), g('car_half')
        self.corr_width, self.corr_step, self.corr_len = g('corr_width'), g('corr_step'), g('corr_len')
        self.corr_back, self.corr_ahead, self.corr_lat = g('corr_back'), g('corr_ahead'), g('corr_lat_max')
        self.corr_gap_min, self.corr_gap_max = g('corr_gap_min'), g('corr_gap_max')
        self.centre_offset = g('centre_offset')
        self.plan_k_max, self.plan_n_k = g('plan_k_max'), int(p('plan_n_k'))
        self.plan_l1, self.plan_l2, self.plan_gap_min = g('plan_l1'), g('plan_l2'), g('plan_gap_min')
        self.w_centre, self.w_head, self.w_smooth = g('plan_w_centre'), g('plan_w_head'), g('plan_w_smooth')
        self.w_gap, self.gap_want = g('plan_w_gap'), g('plan_gap_want')
        self.back_max_sec, self.tight_backs_max = g('back_max_sec'), int(p('tight_backs_max'))
        self.track_ld, self.track_gain = g('track_lookahead'), g('track_gain')
        self.v_max, self.v_min, self.gap_fast, self.k_fast = g('v_max'), g('v_min'), g('gap_fast'), g('k_fast')
        self.guard_x, self.guard_wait, self.guard_back = g('guard_x'), g('guard_wait_sec'), g('guard_back')
        self.rear_clear = g('rear_clear')
        self.way_in_off, self.way_back, self.backs_max = g('way_in_off'), g('way_back'), int(p('backs_max'))
        self.exit_corr_min, self.exit_speed = g('exit_corr_min'), g('exit_speed')
        self.exit_ld, self.exit_gain, self.cy_ref_x = g('exit_lookahead'), g('exit_gain'), g('cy_ref_x')
        self.exit_ok_cy, self.exit_ok_yaw, self.exit_ok_sec = g('exit_ok_cy'), g('exit_ok_yaw'), g('exit_ok_sec')
        self.exit_align_max, self.exit_lane_mem = g('exit_align_max'), g('exit_lane_mem')
        self.exit_no_lane = g('exit_no_lane_sec')
        self.front_x = float(p('front_x'))
        self.w_cap = math.tan(self.max_steer) / self.wheelbase      # x v = omega limit
        self.odom = self.odom_at = None
        self.lane_yaw = self.lane_yaw_at = None
        self.lane_line = self.lane_line_at = None
        # candidate curvature pairs, computed once (car frame, from (0, 0, 0))
        ks = np.linspace(-self.plan_k_max, self.plan_k_max, self.plan_n_k)
        ds = 0.02
        n1, n2 = int(round(self.plan_l1 / ds)), int(round(self.plan_l2 / ds))
        K1, K2 = np.meshgrid(ks, ks, indexing='ij')
        K1, K2 = K1.ravel(), K2.ravel()
        kk = np.concatenate([np.repeat(K1[:, None], n1, 1), np.repeat(K2[:, None], n2, 1)], 1)
        th = np.cumsum(kk * ds, 1) - kk * ds / 2.0
        self.cand_x = np.concatenate([np.zeros((len(K1), 1)), np.cumsum(np.cos(th) * ds, 1)], 1)
        self.cand_y = np.concatenate([np.zeros((len(K1), 1)), np.cumsum(np.sin(th) * ds, 1)], 1)
        self.cand_th = np.concatenate([np.zeros((len(K1), 1)), np.cumsum(kk * ds, 1)], 1)
        self.cand_k1, self.cand_k2 = K1, K2
        self.reset_path()

    def reset_path(self):
        self.cmap = []                   # [[x, y, hits]] cone centres in odom
        self.cpath = None                # dict: pts (odom), v, gap, k1
        self.centre = np.zeros((0, 2))   # car frame, debug
        self.last_k1 = 0.0
        self.guard_since = None
        self.back_from = None
        self.backs = 0                   # 'no way in' backoffs this run
        self.tight_backs = 0             # in-corridor 'none clear' backoffs this run
        self.back_at = None
        self.align_since = self.align_ok_since = None
        self.no_ahead_since = None
        self.corr_line = None            # (x, y, direction) of the corridor end in odom

    # ------------------------------------------------------------------ inputs
    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.odom = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)
        self.odom_at = self.now()

    def odom_fresh(self, now):
        return self.odom_at is not None and now - self.odom_at <= self.odom_timeout

    def yaw_callback(self, msg):
        if len(msg.data) < 3 or not all(math.isfinite(v) for v in msg.data[:3]):
            return
        now = self.now()
        self.lane_yaw, self.lane_yaw_at = (float(msg.data[0]), float(msg.data[2])), now
        if self.odom_fresh(now) and abs(self.lane_yaw[0]) <= 45.0:
            X, Y, H = self.odom
            psi = math.radians(self.lane_yaw[0])
            px, py = self.cy_ref_x, self.lane_yaw[1]
            self.lane_line = (X + px * math.cos(H) - py * math.sin(H),
                              Y + px * math.sin(H) + py * math.cos(H), H + psi)
            self.lane_line_at = now

    # -------------------------------------------------------------- cone map
    def map_add(self, now, x, y, ok):
        """Clusters up to map_range into the odom cone map."""
        if not self.odom_fresh(now):
            return
        roi = ok & (x > 0.0) & (x < self.map_range) & (np.abs(y) < 1.0)
        idx, cur, groups = np.flatnonzero(roi), [], []
        prev = None
        for i in idx:
            if prev is not None and (i != prev + 1 or
                                     math.hypot(x[i] - x[prev], y[i] - y[prev]) > self.cluster_gap):
                groups.append(cur)
                cur = []
            cur.append(i)
            prev = i
        if cur:
            groups.append(cur)
        X, Y, H = self.odom
        c, s = math.cos(H), math.sin(H)
        lx, ly = self.laser[0], self.laser[1]
        for g in groups:
            if len(g) < self.cluster_min_points:
                continue
            gx, gy = x[g], y[g]
            if math.hypot(gx.max() - gx.min(), gy.max() - gy.min()) > self.cluster_max_size:
                continue
            mx, my = float(gx.mean()), float(gy.mean())
            r = math.hypot(mx - lx, my - ly)
            if r > 1e-3:                        # the lidar sees the near surface
                mx += self.cone_surface_r * (mx - lx) / r
                my += self.cone_surface_r * (my - ly) / r
            ox, oy = X + c * mx - s * my, Y + s * mx + c * my
            best, bd = None, self.map_merge
            for m in self.cmap:
                dd = math.hypot(m[0] - ox, m[1] - oy)
                if dd < bd:
                    best, bd = m, dd
            if best is None:
                self.cmap.append([ox, oy, 1])
            else:
                k = min(best[2], 8)
                best[0] = (best[0] * k + ox) / (k + 1)
                best[1] = (best[1] * k + oy) / (k + 1)
                best[2] += 1
        if len(self.cmap) > 400:
            self.cmap.sort(key=lambda m: -m[2])
            del self.cmap[400:]

    def cones_car(self, now):
        """Map cones (seen >= map_min_hits times) in the car frame near the car."""
        if not self.cmap or not self.odom_fresh(now):
            return np.zeros((0, 2))
        arr = np.array([m[:2] for m in self.cmap if m[2] >= self.map_min_hits])
        if not len(arr):
            return np.zeros((0, 2))
        X, Y, H = self.odom
        dx, dy = arr[:, 0] - X, arr[:, 1] - Y
        cx = math.cos(H) * dx + math.sin(H) * dy
        cy = -math.sin(H) * dx + math.cos(H) * dy
        sel = (cx > -0.8) & (cx < self.map_range + 0.3) & (np.abs(cy) < 1.2)
        return np.stack([cx[sel], cy[sel]], 1)

    # ------------------------------------------------------------ centre line
    def centre_line(self, C):
        """Corridor centre (car frame) marched from the car, (N, 2); empty if no cone near."""
        if not len(C):
            return np.zeros((0, 2))
        p, h = np.zeros(2), 0.0
        pts, miss = [], 0
        for i in range(int(self.corr_len / self.corr_step) + 1):
            if i:
                p = p + self.corr_step * np.array([math.cos(h), math.sin(h)])
            d = np.array([math.cos(h), math.sin(h)])
            n = np.array([-math.sin(h), math.cos(h)])
            rel = C - p
            u, v = rel @ d, rel @ n
            win = (u > -self.corr_back) & (u < self.corr_ahead) & (np.abs(v) < self.corr_lat)
            L, R = v[win & (v > 0)], v[win & (v < 0)]
            vs = np.sort(v[win])
            gaps = [(vs[k] + vs[k + 1]) / 2.0 for k in range(len(vs) - 1)
                    if self.corr_gap_min <= vs[k + 1] - vs[k] <= self.corr_gap_max]
            if gaps:
                # the corridor = a gap of about its width between two rows (also when the car
                # is beside it, both rows on one side): the one nearest the line so far
                off = min(gaps, key=abs)
            elif len(L) and len(R):
                off = (L.min() + R.max()) / 2.0
            elif len(L):
                off = L.min() - self.corr_width / 2.0
            elif len(R):
                off = R.max() + self.corr_width / 2.0
            else:
                if not pts:
                    continue            # the corridor starts further ahead: march on to it
                miss += 1
                if miss >= 2:
                    break
                pts.append(p.copy())
                continue
            miss = 0
            off += self.centre_offset           # + left / - right of the middle
            q = p + off * n
            if pts:
                step = q - pts[-1]
                hn = math.atan2(step[1], step[0])
                dh = math.atan2(math.sin(hn - h), math.cos(hn - h))
                lim = 2.0 * self.corr_step         # curvature <= 2 1/m along the line
                h += max(-lim, min(lim, dh))
            p = q
            pts.append(p.copy())
        return np.array(pts) if pts else np.zeros((0, 2))

    # ------------------------------------------------------------------ plan
    def plan(self, now, C):
        """Choose a candidate path; store it in odom. Returns the gap (m) or None."""
        cl = self.centre_line(C)
        self.centre = cl
        X, Y, H = self.odom
        if len(cl) >= 5:                 # the corridor direction at its far end, kept in odom
            a = H + math.atan2(cl[-1, 1] - cl[-3, 1], cl[-1, 0] - cl[-3, 0])
            self.corr_line = (X + math.cos(H) * cl[-1, 0] - math.sin(H) * cl[-1, 1],
                              Y + math.sin(H) * cl[-1, 0] + math.cos(H) * cl[-1, 1], a)
        xs, ys, ths = self.cand_x, self.cand_y, self.cand_th
        # body gap to every cone base, every 2nd pose
        if len(C):
            px, py, pt = xs[:, ::2], ys[:, ::2], ths[:, ::2]
            cs, sn = np.cos(pt)[..., None], np.sin(pt)[..., None]
            dx = C[None, None, :, 0] - px[..., None]
            dy = C[None, None, :, 1] - py[..., None]
            u = cs * dx + sn * dy
            w = -sn * dx + cs * dy
            ex = np.maximum(np.maximum(u - self.front_x, -self.car_rear - u), 0.0)
            ey = np.maximum(np.abs(w) - self.car_half, 0.0)
            inside = (ex == 0.0) & (ey == 0.0)
            dist = np.where(inside, -np.minimum(np.minimum(self.front_x - u, u + self.car_rear),
                                                self.car_half - np.abs(w)), np.hypot(ex, ey))
            gap = dist.min(axis=(1, 2)) - self.cone_body_r
        else:
            gap = np.full(len(xs), 1.0)
        # centre-line cost
        follow_end = len(cl) < 5 and self.corr_line is not None
        end_off = np.zeros(len(xs))
        if follow_end:
            # the corridor ends (only cones beside / behind): along its last direction
            lx, ly, a = self.corr_line
            ca, sa = math.cos(a - H), math.sin(a - H)
            dx, dy = lx - X, ly - Y
            ox, oy = math.cos(H) * dx + math.sin(H) * dy, -math.sin(H) * dx + math.cos(H) * dy
            lat = -(xs[:, ::5] - ox) * sa + (ys[:, ::5] - oy) * ca
            cost_c = np.abs(lat).mean(axis=1)
            dh = ths[:, -1] - math.atan2(sa, ca)
            cost_h = np.abs(np.arctan2(np.sin(dh), np.cos(dh)))
        elif len(cl) >= 2:
            # the centre line extended back from its start along its first direction: before
            # the corridor the path is judged against its approach line, not the far start
            d0 = cl[1] - cl[0]
            d0 = d0 / max(float(np.hypot(*d0)), 1e-6)
            back = cl[0][None, :] - np.arange(10, 0, -1)[:, None] * self.corr_step * d0[None, :]
            cl = np.concatenate([back, cl])
            sx, sy = xs[:, ::5], ys[:, ::5]
            dd = np.hypot(sx[..., None] - cl[None, None, :, 0], sy[..., None] - cl[None, None, :, 1])
            j = dd.argmin(axis=2)
            cost_c = dd.min(axis=2).mean(axis=1)
            end_off = dd.min(axis=2)[:, -1]
            seg = np.diff(cl, axis=0)
            hcl = np.arctan2(seg[:, 1], seg[:, 0])
            hcl = np.concatenate([hcl, hcl[-1:]])
            dh = ths[:, -1] - hcl[j[:, -1]]
            cost_h = np.abs(np.arctan2(np.sin(dh), np.cos(dh)))
        else:
            cost_c = 0.2 * np.abs(ys[:, -1])
            cost_h = np.abs(ths[:, -1])
        cost = self.w_centre * cost_c + self.w_head * cost_h + \
            self.w_smooth * np.abs(self.cand_k1 - self.last_k1) + \
            self.w_gap * np.maximum(0.0, self.gap_want - gap)    # rather away from the cones
        ok = gap >= self.plan_gap_min
        i = int(np.argmin(np.where(ok, cost, np.inf))) if ok.any() else int(np.argmax(gap))
        self.last_k1 = float(self.cand_k1[i])
        kmax = max(abs(self.cand_k1[i]), abs(self.cand_k2[i]))
        g = float(gap[i])
        a = min(1.0, max(0.0, (g - self.plan_gap_min) / max(self.gap_fast - self.plan_gap_min, 1e-3)))
        b = min(1.0, max(0.0, (self.plan_k_max - kmax) / max(self.plan_k_max - self.k_fast, 1e-3)))
        v = self.v_min + (self.v_max - self.v_min) * min(a, b)
        c, s = math.cos(H), math.sin(H)
        pts = np.stack([X + c * xs[i] - s * ys[i], Y + s * xs[i] + c * ys[i]], 1)
        self.cpath = dict(pts=pts, v=v, gap=g, k1=float(self.cand_k1[i]), k2=float(self.cand_k2[i]),
                          clear=bool(ok[i]), idx=0, at=now, n_cl=len(cl),
                          off=float(cost_c[i]) if len(cl) >= 2 and not follow_end else 0.0)
        return g

    def track(self, now):
        """Pure pursuit on the stored path; (v, omega)."""
        pa = self.cpath
        X, Y, H = self.odom
        pts = pa['pts']
        i0 = pa['idx']
        win = pts[i0:min(len(pts), i0 + 30)]
        i = i0 + int(np.argmin(np.hypot(win[:, 0] - X, win[:, 1] - Y)))
        pa['idx'] = i
        seg = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
        j = min(int(np.searchsorted(seg, seg[i] + self.track_ld)), len(pts) - 1)
        dx, dy = pts[j, 0] - X, pts[j, 1] - Y
        lx = math.cos(H) * dx + math.sin(H) * dy
        ly = -math.sin(H) * dx + math.cos(H) * dy
        k = 2.0 * ly / max(lx * lx + ly * ly, 1e-4)
        v = pa['v']
        w = self.track_gain * v * k
        return v, max(-v * self.w_cap, min(v * self.w_cap, w))

    # ------------------------------------------------------------------ lane
    def lane_known(self, now):
        if self.lane_yaw_at is not None and now - self.lane_yaw_at <= self.input_timeout:
            return True
        return self.lane_line_at is not None and now - self.lane_line_at <= self.exit_lane_mem \
            and self.odom_fresh(now)

    def line_car(self, line):
        """(heading deg, y at x = cy_ref_x) of an odom line (x, y, direction), car frame."""
        X, Y, H = self.odom
        lx, ly, a = line
        psi = math.atan2(math.sin(a - H), math.cos(a - H))
        dx, dy = lx - X, ly - Y
        px = math.cos(H) * dx + math.sin(H) * dy
        py = -math.sin(H) * dx + math.cos(H) * dy
        if abs(math.cos(psi)) < 0.2:
            return None
        return math.degrees(psi), py + (self.cy_ref_x - px) * math.tan(psi)

    def lane_now(self, now):
        if self.lane_line is not None and self.odom_fresh(now) \
                and now - self.lane_line_at <= self.exit_lane_mem:
            r = self.line_car(self.lane_line)
            if r is not None:
                return r
        return self.lane_yaw

    def lane_pursuit(self, now, line=None):
        yaw, cy = self.lane_now(now) if line is None else line
        psi = math.radians(max(-60.0, min(60.0, yaw)))
        xt = self.cy_ref_x + self.exit_ld * math.cos(psi)
        yt = cy + self.exit_ld * math.sin(psi)
        w = self.exit_gain * self.exit_speed * 2.0 * yt / (xt * xt + yt * yt)
        lim = self.exit_speed * self.w_cap
        return max(-lim, min(lim, w))

    def rear_room(self, now):
        C = self.cones_car(now)
        if not len(C):
            return float('inf')
        sel = (C[:, 0] < -self.car_rear) & (np.abs(C[:, 1]) < self.car_half + self.cone_body_r + 0.02)
        if not sel.any():
            return float('inf')
        return float((-C[sel, 0]).min()) - self.car_rear - self.cone_body_r

    # ------------------------------------------------------------- per scan
    def path_scan(self, now, x, y, ok):
        """Map + plan (per scan, while running)."""
        if not self.odom_fresh(now):
            self.publish(0.0, 0.0)
            self.set_phase(PATH, 'no odom: stop')
            self.cpath = None
            return
        self.map_add(now, x, y, ok)
        if self.phase in (BACK, ALIGN):
            return
        # guard: something right in front of the bumper (this scan)
        P = self.points
        g = (P[:, 0] > 0.0) & (P[:, 0] < self.front_x + self.guard_x) & \
            (np.abs(P[:, 1]) < self.car_half + 0.03) if len(P) else np.zeros(0, bool)
        if len(P) and g.any():
            self.guard_since = self.guard_since or now
            self.cpath = None
            self.set_phase(GUARD, f'object {float(P[g, 0].min() - self.front_x) * 100:.0f} cm '
                                  f'ahead of the bumper: stop')
            return
        self.guard_since = None
        C = self.cones_car(now)
        gap = self.plan(now, C)
        pa = self.cpath
        beside = (np.abs(C[:, 0]) < 0.3) & (np.abs(C[:, 1]) < 0.6) if len(C) else np.zeros(0, bool)
        if not beside.any() and (not pa['clear'] or pa['off'] > self.way_in_off) \
                and self.backs < self.backs_max:
            # before the corridor (no cone beside yet), no clear path into it from here
            # (entering off-centre / at an angle):
            # back off straight a little and plan again, instead of squeezing or going outside
            room = self.rear_room(now) - self.rear_clear
            if room > 0.05:
                self.backs += 1
                self.cpath = None
                self.back_from = (self.odom[0], self.odom[1], min(self.way_back, room))
                self.set_phase(BACK, f'no clear way in (gap {gap * 100:.0f} cm, off the centre '
                                     f'{pa["off"] * 100:.0f} cm): back off '
                                     f'{self.back_from[2] * 100:.0f} cm ({self.backs}/{self.backs_max})')
                return
        if beside.any() and not pa['clear']:
            # in the corridor and no path keeps plan_gap_min: back off a little (room behind
            # checked on the map), at most tight_backs_max times, else creep on the best one
            room = self.rear_room(now) - self.rear_clear
            if self.tight_backs < self.tight_backs_max and room > 0.05:
                self.tight_backs += 1
                self.cpath = None
                self.back_from = (self.odom[0], self.odom[1], min(0.15, room))
                self.set_phase(BACK, f'no clear path (gap {gap * 100:.0f} cm): back off '
                                     f'{self.back_from[2] * 100:.0f} cm ({self.tight_backs}/'
                                     f'{self.tight_backs_max})')
                return
            pa['v'] = min(pa['v'], 0.15)
        ahead = len(self.centre) * self.corr_step
        front = (C[:, 0] > self.front_x + 0.10) & (np.abs(C[:, 1]) < 0.6) if len(C) else np.zeros(0, bool)
        if not front.any() and not self.clusters:     # no cone ahead any more: the exit
            self.no_ahead_since = self.no_ahead_since or now
        else:
            self.no_ahead_since = None
        txt = (f'path k {pa["k1"]:+.1f}/{pa["k2"]:+.1f}, gap {gap * 100:.0f} cm'
               f'{"" if pa["clear"] else " (none clear)"}, centre {ahead:.1f} m')
        self.set_phase(PATH, txt)
        self.reason = txt

    # ------------------------------------------------------------ per timer
    def path_tick(self, now):
        if self.state != self.MS.RUNNING or not self.active:
            return
        if not self.odom_fresh(now):
            self.publish(0.0, 0.0)
            return
        cam_gone = self.cam_cone_at is None or now - self.cam_cone_at > self.input_timeout
        # exit: nothing ahead -> into the lane
        if self.phase == ALIGN or (self.no_ahead_since is not None and cam_gone
                                   and now - self.no_ahead_since >= 0.3):
            self.step_align(now)
            return
        if self.phase == GUARD:
            self.publish(0.0, 0.0)
            if self.guard_since is not None and now - self.guard_since >= self.guard_wait:
                room = self.rear_room(now) - self.rear_clear
                if room > 0.03:
                    self.back_from = (self.odom[0], self.odom[1], min(self.guard_back, room))
                    self.set_phase(BACK, f'still blocked: back off {self.back_from[2] * 100:.0f} cm')
                else:
                    self.guard_since = now       # no room behind: wait, plan again
            return
        if self.phase == BACK:
            X, Y, _ = self.odom
            gone = math.hypot(X - self.back_from[0], Y - self.back_from[1])
            self.back_at = self.back_at or now
            if gone >= self.back_from[2] or self.rear_room(now) < self.rear_clear \
                    or now - self.back_at >= self.back_max_sec:
                self.back_at = None
                self.publish(0.0, 0.0)
                self.guard_since = None
                self.set_phase(PATH, f'backed off {gone * 100:.0f} cm')
            else:
                self.publish(-0.15, 0.0)
            return
        if self.cpath is None or now - self.cpath['at'] > 0.5:
            self.publish(0.0, 0.0)              # no fresh plan (no scan): stop
            return
        v, w = self.track(now)
        self.publish(v, w)

    def step_align(self, now):
        if self.phase != ALIGN:
            self.align_since, self.align_ok_since = now, None
            self.set_phase(ALIGN, 'no cone ahead: into the yellow lane')
        if not self.lane_known(now):
            # lane not seen yet: on along the corridor's end direction, slowly
            r = self.line_car(self.corr_line) if self.corr_line is not None else None
            self.publish(self.exit_speed, 0.0 if r is None else self.lane_pursuit(now, r))
            if now - self.align_since >= self.exit_no_lane:
                self.finish(f'no cone, no lane for {self.exit_no_lane:.0f} s: lane follow')
            return
        yaw, cy = self.lane_now(now)
        off = cy - self.cy_ref_x * math.tan(math.radians(yaw))     # lane centre at the car
        if abs(off) <= self.exit_ok_cy and abs(yaw) <= self.exit_ok_yaw:
            self.align_ok_since = self.align_ok_since or now
            if now - self.align_ok_since >= self.exit_ok_sec:
                self.publish(self.exit_speed, 0.0)
                self.finish(f'no cone, in the lane (centre {off * 100:+.0f} cm, heading {yaw:+.0f} deg)')
                return
        else:
            self.align_ok_since = None
        if now - self.align_since >= self.exit_align_max:
            self.finish(f'no cone, lane alignment {self.exit_align_max:.0f} s: lane follow')
            return
        self.publish(self.exit_speed, self.align_safe(now, self.lane_pursuit(now)))

    def body_gap(self, C, k, length=0.4):
        """Car body gap to the cone bases along an arc of curvature k from the car."""
        if not len(C):
            return 1.0
        s = np.linspace(0.0, length, 9)
        th = k * s
        x = np.where(abs(k) > 1e-6, np.sin(th) / (k if abs(k) > 1e-6 else 1.0), s)
        y = np.where(abs(k) > 1e-6, (1.0 - np.cos(th)) / (k if abs(k) > 1e-6 else 1.0), 0.0)
        cs, sn = np.cos(th)[:, None], np.sin(th)[:, None]
        dx, dy = C[None, :, 0] - x[:, None], C[None, :, 1] - y[:, None]
        u, w = cs * dx + sn * dy, -sn * dx + cs * dy
        ex = np.maximum(np.maximum(u - self.front_x, -self.car_rear - u), 0.0)
        ey = np.maximum(np.abs(w) - self.car_half, 0.0)
        return float(np.hypot(ex, ey).min()) - self.cone_body_r

    def align_safe(self, now, w):
        """The lane alignment turn, made gentler while a cone (beside / behind) would be hit by
        the swinging body (the rear corner swings out when turning)."""
        C = self.cones_car(now)
        v = max(self.exit_speed, 0.05)
        k = w / (self.exit_gain * v)                    # geometric curvature of the command
        for f in (1.0, 0.7, 0.4, 0.0, -0.3):
            if self.body_gap(C, k * f) >= self.plan_gap_min:
                return w * f
        return 0.0
