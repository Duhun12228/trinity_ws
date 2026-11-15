"""
IPM (inverse perspective mapping) to a bird's-eye view, assuming a flat ground plane.

Coordinates
  ground : X forward, Y left, Z up  [m], origin = point on the floor under the camera lens
  camera : optical frame (x right, y down, z forward)
  BEV    : row 0 = x_max (far), bottom row = x_min (near), centre column = Y 0
"""
import math

import cv2
import numpy as np

# LIMO Pro (Orbbec Dabai) colour camera factory calibration, 640x480 (from camera_info)
DEFAULT_INTRINSICS = {
    'fx': 489.4240417480469, 'fy': 489.4240417480469,
    'cx': 319.5390625, 'cy': 210.84768676757812,
    'dist': [0.07326161116361618, -0.0987033098936081, -0.0003452506789471954,
             0.00010883769573410973, 0.0, 0.0, 0.0, 0.0],
    'size': (640, 480),
}

# Mounting geometry and BEV range (cam_height / pitch must be measured on the robot)
DEFAULT_GEOMETRY = {
    'cam_height': 0.15,   # m, floor to lens centre
    'pitch_deg': 10.0,    # tilt down (+ = down)
    'roll_deg': 0.0,
    'yaw_deg': 0.0,       # + = looking left
    'x_min': 0.20,        # m, nearest distance shown in BEV
    'x_max': 1.50,        # m, farthest distance shown in BEV
    'y_half': 0.60,       # m, half width of BEV
    'resolution': 0.005,  # m/px
}

# ground (X fwd, Y left, Z up) -> camera optical (x right, y down, z fwd)
R_GROUND_TO_CAM = np.array([[0.0, -1.0, 0.0],
                            [0.0, 0.0, -1.0],
                            [1.0, 0.0, 0.0]])


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def intrinsics_from_camera_info(msg):
    """sensor_msgs/CameraInfo -> intrinsics dict. None if the message has no calibration."""
    if not any(msg.k):
        return None
    return {'fx': msg.k[0], 'fy': msg.k[4], 'cx': msg.k[2], 'cy': msg.k[5],
            'dist': list(msg.d), 'size': (msg.width, msg.height)}


def scale_intrinsics(intr, width, height):
    """Rescale K when the image size differs from the calibration size."""
    w0, h0 = intr['size']
    if (width, height) == (w0, h0):
        return dict(intr)
    sx, sy = width / w0, height / h0
    out = dict(intr)
    out.update(fx=intr['fx'] * sx, cx=(intr['cx'] + 0.5) * sx - 0.5,
               fy=intr['fy'] * sy, cy=(intr['cy'] + 0.5) * sy - 0.5, size=(width, height))
    return out


class IPM:
    """Flat-ground IPM. The BEV -> image lookup table is built once in rebuild()."""

    def __init__(self, intr, **geom):
        self.fx, self.fy = float(intr['fx']), float(intr['fy'])
        self.cx, self.cy = float(intr['cx']), float(intr['cy'])
        self.img_w, self.img_h = intr['size']
        self.dist = np.zeros(8)
        d = np.asarray(intr['dist'], float).ravel()[:8]
        self.dist[:len(d)] = d
        self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], float)
        g = dict(DEFAULT_GEOMETRY)
        g.update({k: float(v) for k, v in geom.items() if k in DEFAULT_GEOMETRY})
        self.__dict__.update(g)
        self.rebuild()

    # ---------------------------------------------------------------- geometry
    @property
    def R(self):
        """Ground -> camera rotation (yaw, then pitch, then roll)."""
        return (rot_z(math.radians(self.roll_deg)) @ rot_x(math.radians(self.pitch_deg))
                @ rot_y(math.radians(self.yaw_deg)) @ R_GROUND_TO_CAM)

    def _distort(self, x, y):
        """Apply the OpenCV distortion model (k1, k2, p1, p2, k3, k4, k5, k6)."""
        k1, k2, p1, p2, k3, k4, k5, k6 = self.dist
        r2 = x * x + y * y
        r4, r6 = r2 * r2, r2 * r2 * r2
        radial = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + k4 * r2 + k5 * r4 + k6 * r6)
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return self.fx * xd + self.cx, self.fy * yd + self.cy

    def project_ground(self, X, Y):
        """Ground point (X, Y) [m] -> image pixel (u, v), valid flag."""
        X = np.asarray(X, float).ravel()
        Y = np.asarray(Y, float).ravel()
        P = np.stack([X, Y, np.zeros_like(X)], axis=1)
        Pc = (P - np.array([0.0, 0.0, self.cam_height])) @ self.R.T
        z = Pc[:, 2]
        valid = z > 1e-6
        zs = np.where(valid, z, 1.0)
        x, y = Pc[:, 0] / zs, Pc[:, 1] / zs
        # the distortion polynomial can fold far-outside points back in: reject them first
        m = 0.05 * self.img_w
        ui, vi = self.fx * x + self.cx, self.fy * y + self.cy
        valid &= (ui > -m) & (ui < self.img_w + m) & (vi > -m) & (vi < self.img_h + m)
        u, v = self._distort(x, y)
        return u, v, valid

    def rebuild(self):
        """Rebuild the BEV -> image remap table. Call only when parameters change."""
        r = self.resolution
        self.bev_w = max(1, int(round(2 * self.y_half / r)))
        self.bev_h = max(1, int(round((self.x_max - self.x_min) / r)))
        Xs = self.x_max - (np.arange(self.bev_h) + 0.5) * r
        Ys = self.y_half - (np.arange(self.bev_w) + 0.5) * r
        Yg, Xg = np.meshgrid(Ys, Xs)
        u, v, valid = self.project_ground(Xg, Yg)
        self.map_x = np.where(valid, u, -1).astype(np.float32).reshape(self.bev_h, self.bev_w)
        self.map_y = np.where(valid, v, -1).astype(np.float32).reshape(self.bev_h, self.bev_w)
        self.coverage = float(valid.mean())

    def warp(self, img, nearest=False):
        """Image -> BEV. Use nearest=True for 0/255 masks (no in-between values)."""
        interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        return cv2.remap(img, self.map_x, self.map_y, interp,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # ----------------------------------------------------------- conversions
    def pixel_to_ground(self, u, v):
        """Image pixel -> ground (X, Y) [m]. None above the horizon."""
        n = cv2.undistortPoints(np.array([[[u, v]]], np.float64), self.K, self.dist).reshape(2)
        ray = self.R.T @ np.array([n[0], n[1], 1.0])
        if ray[2] >= -1e-9:
            return None
        s = -self.cam_height / ray[2]
        return float(s * ray[0]), float(s * ray[1])

    def ground_to_bev(self, X, Y):
        """Ground [m] -> BEV pixel (u, v)."""
        return (self.y_half - Y) / self.resolution - 0.5, (self.x_max - X) / self.resolution - 0.5

    def bev_to_ground(self, u, v):
        """BEV pixel -> ground [m]."""
        return self.x_max - (v + 0.5) * self.resolution, self.y_half - (u + 0.5) * self.resolution

    def near_distance(self):
        """How far ahead [m] the bottom-centre image pixel hits the floor (check with a ruler)."""
        p = self.pixel_to_ground(self.cx, self.img_h - 1)
        return None if p is None else p[0]


def draw_grid(bev, ipm, step=0.1, label_every=0.5):
    """Distance grid (step m) and vehicle centre line (yellow) on top of a BEV image."""
    vis = cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR) if bev.ndim == 2 else bev.copy()
    h, w = vis.shape[:2]
    n_major = max(1, int(round(label_every / step)))
    i0 = int(math.ceil(ipm.x_min / step - 1e-9))
    i1 = int(math.floor(ipm.x_max / step + 1e-9))
    for i in range(i0, i1 + 1):
        X = i * step
        _, v = ipm.ground_to_bev(X, 0.0)
        v = int(round(v))
        major = (i % n_major == 0)
        cv2.line(vis, (0, v), (w - 1, v), (0, 160, 255) if major else (90, 90, 90), 1)
        if major:
            cv2.putText(vis, f'{X:.1f}m', (3, max(12, v - 3)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 160, 255), 1, cv2.LINE_AA)
    j1 = int(math.floor(ipm.y_half / step + 1e-9))
    for j in range(-j1, j1 + 1):
        u, _ = ipm.ground_to_bev(0.0, j * step)
        u = int(round(u))
        if j == 0:
            color = (0, 255, 255)
        elif j % n_major == 0:
            color = (0, 160, 255)
        else:
            color = (90, 90, 90)
        cv2.line(vis, (u, 0), (u, h - 1), color, 1)
    return vis
