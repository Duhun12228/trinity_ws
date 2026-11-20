#!/usr/bin/env python3
"""
Closed-loop fake world for three_road_node (v3). Not a pytest file: run it by hand.

  ROS_DOMAIN_ID=77 ros2 run trinity_missions three_road_node --ros-args \
      --params-file src/trinity_missions/config/three_road.yaml &
  ROS_DOMAIN_ID=77 python3 src/trinity_missions/test/road3_world.py --layout slalom --side B

The car starts before the entry (lane following = straight at 0.3 m/s), the world fakes:
/odometry/filtered (odom frame != map frame), /scan (LIMO-sized obstacles, ray cast),
/yolopx/drivable_class (3 lanes near the entry), /yolopx/stop_line_mask (the entry line
0.35 m ahead), /lane_follow/yaw, /yolopx/detections (sign A / B), /master/mode (grants the
REQUEST, back to normal on DONE). The car follows /three_road_cmd_vel with the measured
slip (it turns 0.57 x the command). Prints the result; --png saves the driven track.
"""
from array import array
import argparse
import json
import math
import sys

import cv2
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
import rclpy.executors
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, Imu, LaserScan
from std_msgs.msg import Float32MultiArray, String
from trinity_msgs.msg import MasterMode, MissionState
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from trinity_missions import road3_plan as R3

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=ReliabilityPolicy.RELIABLE)
OBS_L, OBS_W = 0.322, 0.22
CAR_F, CAR_B, CAR_W = 0.16, 0.16, 0.11
LAYOUTS = {
    'slalom': [(8.0, 0.94, 0), (7.1, 0.42, 0), (7.1, 1.46, 0), (6.2, 0.94, 1.57),
               (5.4, 1.46, 0), (5.4, 0.42, 0), (4.8, 0.42, 1.57)],
    'none': [],
}


CW_X, CW_R = 9.57, 0.60          # right road lane centre x, curve radius into lane 2
CW_Y1 = R3.MID_Y + CW_R          # curve starts here (heading -y), ends at (CW_X - R, MID_Y)


def lane_ref(p):
    """lane direction and signed offset (+ = car left of the centre) of the lane-follow path:
    right road down to CW_Y1, quarter curve (right turn), lane 2 to -x."""
    x, y = p[0], p[1]
    cx, cy = CW_X - CW_R, CW_Y1
    if y > cy and x > cx:                          # right road, heading -y; left = +x
        return -math.pi / 2, x - CW_X
    if x > cx and y <= cy:                         # curve about (cx, cy), clockwise
        a = math.atan2(y - cy, x - cx)
        return a - math.pi / 2, math.hypot(x - cx, y - cy) - CW_R
    return math.pi, R3.MID_Y - y                   # lane 2, heading -x; left = -y


def lane_k(p):
    x, y = p[0], p[1]
    return -1.0 / CW_R if (x > CW_X - CW_R and y <= CW_Y1) else 0.0


def rect(c, L, W):
    x, y, a = c
    ca, sa = math.cos(a), math.sin(a)
    return np.array([(x + px * ca - py * sa, y + px * sa + py * ca)
                     for px, py in ((-L / 2, -W / 2), (L / 2, -W / 2), (L / 2, W / 2),
                                    (-L / 2, W / 2))])


def car_rect(p):
    x, y, th = p
    c, s = math.cos(th), math.sin(th)
    return np.array([(x + a * c - b * s, y + a * s + b * c)
                     for a, b in ((CAR_F, CAR_W), (CAR_F, -CAR_W), (-CAR_B, -CAR_W),
                                  (-CAR_B, CAR_W))])


class World(Node):

    def __init__(self, a):
        super().__init__('road3_world')
        self.obs = [tuple(o) for o in (json.loads(a.boxes) if a.boxes else LAYOUTS[a.layout])]
        self.side = 1 if a.side == 'A' else 3
        self.slip = a.slip
        self.yaw_bias = a.entry_yaw_bias
        self.lat_drift = a.odom_lat_drift
        self.dist = 0.0
        self.png = a.png
        self.pose = np.array([9.40, 0.94 + a.dy, math.pi + math.radians(a.dth)])
        self.cw = a.crosswalk
        if self.cw:   # standing at the crosswalk on the right road (driving map -y)
            self.pose = np.array([CW_X + a.dx_cw, a.cw_y, -math.pi / 2 + math.radians(a.dth)])
        self.yaw_scale = a.yaw_scale
        self.opose = np.zeros(3)              # odom: integrated with the yaw scale error
        # odom frame: start pose = origin, yaw 0 (anything but the map frame)
        self.o0 = self.pose.copy()
        self.mode, self.seq = 'normal', 0
        self.cmd = (0.0, 0.0)
        self.t = 0.0
        self.trace = [self.pose.copy()]
        self.result = None
        self.rng = np.random.default_rng(a.seed)
        self.pub_odom = self.create_publisher(Odometry, '/odometry/filtered', 10)
        self.pub_imu = self.create_publisher(Imu, '/imu', 10)
        self.pub_scan = self.create_publisher(LaserScan, '/scan', 10)
        self.pub_cls = self.create_publisher(Image, '/yolopx/drivable_class', 10)
        self.pub_line = self.create_publisher(Image, '/yolopx/stop_line_mask', 10)
        self.pub_yaw = self.create_publisher(Float32MultiArray, '/lane_follow/yaw', 10)
        self.pub_lstate = self.create_publisher(String, '/lane_follow/state', 10)
        self.pub_det = self.create_publisher(Detection2DArray, '/yolopx/detections', 10)
        self.pub_mode = self.create_publisher(MasterMode, '/master/mode', LATCHED)
        self.create_subscription(Twist, '/three_road_cmd_vel', self.on_cmd, 10)
        self.create_subscription(MissionState, '/mission/three_road/state', self.on_state, 10)
        self.dbg_png, self.dbg_at = a.debug_png, a.debug_at
        if self.dbg_png:
            self.create_subscription(Image, '/three_road/debug', self.on_debug, 1)
        self.dt = 0.05
        self.create_timer(self.dt, self.step)
        self.create_timer(0.1, self.sensors)
        self.send_mode('crosswalk' if self.cw else 'normal', 'start')

    def send_mode(self, mode, why):
        self.mode = mode
        self.seq += 1
        m = MasterMode()
        m.mode, m.seq, m.reason = mode, self.seq, why
        self.pub_mode.publish(m)
        self.get_logger().info(f'mode -> {mode} ({why})')

    def on_debug(self, msg):
        if self.result is None and self.pose[0] >= self.dbg_at:
            img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
            cv2.imwrite(self.dbg_png, img)

    def on_cmd(self, msg):
        self.cmd = (msg.linear.x, msg.angular.z)

    def on_state(self, msg):
        if msg.state == MissionState.REQUEST and self.mode == 'normal':
            self.send_mode('3road', '3road requested')
        elif msg.state == MissionState.DONE and self.mode == '3road':
            self.finish(msg.reason.startswith('exit'), msg.reason)

    def finish(self, ok, why):
        if self.result is not None:
            return
        tr = np.array(self.trace)
        near = tr[(tr[:, 0] > 2.8) & (tr[:, 0] < 4.0)]
        if ok and len(near):
            got = 3 if near[:, 1].mean() > R3.MID_Y else 1
            if got != self.side:
                ok, why = False, f'wrong road ({"AB"[got == 3]})'
        self.result = (ok, why, self.t)
        self.send_mode('normal', 'done')
        if self.png:
            self.draw()

    def step(self):
        if self.result is not None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = min(now - getattr(self, 'last', now - self.dt), 0.2)   # real time: physics
        self.last = now                                               # does not depend on the rate
        self.t += dt
        self.steps = getattr(self, 'steps', 0) + 1
        if self.mode == '3road':
            v, w = self.cmd
            w = w * self.slip * (1 + self.rng.normal(0, 0.08))
        else:
            # lane follow: hold the lane centre (crosswalk road, curve, lane 2)
            if self.mode == 'crosswalk' and self.t > 2.5:
                self.send_mode('normal', 'crosswalk done')
            v = 0.3 if self.mode == 'normal' else 0.0
            ldir, e = lane_ref(self.pose)        # e: car left of the lane centre
            herr = math.atan2(math.sin(ldir - self.pose[2]), math.cos(ldir - self.pose[2]))
            w = 1.5 * herr - 2.0 * e + v * lane_k(self.pose)
        self.dist += abs(v) * dt
        for _ in range(4):                    # sub-steps
            self.pose[2] += w * dt / 4
            self.pose[0] += v * math.cos(self.pose[2]) * dt / 4
            self.pose[1] += v * math.sin(self.pose[2]) * dt / 4
            self.opose[2] += w * self.yaw_scale * dt / 4
            self.opose[0] += v * math.cos(self.opose[2]) * dt / 4
            self.opose[1] += v * math.sin(self.opose[2]) * dt / 4
        self.trace.append(self.pose.copy())
        cr = car_rect(self.pose)
        if not R3_inside(cr) and not (self.cw and self.pose[0] > CW_X - CW_R
                                      and self.mode != '3road'):
            self.finish(False, f'yellow line at x {self.pose[0]:.2f} y {self.pose[1]:.2f}')
        for o in self.obs:
            r, _ = cv2.intersectConvexConvex(cr.astype(np.float32),
                                             rect(o, OBS_L, OBS_W).astype(np.float32))
            if r > 1e-6:
                self.finish(False, f'hit the obstacle at {o[0]:.2f}, {o[1]:.2f}')
        if self.t > 70.0:
            self.finish(False, 'timeout')
        if self.mode == 'normal' and self.pose[0] < 7.0 and self.result is None:
            self.finish(False, '3road never started')
        self.odom()

    def odom(self):
        m = Odometry()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'odom'
        m.pose.pose.position.x = self.opose[0]
        m.pose.pose.position.y = self.opose[1] + self.lat_drift * self.dist
        yaw = self.opose[2]
        m.pose.pose.orientation.z = math.sin(yaw / 2)
        m.pose.pose.orientation.w = math.cos(yaw / 2)
        self.pub_odom.publish(m)
        i = Imu()                             # true heading (+ a constant offset)
        i.header.stamp = m.header.stamp
        i.orientation.z = math.sin((self.pose[2] + 1.0) / 2)
        i.orientation.w = math.cos((self.pose[2] + 1.0) / 2)
        self.pub_imu.publish(i)

    def sensors(self):
        x, y, th = self.pose
        stamp = self.get_clock().now().to_msg()
        # lidar
        n = 720
        ox, oy = x + 0.103 * math.cos(th), y + 0.103 * math.sin(th)
        ang = th - math.pi + np.arange(n) * 2 * math.pi / n
        dx, dy = np.cos(ang), np.sin(ang)
        best = np.full(n, np.inf)
        for o in self.obs:
            p = rect(o, OBS_L, OBS_W)
            for i in range(4):
                ax, ay = p[i]
                bx, by = p[(i + 1) % 4]
                ex, ey = bx - ax, by - ay
                den = dx * ey - dy * ex
                with np.errstate(divide='ignore', invalid='ignore'):
                    t = ((ax - ox) * ey - (ay - oy) * ex) / den
                    u = ((ax - ox) * dy - (ay - oy) * dx) / den
                hit = (np.abs(den) > 1e-9) & (t > 0) & (u >= 0) & (u <= 1) & (t < best)
                best = np.where(hit, t, best)
        best = np.where(np.isfinite(best), best + self.rng.normal(0, 0.01, n), np.inf)
        s = LaserScan()
        s.header.stamp, s.header.frame_id = stamp, 'laser'
        s.angle_min, s.angle_increment = -math.pi, 2 * math.pi / n
        s.angle_max = s.angle_min + (n - 1) * s.angle_increment
        s.range_min, s.range_max = 0.05, 12.0
        s.ranges = [float(r) for r in best]
        self.pub_scan.publish(s)
        # lane follow yaw: [near deg, ahead deg, lane centre y (+ left)]
        ldir, e = lane_ref(self.pose)
        rel = math.degrees(math.atan2(math.sin(ldir - th), math.cos(ldir - th)))
        cen = -e
        if x < 8.3:
            lane_y = R3.LANE_Y[min((1, 2, 3), key=lambda k: abs(R3.LANE_Y[k] - y))]
            cen = float(y - lane_y)
        elif x < CW_X - CW_R + 0.05 and abs(rel) < 20:
            rel += self.yaw_bias      # biased near the entry (out of the curve before it)
        msg = Float32MultiArray()
        msg.data = [rel, rel, float(cen)]
        self.pub_yaw.publish(msg)
        self.pub_lstate.publish(String(data='both'))
        # drivable classes: 3 lanes once the car is close to the widening
        img = np.zeros((480, 640), np.uint8)
        img[300:, 220:420] = 1
        if x < 8.95:
            img[330:, 0:200] = 2
            img[330:, 440:] = 3
        self.pub_cls.publish(self.mono(img, stamp))
        # entry line 0.35 m ahead at row 0.80 H
        line = np.zeros((480, 640), np.uint8)
        ahead = x - R3.ENTRY_X
        if 0.0 < ahead < 1.2 and abs(rel) < 30:
            row = int(479 * min(0.80 + (0.35 - ahead) * 0.5, 0.99))
            line[max(row - 12, 0):row + 1, 220:420] = 255
        self.pub_line.publish(self.mono(line, stamp))
        # sign on the island tip, seen from 3.5 m
        det = Detection2DArray()
        det.header.stamp = stamp
        if 0.5 < x - R3.ISL_TIP < 3.5:
            d = Detection2D()
            h = ObjectHypothesisWithPose()
            h.hypothesis.class_id = 'sign_A' if self.side == 1 else 'sign_B'
            h.hypothesis.score = 0.8
            d.results.append(h)
            det.detections.append(d)
        self.pub_det.publish(det)

    @staticmethod
    def mono(a, stamp):
        m = Image()
        m.header.stamp = stamp
        m.height, m.width = a.shape
        m.encoding, m.step = 'mono8', a.shape[1]
        m.data = array('B', a.tobytes())
        return m

    def draw(self):
        sc = 110
        W, H = int((R3.X1 - R3.X0) * sc), int((R3.Y1 - R3.Y0) * sc)
        img = np.full((H, W, 3), 60, np.uint8)

        def f(p):
            return (int((p[0] - R3.X0) * sc), int((R3.Y1 - p[1]) * sc))
        cv2.polylines(img, [np.array([f(p) for p in R3.ROAD], np.int32)], True, (0, 200, 255), 2)
        cv2.polylines(img, [np.array([f(p) for p in R3.ISLAND], np.int32)], True,
                      (0, 200, 255), 2)
        for o in self.obs:
            cv2.polylines(img, [np.array([f(p) for p in rect(o, OBS_L, OBS_W)], np.int32)],
                          True, (0, 0, 255), 2)
        tr = np.array(self.trace)
        cv2.polylines(img, [np.array([f(p) for p in tr[:, :2]], np.int32)], False,
                      (0, 255, 0), 2)
        for p in tr[::20]:
            cv2.polylines(img, [np.array([f(q) for q in car_rect(p)], np.int32)], True,
                          (0, 140, 0), 1)
        ok, why, t = self.result
        cv2.putText(img, f'{"OK" if ok else "FAIL"} {why} {t:.1f} s', (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imwrite(self.png, img)


def R3_inside(pts):
    road = R3.ROAD.astype(np.float32)
    isl = R3.ISLAND.astype(np.float32)
    for q in pts:
        q = (float(q[0]), float(q[1]))
        if q[0] > 9.05:
            continue          # behind the window (entry road): fine
        if cv2.pointPolygonTest(road, q, False) < 0 or cv2.pointPolygonTest(isl, q, False) > 0:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layout', default='slalom', choices=sorted(LAYOUTS))
    ap.add_argument('--boxes', default='', help='json [[x, y, yaw], ...] (overrides --layout)')
    ap.add_argument('--side', default='B', choices=['A', 'B'])
    ap.add_argument('--slip', type=float, default=0.57)
    ap.add_argument('--dy', type=float, default=0.0)
    ap.add_argument('--dth', type=float, default=0.0)
    ap.add_argument('--odom-lat-drift', type=float, default=0.0,
                    help='m of odom y error per m driven (odom frame y = road -y here)')
    ap.add_argument('--entry-yaw-bias', type=float, default=0.0,
                    help='deg added to /lane_follow/yaw while x > 8.3 (a biased start heading)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--crosswalk', action='store_true',
                    help='start standing at the crosswalk (mode crosswalk 2.5 s), then the curve')
    ap.add_argument('--cw-y', type=float, default=2.6, help='map y of the crosswalk stop')
    ap.add_argument('--dx-cw', type=float, default=0.0, help='car x offset at the crosswalk')
    ap.add_argument('--yaw-scale', type=float, default=1.0,
                    help='odom turns this x the real turn (gyro scale error)')
    ap.add_argument('--png', default='')
    ap.add_argument('--debug-png', default='',
                    help='save /three_road/debug (last frame with x >= --debug-at)')
    ap.add_argument('--debug-at', type=float, default=0.0)
    a = ap.parse_args()
    rclpy.init()
    w = World(a)
    ex = rclpy.executors.SingleThreadedExecutor()
    ex.add_node(w)
    while rclpy.ok() and w.result is None:
        ex.spin_once(timeout_sec=0.01)
    for _ in range(5):
        rclpy.spin_once(w, timeout_sec=0.05)
    ok, why, t = w.result
    print(f'RESULT {"OK" if ok else "FAIL"}: {why} ({t:.1f} s, {w.steps / max(t, 1e-3):.0f} Hz)')
    w.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
