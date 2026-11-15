"""
Lane follow node.

Pipeline (run, per lane mask)
  /yolopx/lane_mask -> IPM (BEV) -> ROI -> sliding window -> lane paths
  -> drop lines far from the drivable area (drivable mask of the same frame)
  -> lane tracker (single-line side decision / lost handling) -> centre path
  -> Stanley at the path point nearest to the reference point (+ look-ahead)
  -> /stanley_cmd_vel (master_node decides what goes to /cmd_vel)
Reference point: vehicle centre column at the bottom (near) edge of the ROI.
debug_pub=true -> /debug_bev, /debug_roi, /debug/sliding_window

Structure: callbacks only store the newest data; run() (timer) calls the
feature functions in order.
"""
from array import array
from collections import deque
import math
import signal

import cv2
from geometry_msgs.msg import Twist
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, qos_profile_sensor_data, QoSProfile,
                       ReliabilityPolicy)
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32MultiArray, String
from trinity_msgs.msg import MasterMode

from mission.ipm import (DEFAULT_GEOMETRY, DEFAULT_INTRINSICS, IPM, draw_grid,
                         intrinsics_from_camera_info, scale_intrinsics)
from mission.lane_detect import DEFAULT_LANE_PARAMS, SlidingWindowLaneDetector
from mission.lane_filter import DEFAULT_FILTER_PARAMS, DrivableLaneFilter
from mission.lane_tracker import DEFAULT_TRACK_PARAMS, LaneTracker
from mission.stanley import DEFAULT_STANLEY_PARAMS, StanleyController

DEBUG_SCALE = 2  # /debug/sliding_window is upscaled so the text is readable


class LaneFollowNode(Node):

    def __init__(self):
        super().__init__('lane_follow_node')
        self.init_params()
        self.init_state()
        self.init_ros()
        self.get_logger().info(
            f'ready: {self.lane_mask_topic} -> {self.cmd_vel_topic} '
            f'(v {self.stanley_params["base_speed"]:.2f} m/s, debug_pub={self.debug_pub}) | '
            f'cam_height {self.geom["cam_height"]:.3f} m, pitch {self.geom["pitch_deg"]:.1f} deg '
            f'(measure these on the robot)')

    # ------------------------------------------------------------------ init
    def init_params(self):
        self.declare_parameter('debug_pub', True)
        self.declare_parameter('lane_mask_topic', '/yolopx/lane_mask')
        self.declare_parameter('drivable_mask_topic', '/yolopx/drivable_mask')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('cmd_vel_topic', '/stanley_cmd_vel')
        self.declare_parameter('lane_state_topic', '/lane_follow/state')
        # heading of the car vs the centre path [deg] (near, ahead), NaN without a path
        self.declare_parameter('yaw_topic', '/lane_follow/yaw')
        self.declare_parameter('mode_topic', '/master/mode')
        # leaving these master modes (the car is in another lane now) resets the tracking
        self.declare_parameter('reset_after_modes', ['3road'])
        self.declare_parameter('debug_bev_topic', '/debug_bev')
        self.declare_parameter('debug_roi_topic', '/debug_roi')
        self.declare_parameter('debug_sliding_window_topic', '/debug/sliding_window')
        self.declare_parameter('roi_top_ratio', 0.5)  # cut this fraction off the top of the BEV
        self.declare_parameter('run_rate_hz', 30.0)
        self.declare_parameter('grid_step', 0.1)
        for defaults in (DEFAULT_GEOMETRY, DEFAULT_LANE_PARAMS, DEFAULT_FILTER_PARAMS,
                         DEFAULT_TRACK_PARAMS, DEFAULT_STANLEY_PARAMS):
            for k, v in defaults.items():
                self.declare_parameter(k, v)

        p = self.get_parameter
        self.debug_pub = bool(p('debug_pub').value)
        self.lane_mask_topic = p('lane_mask_topic').value
        self.drivable_mask_topic = p('drivable_mask_topic').value
        self.camera_info_topic = p('camera_info_topic').value
        self.cmd_vel_topic = p('cmd_vel_topic').value
        self.lane_state_topic = p('lane_state_topic').value
        self.yaw_topic = p('yaw_topic').value
        self.mode_topic = p('mode_topic').value
        self.reset_after_modes = set(p('reset_after_modes').value)
        self.debug_bev_topic = p('debug_bev_topic').value
        self.debug_roi_topic = p('debug_roi_topic').value
        self.debug_sw_topic = p('debug_sliding_window_topic').value
        self.roi_top_ratio = min(max(float(p('roi_top_ratio').value), 0.0), 0.95)
        self.run_rate_hz = float(p('run_rate_hz').value)
        self.grid_step = float(p('grid_step').value)
        self.geom = {k: float(p(k).value) for k in DEFAULT_GEOMETRY}
        self.lane_params = {k: p(k).value for k in DEFAULT_LANE_PARAMS}
        self.filter_params = {k: p(k).value for k in DEFAULT_FILTER_PARAMS}
        self.track_params = {k: p(k).value for k in DEFAULT_TRACK_PARAMS}
        self.stanley_params = {k: p(k).value for k in DEFAULT_STANLEY_PARAMS}

    def init_state(self):
        self.intr = dict(DEFAULT_INTRINSICS)
        self.intr_from_info = False
        self.ipm = None             # built lazily for the mask resolution
        self.roi_visible = None     # ROI pixels inside the camera view (built with the IPM)
        self.lane_mask_msg = None   # newest lane mask, set by the callback
        self.drivable_msgs = deque(maxlen=10)  # recent drivable masks, matched by stamp
        self.lane_bev = None        # newest BEV lane mask (0/255)
        self.lane_roi = None        # bottom (near) part of lane_bev

        res = self.geom['resolution']
        # ROI column of the vehicle centre (Y = 0), same formula as IPM.ground_to_bev
        self.center_col = self.geom['y_half'] / res - 0.5
        self.lane_detector = SlidingWindowLaneDetector(res, **self.lane_params)
        self.lane_filter = DrivableLaneFilter(res, **self.filter_params)
        self.lane_tracker = LaneTracker(res, self.lane_detector.path_step_px,
                                        **self.track_params)
        self.stanley = StanleyController(**self.stanley_params)
        self.lanes = None           # {'left'/'right': SideResult}, newest detection
        self.master_mode = None     # newest /master/mode
        self.track = None           # TrackResult, newest centre path

    def init_ros(self):
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(Image, self.lane_mask_topic, self.lane_mask_callback,
                                 qos_profile_sensor_data)
        self.create_subscription(Image, self.drivable_mask_topic, self.drivable_mask_callback,
                                 qos_profile_sensor_data)
        self.pub_cmd = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        # tracker status per frame: both / single_left / single_right / lost / stop
        self.pub_lane_state = self.create_publisher(String, self.lane_state_topic, 10)
        self.pub_yaw = self.create_publisher(Float32MultiArray, self.yaw_topic, 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(MasterMode, self.mode_topic, self.mode_callback, latched)
        self.pub_debug_bev = None
        self.pub_debug_roi = None
        self.pub_debug_sw = None
        if self.debug_pub:
            self.pub_debug_bev = self.create_publisher(Image, self.debug_bev_topic, 1)
            self.pub_debug_roi = self.create_publisher(Image, self.debug_roi_topic, 1)
            self.pub_debug_sw = self.create_publisher(Image, self.debug_sw_topic, 1)
        self.create_timer(1.0 / self.run_rate_hz, self.run)

    # ------------------------------------------------------------- callbacks
    def mode_callback(self, msg):
        prev, self.master_mode = self.master_mode, msg.mode
        if prev in self.reset_after_modes and msg.mode != prev:
            self.reset_tracking()
            self.get_logger().info(f'master {prev} -> {msg.mode}: lane tracking reset')

    def reset_tracking(self):
        """Forget the previous lane (side decision, centre EMA, steering smoothing)."""
        self.lane_tracker = LaneTracker(self.geom['resolution'], self.lane_detector.path_step_px,
                                        **self.track_params)
        self.stanley = StanleyController(**self.stanley_params)
        self.track = None

    def camera_info_callback(self, msg):
        if self.intr_from_info:
            return
        intr = intrinsics_from_camera_info(msg)
        if intr is None:
            return
        self.intr = intr
        self.intr_from_info = True
        self.ipm = None  # rebuild with the real calibration
        self.get_logger().info(
            f'camera_info: {msg.width}x{msg.height} fx={intr["fx"]:.1f} '
            f'cx={intr["cx"]:.1f} cy={intr["cy"]:.1f} ({msg.distortion_model})')

    def lane_mask_callback(self, msg):
        self.lane_mask_msg = msg

    def drivable_mask_callback(self, msg):
        self.drivable_msgs.append(msg)

    # ------------------------------------------------------------- functions
    @staticmethod
    def image_msg_to_mask(msg):
        """sensor_msgs/Image (mono8) -> HxW uint8 array, None for other encodings."""
        if msg.encoding.lower() not in ('mono8', '8uc1'):
            return None
        buf = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
        return np.ascontiguousarray(buf[:, :msg.width])

    def get_ipm(self, width, height):
        """IPM for this image size, (re)built when the size or calibration changes."""
        if self.ipm is None or (self.ipm.img_w, self.ipm.img_h) != (width, height):
            self.ipm = IPM(scale_intrinsics(self.intr, width, height), **self.geom)
            self.roi_visible = self.crop_roi(self.ipm.map_x >= 0)
            nd = self.ipm.near_distance()
            near = f'{nd:.3f} m ahead' if nd is not None else 'does not hit the ground'
            self.get_logger().info(
                f'IPM {width}x{height} -> BEV {self.ipm.bev_w}x{self.ipm.bev_h}px '
                f'({self.ipm.resolution * 1000:.0f} mm/px) | image bottom {near} | '
                f'visible {self.ipm.coverage:.0%}')
        return self.ipm

    def warp_to_bev(self, mask):
        """Camera-view mask -> BEV mask (nearest keeps it 0/255)."""
        ipm = self.get_ipm(mask.shape[1], mask.shape[0])
        return ipm.warp(mask, nearest=True)

    def crop_roi(self, bev):
        """Keep the bottom (near) part of the BEV: cut roi_top_ratio off the top."""
        top = int(round(bev.shape[0] * self.roi_top_ratio))
        return bev[top:, :]

    def detect_lanes(self, roi):
        """Sliding window on the ROI -> lane paths."""
        self.lanes = self.lane_detector.detect(roi)
        return self.lanes

    def get_drivable_roi(self, stamp):
        """BEV ROI drivable mask (bool) of the frame with this stamp, None if not received."""
        for m in reversed(self.drivable_msgs):
            if m.header.stamp.sec == stamp.sec and m.header.stamp.nanosec == stamp.nanosec:
                mask = self.image_msg_to_mask(m)
                if mask is None:
                    return None
                return self.crop_roi(self.warp_to_bev(mask)) > 0
        return None

    def filter_by_drivable(self, lanes, drivable):
        """Drop lines far from the drivable area (no-op without a drivable mask)."""
        return self.lane_filter.apply(lanes, self.lane_detector.points, drivable,
                                      self.roi_visible)

    def reference_point(self, roi_h):
        """Vehicle centre column at the near (bottom) edge of the ROI."""
        return np.array([self.center_col, roi_h - 1.0])

    def track_lanes(self, lanes, ref, drivable):
        """Centre path (single-line side decision uses the drivable mask)."""
        self.track = self.lane_tracker.update(lanes, ref, drivable, self.roi_visible)
        return self.track

    def compute_control(self, track, ref):
        """Stanley on the centre path -> (linear speed, angular velocity)."""
        return self.stanley.compute(track.cpath, track.speed_scale, ref,
                                    self.geom['resolution'])

    def publish_cmd(self, v, w):
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)
        self.pub_cmd.publish(cmd)

    def make_debug_sliding_window(self, roi, lanes, track, ref):
        """
        Debug image of the sliding window and the control.

        Virtual line (orange), centre path (magenta), reference point (white),
        nearest (magenta) / look-ahead (cyan) points, numbers.
        """
        vis = self.lane_detector.draw_debug(roi, lanes)
        h, w = roi.shape[:2]
        s = DEBUG_SCALE
        vis = cv2.resize(vis, (w * s, h * s), interpolation=cv2.INTER_NEAREST)

        def draw_path(pts, color, thickness):
            if pts is not None and len(pts) >= 2:
                cv2.polylines(vis, [np.round(pts * s).astype(np.int32)], False, color,
                              thickness, cv2.LINE_AA)

        def dot(pt, color, r):
            if pt is not None:
                cv2.circle(vis, (int(round(pt[0] * s)), int(round(pt[1] * s))), r, color, -1)

        for res in lanes.values():
            draw_path(res.rejected_path, (128, 128, 128), 1)
        if track.status == 'lost':
            draw_path(track.lpath, (0, 165, 255), 1)
            draw_path(track.rpath, (0, 165, 255), 1)
        elif track.virtual == 'left':
            draw_path(track.lpath, (0, 165, 255), 2)
        elif track.virtual == 'right':
            draw_path(track.rpath, (0, 165, 255), 2)
        used = self.stanley.cpath_used
        draw_path(used if used is not None else track.cpath, (255, 0, 255), 2)
        dot(ref, (255, 255, 255), 4)
        dot(self.stanley.near_pt, (255, 0, 255), 4)
        dot(self.stanley.ahead_pt, (255, 255, 0), 4)

        st = self.stanley
        lines = [f'{track.status}  v {st.speed:.2f}  w {st.omega:+.2f}',
                 f'err {st.error * 100:+.1f}cm  yaw {math.degrees(st.yaw):+.1f} '
                 f'(near {math.degrees(st.yaw_near):+.0f} '
                 f'ahead {math.degrees(st.yaw_ahead):+.0f})',
                 f'steer {math.degrees(st.steer):+.1f}']
        dist_txt = []
        for side in ('left', 'right'):
            res = lanes[side]
            if res.drivable_dist is not None:
                mark = ' x' if res.rejected_path is not None else ''
                dist_txt.append(f'{side[0].upper()} {res.drivable_dist * 100:.0f}cm{mark}')
        if dist_txt:
            lines.append('to drivable: ' + '  '.join(dist_txt))
        elif not self.lane_filter.active:
            lines.append('to drivable: filter off (no drivable mask)')
        if track.scores:
            sc = track.scores
            lines.append(f'side d{sc["drivable"]:+.2f} p{sc["position"]:+.2f} '
                         f'c{sc["continuity"]:+.0f} = {sc["total"]:+.2f}')
        for i, t in enumerate(lines):
            cv2.putText(vis, t, (4, 14 + 15 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255, 255, 255), 1, cv2.LINE_AA)
        return vis

    def make_debug_bev(self, bev):
        """BEV mask + distance grid (BGR) for visual checking."""
        return draw_grid(bev, self.ipm, self.grid_step)

    @staticmethod
    def to_image_msg(img, header):
        msg = Image()
        msg.header = header
        msg.height, msg.width = img.shape[:2]
        msg.encoding = 'bgr8' if img.ndim == 3 else 'mono8'
        msg.step = msg.width * (3 if img.ndim == 3 else 1)
        # array.array skips the per-byte check rclpy runs on bytes (tens of ms per image)
        msg.data = array('B', np.ascontiguousarray(img).tobytes())
        return msg

    def publish_debug_bev(self, debug_img, header):
        if self.pub_debug_bev is None:
            return
        self.pub_debug_bev.publish(self.to_image_msg(debug_img, header))

    def publish_debug_roi(self, roi, header):
        if self.pub_debug_roi is None:
            return
        self.pub_debug_roi.publish(self.to_image_msg(roi, header))

    def publish_debug_sliding_window(self, img, header):
        if self.pub_debug_sw is None:
            return
        self.pub_debug_sw.publish(self.to_image_msg(img, header))

    # ------------------------------------------------------------------- run
    def run(self):
        msg = self.lane_mask_msg
        if msg is None:
            return
        self.lane_mask_msg = None  # process each mask once

        lane_mask = self.image_msg_to_mask(msg)
        if lane_mask is None:
            self.get_logger().error(f'lane mask must be mono8, got {msg.encoding}',
                                    throttle_duration_sec=5.0)
            return
        self.lane_bev = self.warp_to_bev(lane_mask)
        self.lane_roi = self.crop_roi(self.lane_bev)
        ref = self.reference_point(self.lane_roi.shape[0])
        lanes = self.detect_lanes(self.lane_roi)
        drivable = self.get_drivable_roi(msg.header.stamp)
        lanes = self.filter_by_drivable(lanes, drivable)
        track = self.track_lanes(lanes, ref, drivable)
        v, w = self.compute_control(track, ref)
        self.publish_cmd(v, w)
        self.pub_lane_state.publish(String(data=track.status))
        self.publish_yaw()

        if self.debug_pub:
            self.publish_debug_bev(self.make_debug_bev(self.lane_bev), msg.header)
            self.publish_debug_roi(self.lane_roi, msg.header)
            self.publish_debug_sliding_window(
                self.make_debug_sliding_window(self.lane_roi, lanes, track, ref), msg.header)

    def publish_yaw(self):
        """[near, ahead] heading of the car vs the centre path in degrees; NaN when stanley
        had no path this frame (lost / stop), so nobody reads a stale angle."""
        st = self.stanley
        if st.near_pt is None:
            data = [float('nan'), float('nan')]
        else:
            data = [math.degrees(st.yaw_near), math.degrees(st.yaw_ahead)]
        self.pub_yaw.publish(Float32MultiArray(data=data))

    def destroy_node(self):
        try:
            self.publish_cmd(0.0, 0.0)  # leave the robot stopped
        except Exception:  # context already gone
            pass
        super().destroy_node()


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


def main(args=None):
    # Handle SIGINT / SIGTERM ourselves so the ROS context is still valid when the
    # stop command is published in destroy_node().
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    node = LaneFollowNode()
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
