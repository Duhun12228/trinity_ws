"""
YOLOPX segmentation + detection node (TensorRT).

Sub : input_topic             sensor_msgs/CompressedImage (/camera/color/image_raw/compressed)
Pub : drivable_mask_topic     Image mono8 0/255  my lane (drivable class 1) minus line pixels
      lane_mask_topic         Image mono8 0/255  lane lines only (no stop line)
      stop_line_mask_topic    Image mono8 0/255  stop line                     (multiclass engine)
      drivable_class_topic    Image mono8 0..3   bg / my lane / left / right   (multiclass engine)
      detections_topic        vision_msgs/Detection2DArray, every frame (also when empty),
                              class_id = class name, bbox in input image pixels (multiclass engine)
      overlay_topic           CompressedImage  my lane green, left sky, right purple, lane red,
                              stop line pink, boxes
Image topics and the overlay are only built when someone subscribes.

Engines: the multiclass engine (tools/export_onnx.py) gives all of the above; a v1 engine
(tools/v1, outputs drivable_mask / lane_mask) still works for the two mask topics.

pipelined=true: while the GPU runs frame N, the CPU post-processes and publishes frame N-1.
Throughput goes up, results are one camera frame (~40 ms) late. Headers always carry the
stamp of the frame the result belongs to.
"""
from array import array
import os
import signal
import time

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from yolopx_line.trt_infer import YolopxTrt

DET_CLASSES = ['traffic_light_red', 'traffic_light_yellow', 'traffic_light_green',
               'sign_pedestrian', 'sign_slow', 'sign_A', 'sign_B', 'sign_parking',
               'cone', 'road_slow_text', 'crosswalk']

# overlay colours (BGR)
C_MY_LANE = (0, 255, 0)
C_LEFT = (235, 206, 135)
C_RIGHT = (200, 50, 160)
C_LANE = (0, 0, 255)
C_STOP = (203, 192, 255)
C_BOX = (0, 255, 255)


class YolopxNode(Node):

    def __init__(self):
        super().__init__('yolopx_node')
        self.declare_parameter('engine_path', 'models/yolopx_mc_480x640_fp16.engine')
        self.declare_parameter('input_topic', '/camera/color/image_raw/compressed')
        self.declare_parameter('overlay_topic', '/yolopx/overlay/compressed')
        self.declare_parameter('drivable_mask_topic', '/yolopx/drivable_mask')
        self.declare_parameter('lane_mask_topic', '/yolopx/lane_mask')
        self.declare_parameter('stop_line_mask_topic', '/yolopx/stop_line_mask')
        self.declare_parameter('drivable_class_topic', '/yolopx/drivable_class')
        self.declare_parameter('detections_topic', '/yolopx/detections')
        self.declare_parameter('det_classes', DET_CLASSES)
        self.declare_parameter('det_conf_thres', 0.3)
        self.declare_parameter('det_iou_thres', 0.45)
        self.declare_parameter('pipelined', True)
        self.declare_parameter('overlay_alpha', 0.5)
        self.declare_parameter('jpeg_quality', 80)
        self.declare_parameter('log_period_sec', 5.0)

        p = self.get_parameter
        engine_path = p('engine_path').value
        if not os.path.isabs(engine_path):
            engine_path = os.path.join(get_package_share_directory('yolopx_line'), engine_path)
        self.pipelined = bool(p('pipelined').value)
        self.alpha = float(p('overlay_alpha').value)
        self.jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, int(p('jpeg_quality').value)]
        self.log_period = float(p('log_period_sec').value)
        self.det_classes = list(p('det_classes').value)
        self.conf_thres = float(p('det_conf_thres').value)
        self.iou_thres = float(p('det_iou_thres').value)

        self.get_logger().info(f'loading engine: {engine_path}')
        self.trt = YolopxTrt(engine_path)
        outs = set(self.trt.outputs)
        if {'drivable_class', 'lane_class', 'det'} <= outs:
            self.multiclass = True
            self.da_name, self.ll_name = 'drivable_class', 'lane_class'
        elif {'drivable_mask', 'lane_mask'} <= outs:
            self.multiclass = False  # v1: 0/1 masks, same "class 1" meaning
            self.da_name, self.ll_name = 'drivable_mask', 'lane_mask'
        else:
            raise RuntimeError(f'unknown engine outputs {sorted(outs)}')
        w, h = self.trt.width, self.trt.height
        self.size = (w, h)
        self.solid = {c: np.full((h, w, 3), c, np.uint8)
                      for c in (C_MY_LANE, C_LEFT, C_RIGHT, C_LANE, C_STOP)}
        for _ in range(3):  # warm-up
            self.trt.infer(np.zeros((h, w, 3), np.uint8))

        # (header, bgr) of the frame currently on the GPU, when pipelined
        self.pending = None

        # depth 1 + best effort: always work on the newest frame, drop the backlog.
        qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                         reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pub_overlay = self.create_publisher(CompressedImage, p('overlay_topic').value, 1)
        self.pub_da = self.create_publisher(Image, p('drivable_mask_topic').value, 1)
        self.pub_ll = self.create_publisher(Image, p('lane_mask_topic').value, 1)
        self.pub_stop = self.pub_da_cls = self.pub_det = None
        if self.multiclass:
            self.pub_stop = self.create_publisher(Image, p('stop_line_mask_topic').value, 1)
            self.pub_da_cls = self.create_publisher(Image, p('drivable_class_topic').value, 1)
            self.pub_det = self.create_publisher(Detection2DArray, p('detections_topic').value, 1)
        self.sub = self.create_subscription(
            CompressedImage, p('input_topic').value, self.on_image, qos)

        self._reset_stats()
        self.get_logger().info(
            f'ready: {p("input_topic").value} -> /yolopx/* (engine {w}x{h}, '
            f'{"multiclass + detection" if self.multiclass else "v1 2-class"}, '
            f'pipelined={self.pipelined})')

    def on_image(self, msg):
        t0 = time.perf_counter()
        bgr = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            self.get_logger().warn('failed to decode image', throttle_duration_sec=5.0)
            return
        if (bgr.shape[1], bgr.shape[0]) != self.size:
            bgr = cv2.resize(bgr, self.size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t1 = time.perf_counter()

        if self.pipelined:
            prev = self.pending
            prev_out = self.trt.wait() if prev is not None else None  # frame N-1 done
            t2 = time.perf_counter()
            self.trt.start(rgb)                                       # frame N to GPU
            self.pending = (msg.header, bgr)
            if prev is not None:
                self._publish(prev[0], prev[1], prev_out)             # overlaps GPU
        else:
            out = self.trt.infer(rgb)
            t2 = time.perf_counter()
            self._publish(msg.header, bgr, out)
        t3 = time.perf_counter()

        self._log_stats(t1 - t0, t2 - t1, t3 - t2)

    # ---------------------------------------------------------------- publish
    def _publish(self, header, bgr, out):
        da, ll = out[self.da_name], out[self.ll_name]
        # my lane with line pixels removed (same as demo.py), as 0/255 masks
        da_m = cv2.bitwise_and(cv2.compare(da, 1, cv2.CMP_EQ), cv2.compare(ll, 0, cv2.CMP_EQ))
        ll_m = cv2.compare(ll, 1, cv2.CMP_EQ)
        if self.pub_da.get_subscription_count() > 0:
            self.pub_da.publish(self._mono8(header, da_m))
        if self.pub_ll.get_subscription_count() > 0:
            self.pub_ll.publish(self._mono8(header, ll_m))

        dets = []
        if self.multiclass:
            if self.pub_stop.get_subscription_count() > 0:
                self.pub_stop.publish(self._mono8(header, cv2.compare(ll, 2, cv2.CMP_EQ)))
            if self.pub_da_cls.get_subscription_count() > 0:
                self.pub_da_cls.publish(self._mono8(header, da))
            dets = self._nms(out['det'])
            self.pub_det.publish(self._det_msg(header, dets))
            self.n_det += len(dets)

        if self.pub_overlay.get_subscription_count() > 0:
            self._publish_overlay(header, bgr, da, ll, dets)

    def _nms(self, det):
        """det [N,6] cx,cy,w,h,score,cls -> [(cls, score, cx, cy, w, h)] after per-class NMS."""
        det = det[det[:, 4] >= self.conf_thres]
        if len(det) == 0:
            return []
        boxes = np.stack([det[:, 0] - det[:, 2] / 2, det[:, 1] - det[:, 3] / 2,
                          det[:, 2], det[:, 3]], axis=1).astype(np.float32)
        keep = cv2.dnn.NMSBoxesBatched(boxes.tolist(), det[:, 4].tolist(),
                                       det[:, 5].astype(np.int32).tolist(),
                                       self.conf_thres, self.iou_thres)
        return [(int(det[i, 5]), float(det[i, 4]), *map(float, det[i, :4]))
                for i in np.asarray(keep).reshape(-1)]

    def _det_msg(self, header, dets):
        arr = Detection2DArray()
        arr.header = header
        for cls, score, cx, cy, w, h in dets:
            d = Detection2D()
            d.header = header
            d.bbox.center.position.x = cx
            d.bbox.center.position.y = cy
            d.bbox.size_x = w
            d.bbox.size_y = h
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = self._name(cls)
            hyp.hypothesis.score = score
            d.results.append(hyp)
            arr.detections.append(d)
        return arr

    def _publish_overlay(self, header, bgr, da, ll, dets):
        color = bgr.copy()
        for mask, c in ((cv2.compare(da, 1, cv2.CMP_EQ), C_MY_LANE),
                        (cv2.compare(da, 2, cv2.CMP_EQ), C_LEFT),
                        (cv2.compare(da, 3, cv2.CMP_EQ), C_RIGHT),
                        (cv2.compare(ll, 1, cv2.CMP_EQ), C_LANE),
                        (cv2.compare(ll, 2, cv2.CMP_EQ), C_STOP)):
            cv2.copyTo(self.solid[c], mask, color)
        overlay = cv2.addWeighted(bgr, 1.0 - self.alpha, color, self.alpha, 0.0)
        for cls, score, cx, cy, w, h in dets:
            p1 = (int(cx - w / 2), int(cy - h / 2))
            p2 = (int(cx + w / 2), int(cy + h / 2))
            cv2.rectangle(overlay, p1, p2, C_BOX, 2)
            cv2.putText(overlay, f'{self._name(cls)} {score:.2f}', (p1[0], max(p1[1] - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_BOX, 1, cv2.LINE_AA)
        ok, jpg = cv2.imencode('.jpg', overlay, self.jpeg_params)
        if ok:
            out = CompressedImage()
            out.header = header
            out.format = 'bgr8; jpeg compressed bgr8'
            out.data = array('B', jpg.tobytes())  # array: skips rclpy per-byte check
            self.pub_overlay.publish(out)

    def _name(self, cls):
        return self.det_classes[cls] if 0 <= cls < len(self.det_classes) else str(cls)

    @staticmethod
    def _mono8(header, mask):
        img = Image()
        img.header = header
        img.height, img.width = mask.shape
        img.encoding = 'mono8'
        img.step = img.width
        img.data = array('B', mask.tobytes())  # bytes would cost ~57 ms (per-byte check)
        return img

    # ------------------------------------------------------------------ stats
    def _reset_stats(self):
        self.stats = {'n': 0, 'decode': 0.0, 'gpu_wait': 0.0, 'post': 0.0}
        self.n_det = 0
        self.t_last_log = time.monotonic()

    def _log_stats(self, dec, gpu, post):
        s = self.stats
        s['n'] += 1
        s['decode'] += dec
        s['gpu_wait'] += gpu
        s['post'] += post
        now = time.monotonic()
        if now - self.t_last_log < self.log_period:
            return
        n = s['n']
        det = f', {self.n_det / n:.1f} det/frame' if self.multiclass else ''
        self.get_logger().info(
            f'{n / (now - self.t_last_log):.1f} fps | decode {s["decode"] / n * 1e3:.1f} ms, '
            f'gpu wait {s["gpu_wait"] / n * 1e3:.1f} ms, post {s["post"] / n * 1e3:.1f} ms{det}')
        self._reset_stats()

    def destroy_node(self):
        if self.pending is not None:
            self.trt.wait()
        self.trt.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YolopxNode()
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
